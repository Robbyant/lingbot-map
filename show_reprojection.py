"""Render reprojected 3D points onto a frame image.

Same-frame (default): unproject frame S's depth to world points, reproject back into
frame S, and overlay the reprojected pixels (colored by depth) on frame S's image. If
the convention is correct the dots tile the image exactly (this is the visual form of
the self-test "PASS").

Cross-frame (--target T): unproject frame S's depth to world points, then reproject
them into frame T's camera and overlay on frame T's image. Now alignment depends on
depth AND relative pose being correct -- misalignment here reveals real geometry error,
which same-frame reprojection cannot.

Usage:
    python show_reprojection.py <results_dir> --source 0 [--target 20] [--out overlay.png]

Requires a run exported with depth_raw/ and rgb/ (demo.py --export_results).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3


def _load(results_dir, frame):
    d = np.load(Path(results_dir) / "depth_raw" / f"{frame:06d}.npy")
    img = cv2.imread(str(Path(results_dir) / "rgb" / f"{frame:06d}.png"))
    return d, img


def _w2c(c2w_3x4):
    m = np.eye(4)
    m[:3, :4] = np.array(c2w_3x4)
    return closed_form_inverse_se3(m[None])[0][:3, :4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--source", type=int, default=0, help="Frame whose depth is unprojected")
    ap.add_argument("--target", type=int, default=None, help="Frame to reproject INTO (default = source)")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--stride", type=int, default=6, help="Draw every Nth pixel (density)")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    tgt = args.target if args.target is not None else args.source
    with open(rdir / "poses.json") as f:
        frames = json.load(f)["frames"]

    def entry(i):
        return frames[i]

    K = np.array(entry(args.source)["intrinsic"], dtype=np.float64)
    depth, _ = _load(rdir, args.source)
    _, tgt_img = _load(rdir, tgt)
    if tgt_img is None:
        raise SystemExit(f"No rgb/{tgt:06d}.png -- rerun demo.py so the target frame's RGB is exported.")

    # Unproject source depth -> world points (function wants a w2c extrinsic).
    w2c_src = _w2c(entry(args.source)["extrinsic"])
    world, cam, valid = depth_to_world_coords_points(depth, w2c_src, K)
    world = world.reshape(-1, 3)
    depth_flat = depth.reshape(-1)
    valid = valid.reshape(-1)

    # Reproject world points into the TARGET camera.
    w2c_tgt = _w2c(entry(tgt)["extrinsic"])
    R, t = w2c_tgt[:3, :3], w2c_tgt[:3, 3]
    Xc = (R @ world.T + t[:, None]).T          # world -> target camera coords
    z = Xc[:, 2]
    front = z > 1e-6
    uv = (K @ Xc.T).T
    uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)

    H, W = tgt_img.shape[:2]
    on = valid & front & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    # Same-frame reprojection error (only meaningful when target == source).
    # world/uv are in flattened (row-major) pixel order, so compare against the same grid.
    if tgt == args.source:
        Hs, Ws = depth.shape
        gx, gy = np.meshgrid(np.arange(Ws), np.arange(Hs))
        grid = np.stack([gx.reshape(-1), gy.reshape(-1)], 1).astype(np.float64)
        m = valid & front
        err = np.linalg.norm(uv[m] - grid[m], axis=1)
        print(f"Same-frame reprojection error: mean={err.mean():.4f}px max={err.max():.4f}px")

    # Color reprojected dots by source depth (near=red, far=blue via TURBO).
    dsel = depth_flat[on]
    if dsel.size:
        dn = np.clip((dsel - dsel.min()) / (np.ptp(dsel) + 1e-9), 0, 1)
        colors = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    overlay = tgt_img.copy()
    pts = uv[on].astype(int)
    for j in range(0, len(pts), args.stride):
        cv2.circle(overlay, (pts[j, 0], pts[j, 1]), 1, tuple(int(c) for c in colors[j]), -1)

    blended = cv2.addWeighted(tgt_img, 0.45, overlay, 0.55, 0)
    out = args.out or str(rdir / (f"reproj_{args.source}_to_{tgt}.png"))
    cv2.imwrite(out, blended)
    kind = "SAME-frame" if tgt == args.source else f"CROSS-frame ({args.source}->{tgt})"
    print(f"{kind}: drew {on.sum()} reprojected points onto frame {tgt}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
