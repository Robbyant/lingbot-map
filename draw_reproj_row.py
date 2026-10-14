"""Draw a row of N consecutive frames, each with the SOURCE frame's 3D points reprojected
into it (colored by depth). Shows how a single frame's reconstruction tracks across the
sequence -- drift accumulates left-to-right if depth/pose are imperfect.

Requires per-frame RGB + the source frame's raw depth, i.e. a run exported with
`--comparison_stride 1` so every frame has rgb/ and depth_raw/.

Usage:
    python draw_reproj_row.py <results_dir> [--source 0] [--count 10] [--out row.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3


def _w2c(c2w_3x4):
    m = np.eye(4)
    m[:3, :4] = np.array(c2w_3x4, dtype=np.float64)
    return closed_form_inverse_se3(m[None])[0][:3, :4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--source", type=int, default=0)
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--stride", type=int, default=4, help="Draw every Nth reprojected pixel")
    ap.add_argument("--tile_h", type=int, default=200, help="Height of each tile in the row")
    ap.add_argument("--mode", choices=["own", "track", "accumulate"], default="own",
                    help="own: each tile = THAT frame's own point cloud reprojected into itself "
                         "(per-frame depth). track: source frame's cloud reprojected into each frame "
                         "(pose drift). accumulate: each frame's cloud reprojected into the source "
                         "camera (multi-frame fusion consistency).")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    with open(rdir / "poses.json") as f:
        frames = json.load(f)["frames"]

    def cloud_of(fi):
        """Unproject frame fi's own depth to world points (+ its depth + valid mask)."""
        K = np.array(frames[fi]["intrinsic"], dtype=np.float64)
        d = np.load(rdir / "depth_raw" / f"{fi:06d}.npy")
        w, _, v = depth_to_world_coords_points(d, _w2c(frames[fi]["extrinsic"]), K)
        return w.reshape(-1, 3), d.reshape(-1), v.reshape(-1)

    # For 'track' the same source cloud is reused every tile.
    src_cloud = cloud_of(args.source) if args.mode == "track" else None

    tiles = []
    for t in range(args.source, args.source + args.count):
        # Which cloud, which camera to project into, and which image is the backdrop.
        if args.mode == "own":
            world, dflat, valid = cloud_of(t)
            cam_i, bg_i = t, t                      # own cloud into its own view
        elif args.mode == "track":
            world, dflat, valid = src_cloud
            cam_i, bg_i = t, t                      # source cloud into frame t's view
        else:  # accumulate
            world, dflat, valid = cloud_of(t)
            cam_i, bg_i = args.source, args.source  # each frame's cloud into the source view

        img = cv2.imread(str(rdir / "rgb" / f"{bg_i:06d}.png"))
        if img is None:
            print(f"  frame {bg_i}: no rgb/{bg_i:06d}.png -- rerun demo.py with --comparison_stride 1")
            continue
        H, W = img.shape[:2]
        K = np.array(frames[cam_i]["intrinsic"], dtype=np.float64)
        w2c = _w2c(frames[cam_i]["extrinsic"])
        Xc = (w2c[:3, :3] @ world.T + w2c[:3, 3:4]).T
        z = Xc[:, 2]
        uv = (K @ Xc.T).T
        uv = uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None)
        on = valid & (z > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

        overlay = img.copy()
        ds = dflat[on]
        if ds.size:
            dn = np.clip((ds - ds.min()) / (np.ptp(ds) + 1e-9), 0, 1)
            cols = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
            pts = uv[on].astype(int)
            for j in range(0, len(pts), args.stride):
                cv2.circle(overlay, (pts[j, 0], pts[j, 1]), 1, tuple(int(c) for c in cols[j]), -1)
        tile = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
        cv2.putText(tile, f"f{t} ({int(on.sum())})", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        scale = args.tile_h / tile.shape[0]
        tiles.append(cv2.resize(tile, (int(tile.shape[1] * scale), args.tile_h)))

    if not tiles:
        raise SystemExit("No frames drawn -- need rgb/ for consecutive frames (--comparison_stride 1).")

    row = np.hstack(tiles)
    out = args.out or str(rdir / f"reproj_row_{args.mode}_{args.source}_x{len(tiles)}.png")
    cv2.imwrite(out, row)
    desc = {"own": "each frame's OWN point cloud reprojected into itself",
            "track": f"frame {args.source}'s cloud reprojected into each frame",
            "accumulate": f"each frame's cloud reprojected into frame {args.source}"}[args.mode]
    print(f"Drew {len(tiles)} frames -- mode={args.mode} ({desc}).")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
