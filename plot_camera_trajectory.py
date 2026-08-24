"""Reproject the camera trajectory (each frame's camera center) into one reference
frame's view. Shows where the camera moved to/from, as seen from that frame -- e.g. a
forward walk traces a line of dots receding toward the vanishing point.

Usage:
    python plot_camera_trajectory.py <results_dir> [--frame0 0] [--start 0] [--end -1] [--out traj.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.geometry import closed_form_inverse_se3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--frame0", type=int, default=0, help="Frame whose view the trajectory is drawn onto")
    ap.add_argument("--start", type=int, default=0, help="First frame index of the trajectory to plot")
    ap.add_argument("--end", type=int, default=-1, help="Last frame index (inclusive), -1 = last available frame")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--no-line", dest="line", action="store_false", help="Don't connect consecutive points with a polyline")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    with open(rdir / "poses.json") as f:
        frames = json.load(f)["frames"]

    end = args.end if args.end >= 0 else frames[-1]["frame"]
    by_idx = {fr["frame"]: fr for fr in frames}

    f0 = by_idx[args.frame0]
    K0 = np.array(f0["intrinsic"], dtype=np.float64)
    c2w0 = np.eye(4)
    c2w0[:3, :4] = np.array(f0["extrinsic"], dtype=np.float64)
    w2c0 = closed_form_inverse_se3(c2w0[None])[0][:3, :4]
    R0, t0 = w2c0[:, :3], w2c0[:, 3]

    img = cv2.imread(str(rdir / "rgb" / f"{args.frame0:06d}.png"))
    if img is None:
        raise SystemExit(f"No rgb/{args.frame0:06d}.png in {rdir} -- need a run exported with --comparison_stride 1")
    H, W = img.shape[:2]

    frame_ids = sorted(i for i in by_idx if args.start <= i <= end)
    pts_2d, colors, labels = [], [], []
    for i in frame_ids:
        Ci = np.array(by_idx[i]["extrinsic"], dtype=np.float64)[:, 3]
        Xc = R0 @ Ci + t0
        if Xc[2] <= 1e-6:
            continue
        uv = K0 @ Xc
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if 0 <= u < W and 0 <= v < H:
            pts_2d.append((int(round(u)), int(round(v))))
            colors.append(i)
            labels.append(i)

    if not pts_2d:
        raise SystemExit("No camera centers reproject inside frame0's view for the given frame range.")

    lo, hi = min(colors), max(colors)
    span = max(hi - lo, 1)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)

    out_img = img.copy()
    if args.line:
        for a, b in zip(pts_2d[:-1], pts_2d[1:]):
            cv2.line(out_img, a, b, (200, 200, 200), 1, cv2.LINE_AA)
    for (u, v), i in zip(pts_2d, colors):
        col = tuple(int(c) for c in cmap[int((i - lo) / span * 255)])
        cv2.circle(out_img, (u, v), args.radius, col, -1, cv2.LINE_AA)

    for j in (0, len(pts_2d) - 1):
        u, v = pts_2d[j]
        cv2.putText(out_img, f"f{labels[j]}", (u + 5, v - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    out = args.out or str(rdir / f"traj_reproj_{args.frame0}.png")
    cv2.imwrite(out, out_img)
    print(f"Reprojected {len(pts_2d)}/{len(frame_ids)} camera centers (frames {frame_ids[0]}-{frame_ids[-1]}) "
          f"into frame {args.frame0}'s view.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
