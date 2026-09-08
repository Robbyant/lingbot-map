"""Plot the raw world-space camera trajectory from poses.json in a top-down (bird's-eye)
2D plane -- no perspective projection through any single frame's camera, so unlike
plot_camera_trajectory.py this can't blow up when the path moves along a camera's own
viewing axis.

Usage:
    python plot_camera_trajectory_topdown.py <results_dir> [--start 0] [--end -1] [--plane xz] [--out traj.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="-1 = last available frame")
    ap.add_argument("--plane", choices=list(PLANE_AXES), default="xz")
    ap.add_argument("--size", type=int, default=900, help="Output canvas size in pixels")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    with open(rdir / "poses.json") as f:
        frames = json.load(f)["frames"]

    end = args.end if args.end >= 0 else frames[-1]["frame"]
    selected = sorted((fr for fr in frames if args.start <= fr["frame"] <= end), key=lambda fr: fr["frame"])
    if not selected:
        raise SystemExit(f"No frames in range [{args.start}, {end}]")

    centers = np.array([fr["extrinsic"] for fr in selected], dtype=np.float64)[:, :, 3]
    ids = [fr["frame"] for fr in selected]

    ax_a, ax_b = PLANE_AXES[args.plane]
    pts = centers[:, [ax_a, ax_b]]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.08
    scale = (1 - 2 * margin) * args.size / span.max()
    center_data = (lo + hi) / 2
    center_canvas = np.array([args.size / 2, args.size / 2])

    def to_canvas(p):
        c = (p - center_data) * scale
        return (int(round(center_canvas[0] + c[0])), int(round(center_canvas[1] - c[1])))

    canvas = np.full((args.size, args.size, 3), 30, dtype=np.uint8)
    cv2.line(canvas, (args.size // 2, 0), (args.size // 2, args.size), (60, 60, 60), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, args.size // 2), (args.size, args.size // 2), (60, 60, 60), 1, cv2.LINE_AA)

    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    canvas_pts = [to_canvas(p) for p in pts]

    for a, b in zip(canvas_pts[:-1], canvas_pts[1:]):
        cv2.line(canvas, a, b, (140, 140, 140), 1, cv2.LINE_AA)

    n = max(len(canvas_pts) - 1, 1)
    for j, p in enumerate(canvas_pts):
        col = tuple(int(c) for c in cmap[int(j / n * 255)])
        cv2.circle(canvas, p, 3, col, -1, cv2.LINE_AA)

    cv2.circle(canvas, canvas_pts[0], 8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"start f{ids[0]}", (canvas_pts[0][0] + 10, canvas_pts[0][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    half = 8
    p_end = canvas_pts[-1]
    cv2.rectangle(canvas, (p_end[0] - half, p_end[1] - half), (p_end[0] + half, p_end[1] + half),
                  (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"end f{ids[-1]}", (p_end[0] + 10, p_end[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    label_a, label_b = args.plane[0].upper(), args.plane[1].upper()
    cv2.putText(canvas, f"plane: {label_a}/{label_b}  ({len(ids)} frames, f{ids[0]}-f{ids[-1]})",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    out = args.out or str(rdir / f"traj_topdown_{args.plane}.png")
    cv2.imwrite(out, canvas)
    print(f"Plotted {len(ids)} camera centers (frames {ids[0]}-{ids[-1]}) in the {args.plane.upper()} plane.")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
