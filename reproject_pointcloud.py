"""
Reproject the full merged point_cloud.ply back into one frame's camera view (using
that frame's extrinsic/intrinsic from poses.json), and show it side by side with the
actual RGB frame. This tests GLOBAL consistency -- points contributed by every other
frame should still land in roughly the right place when viewed from this frame's pose,
not just this frame's own depth (which would trivially match itself).
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str, help="Folder containing poses.json, point_cloud.ply, rgb/")
parser.add_argument("--frame", type=int, default=0, help="Frame index to reproject into (must exist in rgb/)")
parser.add_argument("--point_size", type=int, default=2)
parser.add_argument("--output", type=str, default=None)
args = parser.parse_args()

results_dir = Path(args.results_dir)
rgb_path = results_dir / "rgb" / f"{args.frame:06d}.png"
assert rgb_path.exists(), f"No RGB frame at {rgb_path} (rgb/ only has every-20th frame from export_results)"
rgb = cv2.imread(str(rgb_path))
h, w = rgb.shape[:2]

with open(results_dir / "poses.json") as f:
    poses = json.load(f)
frame_entries = {fr["frame"]: fr for fr in poses["frames"]}
entry = frame_entries[args.frame]
extrinsic = np.array(entry["extrinsic"])  # 3x4, c2w
intrinsic = np.array(entry["intrinsic"])  # 3x3


def load_ply(path):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert f.readline().strip() == b"format ascii 1.0"
        n_vertex = int(f.readline().split()[-1])
        while f.readline().strip() != b"end_header":
            pass
        data = np.loadtxt(f, max_rows=n_vertex)
    points = data[:, 0:3].astype(np.float64)
    colors = data[:, 3:6].astype(np.uint8)
    return points, colors


points_w, colors = load_ply(results_dir / "point_cloud.ply")
print(f"Loaded {points_w.shape[0]} world points from point_cloud.ply")

# c2w -> w2c: invert the 3x4 [R|t] (rotation transpose, translation re-derived)
R_c2w = extrinsic[:, :3]
t_c2w = extrinsic[:, 3]
R_w2c = R_c2w.T
t_w2c = -R_w2c @ t_c2w

points_c = (R_w2c @ points_w.T).T + t_w2c  # (N, 3), camera space
in_front = points_c[:, 2] > 1e-6
points_c = points_c[in_front]
colors_f = colors[in_front]

fx, fy = intrinsic[0, 0], intrinsic[1, 1]
cx, cy = intrinsic[0, 2], intrinsic[1, 2]
u = fx * points_c[:, 0] / points_c[:, 2] + cx
v = fy * points_c[:, 1] / points_c[:, 2] + cy
in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

u, v = u[in_bounds], v[in_bounds]
depth = points_c[in_bounds, 2]
colors_f = colors_f[in_bounds]
print(f"{u.shape[0]} points project inside the {w}x{h} frame")

# Paint far-to-near so nearer points win at overlapping pixels (poor-man's z-buffer).
order = np.argsort(-depth)
canvas = np.zeros((h, w, 3), dtype=np.uint8)
for i in order:
    cv2.circle(canvas, (int(u[i]), int(v[i])), args.point_size, colors_f[i].tolist(), -1)
canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

divider = np.full((h, 4, 3), 255, dtype=np.uint8)
composite = np.hstack([rgb, divider, canvas])
label_y = 20
cv2.putText(composite, "RGB (actual)", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(composite, "Reprojected point cloud (all frames)", (w + 12, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

output_path = args.output or str(results_dir / f"reprojection_{args.frame:06d}.png")
cv2.imwrite(output_path, composite)
print(f"Wrote {output_path}")
