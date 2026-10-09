"""
Self-consistency test: unproject one frame's own RAW depth (depth_raw/*.npy, from
demo.py --export_results) to world points using the codebase's own
depth_to_world_coords_points, then reproject those points back into the SAME frame's
camera. This must land near-exactly back on the original pixel grid if the
extrinsic/intrinsic convention is being used correctly -- validates the reprojection
math independent of any cross-frame point-cloud-file ordering confusion.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.geometry import depth_to_world_coords_points, closed_form_inverse_se3

parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str, help="Folder containing poses.json and depth_raw/")
parser.add_argument("--frame", type=int, default=0)
args = parser.parse_args()

results_dir = Path(args.results_dir)
depth_path = results_dir / "depth_raw" / f"{args.frame:06d}.npy"
assert depth_path.exists(), f"No raw depth at {depth_path} (requires a run with the depth_raw/ export)"
depth = np.load(depth_path)  # (H, W)
h, w = depth.shape

with open(results_dir / "poses.json") as f:
    poses = json.load(f)
entry = [fr for fr in poses["frames"] if fr["frame"] == args.frame][0]
extrinsic = np.array(entry["extrinsic"])  # 3x4, c2w (this is what postprocess() stores)
intrinsic = np.array(entry["intrinsic"])  # 3x3

# depth_to_world_coords_points expects a w2c (cam-from-world) extrinsic -- see its
# docstring. poses.json stores c2w, so invert first using the SAME codebase function
# used elsewhere (closed_form_inverse_se3), rather than hand-rolled inversion.
extrinsic_4x4 = np.eye(4)
extrinsic_4x4[:3, :4] = extrinsic
w2c_4x4 = closed_form_inverse_se3(extrinsic_4x4[None])[0]
w2c = w2c_4x4[:3, :4]

world_points, cam_points, valid_mask = depth_to_world_coords_points(depth, w2c, intrinsic)
print(f"World points shape: {world_points.shape}, valid points: {valid_mask.sum()}/{valid_mask.size}")

# Reproject the SAME world points back through the SAME (w2c, intrinsic) -- this must
# reproduce the original pixel grid almost exactly (mod float rounding).
world_flat = world_points.reshape(-1, 3)
valid_flat = valid_mask.reshape(-1)
R_w2c, t_w2c = w2c[:, :3], w2c[:, 3]
cam_pts = (R_w2c @ world_flat.T).T + t_w2c
fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
u = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
v = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy

yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
orig_u, orig_v = xx.reshape(-1).astype(np.float64), yy.reshape(-1).astype(np.float64)

err = np.sqrt((u[valid_flat] - orig_u[valid_flat]) ** 2 + (v[valid_flat] - orig_v[valid_flat]) ** 2)
print(f"Self-reprojection error (should be ~0 if convention is correct): "
      f"mean={err.mean():.4f}px, max={err.max():.4f}px, median={np.median(err):.4f}px")
if err.mean() < 0.5:
    print("PASS -- reprojection math/convention is correct.")
else:
    print("FAIL -- something is wrong with the extrinsic convention or inversion.")
