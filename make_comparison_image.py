"""
Build a side-by-side RGB | depth comparison image for one frame from a demo.py
--export_results output folder, with per-frame camera stats (fx, fy, FOV, confidence)
overlaid as text. Only works for frame indices that exist in rgb/ (every 20th, per
export_results) -- pass --frame to pick which one.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("results_dir", type=str, help="Folder containing poses.json, rgb/, depth/")
parser.add_argument("--frame", type=int, default=0, help="Frame index (must exist in rgb/, i.e. a multiple of 20)")
parser.add_argument("--output", type=str, default=None, help="Output path (default: <results_dir>/comparison_<frame>.png)")
args = parser.parse_args()

results_dir = Path(args.results_dir)
rgb_path = results_dir / "rgb" / f"{args.frame:06d}.png"
depth_path = results_dir / "depth" / f"{args.frame:06d}.png"
assert rgb_path.exists(), f"No RGB frame at {rgb_path} (rgb/ only has every-20th frame from export_results)"
assert depth_path.exists(), f"No depth frame at {depth_path}"

with open(results_dir / "poses.json") as f:
    poses = json.load(f)
frame_entries = {fr["frame"]: fr for fr in poses["frames"]}
entry = frame_entries.get(args.frame, {})
conf_stats = poses.get("point_cloud_confidence", {})

rgb = cv2.imread(str(rgb_path))
depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
depth_colored = cv2.applyColorMap(depth, cv2.COLORMAP_MAGMA)

h = rgb.shape[0]
if depth_colored.shape[0] != h:
    depth_colored = cv2.resize(depth_colored, (int(depth_colored.shape[1] * h / depth_colored.shape[0]), h))

divider = np.full((h, 4, 3), 255, dtype=np.uint8)
composite = np.hstack([rgb, divider, depth_colored])

lines = [f"frame {args.frame}"]
if "fx" in entry:
    lines.append(f"fx={entry['fx']:.1f}  fy={entry['fy']:.1f}")
if entry.get("fov_h_deg") is not None:
    lines.append(f"FOV_h={entry['fov_h_deg']:.1f} deg")
if entry.get("depth_conf_mean") is not None:
    lines.append(f"depth_conf_mean={entry['depth_conf_mean']:.3f}")
if entry.get("translation_delta") is not None:
    lines.append(f"translation_delta={entry['translation_delta']:.4f}")
if conf_stats:
    lines.append(f"pc_conf_mean={conf_stats.get('all_points_conf_mean', float('nan')):.3f}  "
                 f"kept={conf_stats.get('kept_fraction', float('nan')) * 100:.1f}%")

pad_top = 22 * (len(lines) + 1)
canvas = np.full((composite.shape[0] + pad_top, composite.shape[1], 3), 30, dtype=np.uint8)
canvas[pad_top:, :] = composite
for i, line in enumerate(lines):
    cv2.putText(canvas, line, (10, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

label_y = pad_top + 18
cv2.putText(canvas, "RGB", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
cv2.putText(canvas, "Depth", (rgb.shape[1] + 12, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

output_path = args.output or str(results_dir / f"comparison_{args.frame:06d}.png")
cv2.imwrite(output_path, canvas)
print(f"Wrote {output_path}")
