"""
Independent sanity check on lingbot-map's estimated camera pose (extrinsics), using
classical OpenCV two-view geometry (ORB features + essential matrix + recoverPose)
across consecutive frames.

Note: this does NOT independently estimate intrinsics -- that needs a calibration
target (checkerboard) this footage doesn't have. Instead it assumes intrinsics from a
known ground-truth FOV (e.g. CS:GO's fixed 90 deg, --fov_deg) to build K, then compares
the resulting *relative rotation per frame pair* against the model's own extrinsics
from poses.json. Translation is scale-ambiguous in monocular two-view geometry, so only
rotation magnitude is compared, not translation.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True, help="Folder of numbered source frames")
parser.add_argument("--poses_json", type=str, required=True, help="poses.json from demo.py --export_results")
parser.add_argument("--fov_deg", type=float, default=90.0, help="Assumed ground-truth horizontal FOV for OpenCV's K")
parser.add_argument("--max_pairs", type=int, default=None, help="Limit number of consecutive-frame pairs (for speed)")
args = parser.parse_args()

with open(args.poses_json) as f:
    poses = json.load(f)
frames_meta = poses["frames"]

image_paths = sorted(Path(args.image_folder).glob("*.png")) + sorted(Path(args.image_folder).glob("*.jpg"))
image_paths = sorted(image_paths)[: len(frames_meta)]
assert len(image_paths) >= 2, f"Need at least 2 frames, found {len(image_paths)}"

sample = cv2.imread(str(image_paths[0]))
h, w = sample.shape[:2]
fx = (w / 2) / math.tan(math.radians(args.fov_deg / 2))
K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
print(f"Assumed intrinsics from {args.fov_deg} deg FOV, image {w}x{h}: fx=fy={fx:.1f}")

def _model_fov_h_deg(frame_entry):
    if frame_entry.get("fov_h_deg") is not None:
        return frame_entry["fov_h_deg"]
    fx_model = frame_entry["intrinsic"][0][0]  # older poses.json without fov_h_deg -- derive from intrinsic
    return math.degrees(2 * math.atan((w / 2) / fx_model)) if fx_model > 0 else None

model_fovs = [_model_fov_h_deg(f) for f in frames_meta]
model_fovs = [v for v in model_fovs if v is not None]
if model_fovs:
    print(f"Model's own estimated FOV: mean={np.mean(model_fovs):.1f} deg, "
          f"range=[{min(model_fovs):.1f}, {max(model_fovs):.1f}] deg  "
          f"(vs assumed ground truth: {args.fov_deg:.1f} deg)")

orb = cv2.ORB_create(nfeatures=2000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)


def relative_rotation_deg_from_c2w(c2w_a, c2w_b):
    """Rotation angle (deg) of the camera motion from pose a to pose b (both 3x4 c2w)."""
    Ra = np.array(c2w_a)[:, :3]
    Rb = np.array(c2w_b)[:, :3]
    R_rel = Rb @ Ra.T
    rvec, _ = cv2.Rodrigues(R_rel)
    return math.degrees(np.linalg.norm(rvec))


num_pairs = len(image_paths) - 1
if args.max_pairs is not None:
    num_pairs = min(num_pairs, args.max_pairs)

opencv_angles, model_angles = [], []
prev_img = cv2.imread(str(image_paths[0]), cv2.IMREAD_GRAYSCALE)
prev_kp, prev_des = orb.detectAndCompute(prev_img, None)

for i in range(num_pairs):
    img = cv2.imread(str(image_paths[i + 1]), cv2.IMREAD_GRAYSCALE)
    kp, des = orb.detectAndCompute(img, None)

    if prev_des is None or des is None or len(prev_des) < 8 or len(des) < 8:
        opencv_angles.append(None)
    else:
        matches = matcher.knnMatch(prev_des, des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            opencv_angles.append(None)
        else:
            pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good])
            pts2 = np.float32([kp[m.trainIdx].pt for m in good])
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is None:
                opencv_angles.append(None)
            else:
                _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
                rvec, _ = cv2.Rodrigues(R)
                opencv_angles.append(math.degrees(np.linalg.norm(rvec)))

    model_angles.append(relative_rotation_deg_from_c2w(
        frames_meta[i]["extrinsic"], frames_meta[i + 1]["extrinsic"]
    ))

    prev_kp, prev_des = kp, des

valid = [(o, m) for o, m in zip(opencv_angles, model_angles) if o is not None]
print(f"\n{len(valid)}/{num_pairs} frame pairs had enough matches for OpenCV pose recovery.")
if valid:
    o_arr = np.array([v[0] for v in valid])
    m_arr = np.array([v[1] for v in valid])
    diff = np.abs(o_arr - m_arr)
    print(f"Per-frame relative rotation (deg): OpenCV mean={o_arr.mean():.2f} std={o_arr.std():.2f}  "
          f"|  Model mean={m_arr.mean():.2f} std={m_arr.std():.2f}")
    print(f"Mean absolute difference: {diff.mean():.2f} deg  (median {np.median(diff):.2f} deg)")
    corr = np.corrcoef(o_arr, m_arr)[0, 1] if len(o_arr) > 1 else float("nan")
    print(f"Correlation between OpenCV and model per-frame rotation magnitude: {corr:.3f}")
