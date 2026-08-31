"""
Independent OpenCV-only estimate of horizontal FOV (no calibration target, no
assumed ground truth) via a focal-length sweep + 3-view reprojection-error check:

For each candidate fx: triangulate 3D points from frames (i, i+1) using the essential
matrix + recoverPose, then solvePnP frame (i+2)'s pose from those same 3D points
against their 2D detections in frame i+2. A wrong fx makes the triangulated points
geometrically inconsistent with frame i+2's actual observations, inflating PnP
reprojection error -- the fx that minimizes mean reprojection error across many
triplets is OpenCV's best self-calibration estimate.

Caveats: no bundle adjustment / joint refinement, ORB features only, assumes a static
scene (no independently-moving objects) between the 3 frames of each triplet.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, required=True)
parser.add_argument("--num_frames", type=int, default=150, help="Subset of frames to use (for speed)")
parser.add_argument("--triplet_step", type=int, default=3, help="Frame stride within each (i, i+step, i+2*step) triplet")
parser.add_argument("--fov_min", type=float, default=30.0)
parser.add_argument("--fov_max", type=float, default=110.0)
parser.add_argument("--fov_step", type=float, default=5.0)
parser.add_argument("--poses_json", type=str, default=None,
                     help="Optional poses.json to print the model's own per-frame FOV alongside the OpenCV estimate")
args = parser.parse_args()

model_fov_by_frame = {}
if args.poses_json:
    with open(args.poses_json) as f:
        for fr in json.load(f)["frames"]:
            if fr.get("fov_h_deg") is not None:
                model_fov_by_frame[fr["frame"]] = fr["fov_h_deg"]

image_paths = sorted(Path(args.image_folder).glob("*.png")) + sorted(Path(args.image_folder).glob("*.jpg"))
image_paths = sorted(image_paths)[: args.num_frames]
assert len(image_paths) >= 2 * args.triplet_step + 1, "Not enough frames for the requested triplet_step"

sample = cv2.imread(str(image_paths[0]))
h, w = sample.shape[:2]

orb = cv2.ORB_create(nfeatures=3000)
matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

gray = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in image_paths]
kp_des = [orb.detectAndCompute(g, None) for g in gray]


def match_pair(a, b):
    kp1, des1 = kp_des[a]
    kp2, des2 = kp_des[b]
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return None
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 15:
        return None
    return kp1, kp2, good


def track_triplet(i0, i1, i2):
    """Find keypoints tracked across all 3 frames via index-chained pairwise matches."""
    m01 = match_pair(i0, i1)
    m12 = match_pair(i1, i2)
    if m01 is None or m12 is None:
        return None
    kp0, kp1, good01 = m01
    kp1b, kp2, good12 = m12
    idx1_to_pt0 = {m.trainIdx: m.queryIdx for m in good01}
    idx1_to_pt2 = {m.queryIdx: m.trainIdx for m in good12}
    common = set(idx1_to_pt0.keys()) & set(idx1_to_pt2.keys())
    if len(common) < 15:
        return None
    pts0 = np.float32([kp0[idx1_to_pt0[i]].pt for i in common])
    pts1 = np.float32([kp1[i].pt for i in common])
    pts2 = np.float32([kp2[idx1_to_pt2[i]].pt for i in common])
    return pts0, pts1, pts2


triplets = []  # list of (start_frame_idx, pts0, pts1, pts2)
for i0 in range(0, len(image_paths) - 2 * args.triplet_step, args.triplet_step):
    t = track_triplet(i0, i0 + args.triplet_step, i0 + 2 * args.triplet_step)
    if t is not None:
        triplets.append((i0,) + t)
print(f"Built {len(triplets)} usable 3-view tracks (of {len(image_paths) // args.triplet_step} candidate triplets).")


def reproj_error_for_triplet(pts0, pts1, pts2, fx):
    """Mean PnP reprojection error (px) in frame 2, for one triplet at one candidate fx."""
    K = np.array([[fx, 0, w / 2], [0, fx, h / 2], [0, 0, 1]], dtype=np.float64)
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        return None
    _, R, t, mask_pose = cv2.recoverPose(E, pts0, pts1, K, mask=mask)
    inliers = mask_pose.ravel().astype(bool)
    if inliers.sum() < 8:
        return None
    P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = K @ np.hstack([R, t])
    pts4d = cv2.triangulatePoints(P0, P1, pts0[inliers].T, pts1[inliers].T)
    pts3d = (pts4d[:3] / pts4d[3]).T
    valid = np.isfinite(pts3d).all(axis=1) & (pts4d[3] != 0)
    if valid.sum() < 8:
        return None
    ok, rvec, tvec, pnp_inliers = cv2.solvePnPRansac(
        pts3d[valid].astype(np.float32), pts2[inliers][valid].astype(np.float32), K, None,
        reprojectionError=8.0,
    )
    if not ok or pnp_inliers is None or len(pnp_inliers) < 6:
        return None
    proj, _ = cv2.projectPoints(pts3d[valid], rvec, tvec, K, None)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj[pnp_inliers.ravel()] - pts2[inliers][valid][pnp_inliers.ravel()], axis=1)
    return float(err.mean())


fov_candidates = np.arange(args.fov_min, args.fov_max + 1e-6, args.fov_step)
print(f"\nPer-frame OpenCV FOV estimate (sweeping {args.fov_min}-{args.fov_max} deg, "
      f"step {args.fov_step}, image {w}x{h}):")
print(f"{'frame':>6}  {'opencv_fov_deg':>14}  {'reproj_err_px':>13}  {'model_fov_deg':>13}")

opencv_fovs = []
for frame_idx, pts0, pts1, pts2 in triplets:
    scored = []
    for fov in fov_candidates:
        fx = (w / 2) / math.tan(math.radians(fov / 2))
        err = reproj_error_for_triplet(pts0, pts1, pts2, fx)
        if err is not None:
            scored.append((fov, err))
    if not scored:
        print(f"{frame_idx:6d}  {'N/A':>14}  {'N/A':>13}  "
              f"{model_fov_by_frame.get(frame_idx, 'N/A'):>13}")
        continue
    best_fov, best_err = min(scored, key=lambda s: s[1])
    opencv_fovs.append(best_fov)
    model_fov = model_fov_by_frame.get(frame_idx)
    model_str = f"{model_fov:.1f}" if model_fov is not None else "N/A"
    print(f"{frame_idx:6d}  {best_fov:14.1f}  {best_err:13.3f}  {model_str:>13}")

if opencv_fovs:
    arr = np.array(opencv_fovs)
    print(f"\nOpenCV self-calibration FOV across {len(arr)} frames: "
          f"mean={arr.mean():.1f} deg, std={arr.std():.1f} deg, "
          f"range=[{arr.min():.1f}, {arr.max():.1f}] deg")
else:
    print("\nNo triplet produced a valid estimate at any candidate FOV.")
