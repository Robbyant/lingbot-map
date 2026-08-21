"""Compare lingbot-map's camera extrinsics/intrinsics against classical OpenCV
feature-based visual odometry on the SAME frames.

lingbot-map predicts, per frame, an intrinsic K (3x3) and a camera-to-world (c2w)
extrinsic. OpenCV can independently estimate the *relative* camera motion between
consecutive frames from 2D feature correspondences (ORB -> essential matrix ->
recoverPose), given a calibration K. We:

  1. Use lingbot's K to run OpenCV recoverPose between consecutive frames, giving a
     relative rotation R_cv and unit translation direction t_cv (monocular VO is
     scale-free, so only the translation DIRECTION is comparable, not its length).
  2. Derive lingbot's own relative motion from its c2w extrinsics for the same pair.
  3. Report the rotation geodesic error (deg) and translation-direction angle (deg).

On intrinsics: OpenCV cannot recover an absolute focal length from arbitrary gameplay
frames without a calibration target (chessboard). So we don't "estimate K with OpenCV";
we report lingbot's K/FOV and use it as the calibration the OpenCV VO relies on -- the
extrinsic agreement below is itself an indirect check that K is sane (a badly wrong K
makes the essential-matrix motion disagree with lingbot's poses).

Usage:
    python compare_opencv_poses.py <results_dir> --image_folder <frames> [--num 21]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.load_fn import load_and_preprocess_images


def _rel_from_c2w(c2w_i, c2w_j):
    """Relative camera motion mapping cam_i coords -> cam_j coords (OpenCV convention:
    X_cam_j = R_rel @ X_cam_i + t_rel), derived from two camera-to-world 3x4 poses."""
    def w2c(c2w):
        R = c2w[:3, :3]
        t = c2w[:3, 3]
        Rw = R.T
        tw = -Rw @ t
        return Rw, tw
    Rwi, twi = w2c(c2w_i)
    Rwj, twj = w2c(c2w_j)
    R_rel = Rwj @ Rwi.T
    t_rel = twj - R_rel @ twi
    return R_rel, t_rel


def _rot_geodesic_deg(Ra, Rb):
    R = Ra @ Rb.T
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _dir_angle_deg(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    # abs() because monocular translation has an inherent sign/scale ambiguity.
    c = abs(float(a @ b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str, help="Folder with lingbot poses.json")
    ap.add_argument("--image_folder", type=str, required=True,
                    help="Folder of source frames matching the poses (frame order)")
    ap.add_argument("--num", type=int, default=21, help="Number of frames to compare")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    args = ap.parse_args()

    with open(Path(args.results_dir) / "poses.json") as f:
        data = json.load(f)
    frames = data["frames"]
    K = np.array(frames[0]["intrinsic"], dtype=np.float64)
    c2ws = [np.array(fr["extrinsic"], dtype=np.float64) for fr in frames]
    n = min(args.num, len(c2ws))

    print("=" * 68)
    print("INTRINSICS (lingbot-map predicts these; OpenCV needs a calibration")
    print("target to recover K independently, so we report + use lingbot's):")
    print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
    print(f"  fov_h={frames[0].get('fov_h_deg', float('nan')):.2f} deg")
    print("=" * 68)

    # Preprocess the SAME way the model saw them, so pixel coords match K.
    paths = sorted(Path(args.image_folder).glob("*.png")) + sorted(Path(args.image_folder).glob("*.jpg"))
    paths = [str(p) for p in paths[:n]]
    imgs = load_and_preprocess_images(paths, mode="crop",
                                      image_size=args.image_size, patch_size=args.patch_size)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    grays = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs]

    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    rot_errs, trans_errs, used = [], [], 0
    print(f"\n{'pair':>7} {'matches':>8} {'inliers':>8} {'rot_err_deg':>12} {'trans_dir_deg':>14}")
    for i in range(n - 1):
        k1, d1 = orb.detectAndCompute(grays[i], None)
        k2, d2 = orb.detectAndCompute(grays[i + 1], None)
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            print(f"{i:>3}->{i+1:<3} {'--':>8} {'--':>8}   (too few features)")
            continue
        matches = bf.match(d1, d2)
        if len(matches) < 8:
            print(f"{i:>3}->{i+1:<3} {len(matches):>8} {'--':>8}   (too few matches)")
            continue
        pts1 = np.float64([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float64([k2[m.trainIdx].pt for m in matches])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None or E.shape != (3, 3):
            print(f"{i:>3}->{i+1:<3} {len(matches):>8} {'--':>8}   (no essential matrix)")
            continue
        n_in, R_cv, t_cv, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        R_rel, t_rel = _rel_from_c2w(c2ws[i], c2ws[i + 1])
        rot_e = _rot_geodesic_deg(R_cv, R_rel)
        tr_e = _dir_angle_deg(t_cv, t_rel)
        rot_errs.append(rot_e)
        trans_errs.append(tr_e)
        used += 1
        print(f"{i:>3}->{i+1:<3} {len(matches):>8} {int(n_in):>8} {rot_e:>12.3f} {tr_e:>14.3f}")

    print("-" * 68)
    if used:
        ra = np.array(rot_errs)
        ta = np.array(trans_errs)
        print(f"Rotation error (deg)      : mean={ra.mean():.3f}  median={np.median(ra):.3f}  max={ra.max():.3f}")
        print(f"Translation dir err (deg) : mean={np.nanmean(ta):.3f}  median={np.nanmedian(ta):.3f}  max={np.nanmax(ta):.3f}")
        print(f"Compared {used}/{n-1} consecutive pairs.")
        print("\nNote: monocular VO recovers translation only up to scale+sign, so only the")
        print("translation DIRECTION is compared. Small rotation errors + small translation-")
        print("direction errors mean OpenCV's feature geometry agrees with lingbot's extrinsics")
        print("under lingbot's intrinsic K.")
    else:
        print("No comparable pairs (feature matching failed on all pairs).")


if __name__ == "__main__":
    main()
