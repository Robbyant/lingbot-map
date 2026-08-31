"""Compare lingbot-map's predicted FOV against an OpenCV self-calibration FOV estimate.

OpenCV can't read an absolute focal off a single frame, but from the FUNDAMENTAL matrix
between two views it can self-calibrate the focal: for the correct focal f, the essential
matrix E = KᵀFK must have singular values (sigma, sigma, 0). We grid-search the focal that
best satisfies that (principal point fixed at the image centre, as lingbot's K also has),
convert to a horizontal FOV, and compare to lingbot's per-frame FOV.

Self-calibration from two views is inherently noisy (rotation-dominant or small-baseline
pairs are near-degenerate), so treat the OpenCV FOV as a rough cross-check, not ground truth.

Usage:
    python compare_opencv_fov.py <results_dir> --image_folder <frames> [--num 21]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.load_fn import load_and_preprocess_images


def _focal_from_F(F, cx, cy, f_lo=150.0, f_hi=1600.0, steps=290):
    """Grid-search focal so E=KᵀFK is closest to a valid essential matrix (s0≈s1, s2≈0).
    Returns (best_focal, best_cost) or (nan, nan) if F is unusable."""
    if F is None or F.shape != (3, 3):
        return float("nan"), float("nan")
    best_f, best_c = float("nan"), np.inf
    for f in np.linspace(f_lo, f_hi, steps):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        E = K.T @ F @ K
        s = np.linalg.svd(E, compute_uv=False)
        if s[0] < 1e-9:
            continue
        cost = abs(s[0] - s[1]) / s[0] + s[2] / s[0]   # 0 for a perfect essential matrix
        if cost < best_c:
            best_c, best_f = cost, float(f)
    return best_f, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--image_folder", type=str, required=True)
    ap.add_argument("--num", type=int, default=21)
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    args = ap.parse_args()

    with open(Path(args.results_dir) / "poses.json") as f:
        frames = json.load(f)["frames"]
    n = min(args.num, len(frames))
    K0 = np.array(frames[0]["intrinsic"], dtype=np.float64)
    W, H = 2 * K0[0, 2], 2 * K0[1, 2]
    cx, cy = K0[0, 2], K0[1, 2]

    def fov_h(focal):
        return float(np.degrees(2 * np.arctan(W / (2 * focal))))

    paths = sorted(Path(args.image_folder).glob("*.png")) + sorted(Path(args.image_folder).glob("*.jpg"))
    paths = [str(p) for p in paths[:n]]
    imgs = load_and_preprocess_images(paths, mode="crop",
                                      image_size=args.image_size, patch_size=args.patch_size)
    imgs = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    grays = [cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) for im in imgs]

    orb = cv2.ORB_create(nfeatures=4000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    print(f"processed image {W:.0f}x{H:.0f}, principal point ({cx:.0f},{cy:.0f})")
    print(f"\n{'pair':>7} {'inliers':>7} {'lingbot_FOV':>12} {'opencv_FOV':>11} {'diff_deg':>9} {'fitcost':>8}")
    ling, cvv = [], []
    for i in range(n - 1):
        k1, d1 = orb.detectAndCompute(grays[i], None)
        k2, d2 = orb.detectAndCompute(grays[i + 1], None)
        lf = fov_h(np.array(frames[i]["intrinsic"])[0, 0])
        if d1 is None or d2 is None:
            print(f"{i:>3}->{i+1:<3} {'--':>7} {lf:>12.2f} {'--':>11}")
            continue
        matches = bf.match(d1, d2)
        if len(matches) < 15:
            print(f"{i:>3}->{i+1:<3} {len(matches):>7} {lf:>12.2f} {'--':>11}  (few matches)")
            continue
        pts1 = np.float64([k1[m.queryIdx].pt for m in matches])
        pts2 = np.float64([k2[m.trainIdx].pt for m in matches])
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.999)
        n_in = int(mask.sum()) if mask is not None else 0
        of, cost = _focal_from_F(F, cx, cy)
        if not np.isfinite(of):
            print(f"{i:>3}->{i+1:<3} {n_in:>7} {lf:>12.2f} {'--':>11}  (no F)")
            continue
        ov = fov_h(of)
        ling.append(lf); cvv.append(ov)
        print(f"{i:>3}->{i+1:<3} {n_in:>7} {lf:>12.2f} {ov:>11.2f} {ov-lf:>9.2f} {cost:>8.3f}")

    print("-" * 60)
    if ling:
        ling, cvv = np.array(ling), np.array(cvv)
        print(f"lingbot FOV_h : mean={ling.mean():.2f}  range=[{ling.min():.2f},{ling.max():.2f}]")
        print(f"opencv  FOV_h : mean={cvv.mean():.2f}  range=[{cvv.min():.2f},{cvv.max():.2f}]")
        print(f"mean |diff|   : {np.abs(cvv-ling).mean():.2f} deg   median |diff|: {np.median(np.abs(cvv-ling)):.2f} deg")
        print("\nOpenCV focal self-calibration from 2 views is noisy; use as a coarse cross-check.")
    else:
        print("No usable pairs for OpenCV self-calibration.")


if __name__ == "__main__":
    main()
