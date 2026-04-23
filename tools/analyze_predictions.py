#!/usr/bin/env python3
"""Diagnose predictions.npz from demo.py.

Usage:
    python tools/analyze_predictions.py output/predictions.npz
    python tools/analyze_predictions.py output/predictions.npz --cameras output/cameras.json
    python tools/analyze_predictions.py output/predictions.npz --save report.png
"""
import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def stat(arr, name):
    if arr is None:
        print(f"  {name}: missing")
        return
    finite = np.isfinite(arr)
    n_inf = np.sum(~finite)
    valid = arr[finite]
    print(f"  {name}: shape={arr.shape}  dtype={arr.dtype}")
    if valid.size:
        print(f"    range=[{valid.min():.4g}, {valid.max():.4g}]  mean={valid.mean():.4g}  "
              f"nan/inf={n_inf} ({100*n_inf/arr.size:.1f}%)")
    else:
        print(f"    ALL VALUES INVALID (nan/inf)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to predictions.npz")
    parser.add_argument("--cameras", default=None, help="Path to cameras.json (optional)")
    parser.add_argument("--save", default=None, help="Save figure to file")
    parser.add_argument("--conf_threshold", type=float, default=None,
                        help="Confidence threshold to simulate (default: show distribution)")
    args = parser.parse_args()

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz, allow_pickle=False)
    keys = list(d.keys())
    print(f"Keys: {keys}\n")

    world_points = d.get("world_points")   # (S, H, W, 3)
    depth_conf   = d.get("depth_conf") if "depth_conf" in keys else (
                   d.get("world_points_conf") if "world_points_conf" in keys else None)
    images       = d.get("images")         # (S, 3, H, W) or (S, H, W, 3)
    extrinsic    = d.get("extrinsic")      # (S, 3, 4)

    print("=== Array stats ===")
    stat(world_points, "world_points")
    stat(depth_conf,   "depth_conf / world_points_conf")
    stat(images,       "images")
    stat(extrinsic,    "extrinsic")
    print()

    if world_points is None:
        print("No world_points found — cannot analyse point cloud.")
        sys.exit(1)

    S, H, W = world_points.shape[:3]
    pts_flat = world_points.reshape(-1, 3)
    finite_mask = np.isfinite(pts_flat).all(axis=1)

    print("=== Point cloud sanity ===")
    print(f"  Total pixels : {len(pts_flat):,}")
    print(f"  Finite points: {finite_mask.sum():,}  ({100*finite_mask.mean():.1f}%)")

    if depth_conf is not None:
        conf_flat = depth_conf.reshape(-1)
        print(f"\n  Confidence stats (all):")
        pcts = [0, 1, 5, 25, 50, 75, 90, 95, 99, 100]
        vals = np.nanpercentile(conf_flat, pcts)
        for p, v in zip(pcts, vals):
            print(f"    p{p:3d}: {v:.4f}")

        for thr in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            n_pass = int((conf_flat[finite_mask] > thr).sum())
            print(f"  conf > {thr:.1f}: {n_pass:>8,} pts  ({100*n_pass/max(finite_mask.sum(),1):.1f}% of finite)")

    pts_valid = pts_flat[finite_mask]
    if len(pts_valid):
        print(f"\n  X range: [{pts_valid[:,0].min():.4f}, {pts_valid[:,0].max():.4f}]")
        print(f"  Y range: [{pts_valid[:,1].min():.4f}, {pts_valid[:,1].max():.4f}]")
        print(f"  Z range: [{pts_valid[:,2].min():.4f}, {pts_valid[:,2].max():.4f}]")
        dist = np.linalg.norm(pts_valid, axis=1)
        print(f"  Distance from origin: min={dist.min():.4f}  median={np.median(dist):.4f}  "
              f"p99={np.percentile(dist,99):.4f}  max={dist.max():.4f}")

    if images is not None:
        print(f"\n  Image pixel range: [{images.min():.4f}, {images.max():.4f}]")
        if images.max() <= 1.01:
            print("  → pixels appear to be in [0,1] — color mapping should be fine")
        elif images.max() <= 255.5:
            print("  → pixels appear to be in [0,255] — need /255 before color mapping (possible color bug!)")
        else:
            print("  WARNING: pixel values outside expected range — color mapping will be wrong")

    # ── Figures ──────────────────────────────────────────────────────────────
    cam_positions = None
    if args.cameras:
        with open(args.cameras) as f:
            cams = json.load(f)
        cam_positions = np.array([[c["c2w"][0][3], c["c2w"][1][3], c["c2w"][2][3]] for c in cams])

    if extrinsic is not None and cam_positions is None:
        cam_positions = extrinsic[:, :3, 3]  # translation from c2w

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # Sample points for plotting (avoid OOM on huge arrays)
    MAX_PLOT = 50_000
    if depth_conf is not None:
        conf_thr = args.conf_threshold if args.conf_threshold is not None else 2.0
        sel = finite_mask & (depth_conf.reshape(-1) > conf_thr)
    else:
        sel = finite_mask
    pts_sel = pts_flat[sel]
    if len(pts_sel) > MAX_PLOT:
        idx = np.random.choice(len(pts_sel), MAX_PLOT, replace=False)
        pts_sel = pts_sel[idx]
    print(f"\n  Plotting {len(pts_sel):,} points (conf>{args.conf_threshold if args.conf_threshold else 2.0:.1f})")

    # Top-down view (X-Z)
    ax1 = fig.add_subplot(gs[0, 0])
    if len(pts_sel):
        ax1.scatter(pts_sel[:, 0], pts_sel[:, 2], s=0.5, alpha=0.3, c="steelblue")
    if cam_positions is not None:
        ax1.plot(cam_positions[:, 0], cam_positions[:, 2], "r-", lw=1, label="cameras")
        ax1.scatter(cam_positions[0, 0], cam_positions[0, 2], c="lime", s=60, zorder=5)
        ax1.scatter(cam_positions[-1, 0], cam_positions[-1, 2], c="red", s=60, zorder=5)
        ax1.legend(fontsize=7)
    ax1.set_xlabel("X"); ax1.set_ylabel("Z")
    ax1.set_title("Top-down (X-Z)")
    ax1.set_aspect("equal")

    # Side view (X-Y)
    ax2 = fig.add_subplot(gs[0, 1])
    if len(pts_sel):
        ax2.scatter(pts_sel[:, 0], pts_sel[:, 1], s=0.5, alpha=0.3, c="steelblue")
    if cam_positions is not None:
        ax2.plot(cam_positions[:, 0], cam_positions[:, 1], "r-", lw=1)
        ax2.scatter(cam_positions[0, 0], cam_positions[0, 1], c="lime", s=60, zorder=5)
        ax2.scatter(cam_positions[-1, 0], cam_positions[-1, 1], c="red", s=60, zorder=5)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y")
    ax2.set_title("Side (X-Y)")
    ax2.set_aspect("equal")

    # Front view (Y-Z)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(pts_sel):
        ax3.scatter(pts_sel[:, 2], pts_sel[:, 1], s=0.5, alpha=0.3, c="steelblue")
    if cam_positions is not None:
        ax3.plot(cam_positions[:, 2], cam_positions[:, 1], "r-", lw=1)
    ax3.set_xlabel("Z"); ax3.set_ylabel("Y")
    ax3.set_title("Front (Z-Y)")
    ax3.set_aspect("equal")

    # Confidence histogram
    ax4 = fig.add_subplot(gs[1, 0])
    if depth_conf is not None:
        cf = depth_conf.reshape(-1)
        cf_finite = cf[np.isfinite(cf)]
        ax4.hist(cf_finite, bins=100, color="steelblue", alpha=0.7)
        for thr in [1.0, 2.0, 3.0]:
            ax4.axvline(thr, color="red", lw=1, linestyle="--", label=f"thr={thr}")
        ax4.set_xlabel("confidence"); ax4.set_ylabel("count")
        ax4.set_title("Confidence distribution")
        ax4.legend(fontsize=7)
    else:
        ax4.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax4.transAxes)

    # Distance histogram
    ax5 = fig.add_subplot(gs[1, 1])
    if len(pts_valid):
        dist = np.linalg.norm(pts_valid, axis=1)
        p99 = np.percentile(dist, 99)
        ax5.hist(dist[dist < p99 * 2], bins=100, color="darkorange", alpha=0.7)
        ax5.axvline(p99, color="red", lw=1, linestyle="--", label=f"p99={p99:.2f}")
        ax5.set_xlabel("distance from origin"); ax5.set_ylabel("count")
        ax5.set_title("Point distance distribution")
        ax5.legend(fontsize=7)

    # Point count per frame
    ax6 = fig.add_subplot(gs[1, 2])
    if depth_conf is not None:
        conf_thr = args.conf_threshold if args.conf_threshold is not None else 2.0
        counts = []
        for i in range(S):
            pts_i = world_points[i].reshape(-1, 3)
            fin = np.isfinite(pts_i).all(axis=1)
            cnf = depth_conf[i].reshape(-1) > conf_thr
            counts.append(int((fin & cnf).sum()))
        ax6.bar(range(S), counts, color="mediumseagreen", alpha=0.8)
        ax6.set_xlabel("frame"); ax6.set_ylabel("valid points")
        ax6.set_title(f"Valid points per frame (conf>{conf_thr:.1f})")
    else:
        ax6.text(0.5, 0.5, "No confidence data", ha="center", va="center", transform=ax6.transAxes)

    plt.suptitle(f"LingBot-Map predictions analysis  (S={S}, H={H}, W={W})", fontsize=12)
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150)
        print(f"\nSaved → {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
