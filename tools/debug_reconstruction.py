#!/usr/bin/env python3
"""Debug reconstruction quality from predictions.npz.

Checks:
  1. Are world_points in front of or behind each camera?
  2. Do reprojected points align with the original image?
  3. Depth map plausibility per frame.

Usage:
    python tools/debug_reconstruction.py output/predictions.npz
    python tools/debug_reconstruction.py output/predictions.npz --frames 0 10 50 100
    python tools/debug_reconstruction.py output/predictions.npz --save debug/
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def c2w_to_w2c(c2w_3x4):
    """Invert c2w (3x4) → w2c (3x4)."""
    R = c2w_3x4[:3, :3]
    t = c2w_3x4[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    w2c = np.eye(4)
    w2c[:3, :3] = R_inv
    w2c[:3, 3] = t_inv
    return w2c[:3, :]  # (3, 4)


def project_to_camera(world_pts, w2c_3x4, K_3x3):
    """world_pts (N,3) → pixel coords (N,2) and depth (N,)."""
    R, t = w2c_3x4[:3, :3], w2c_3x4[:3, 3]
    cam_pts = (R @ world_pts.T).T + t          # (N, 3) in camera space
    depth_cam = cam_pts[:, 2]
    fx, fy = K_3x3[0, 0], K_3x3[1, 1]
    cx, cy = K_3x3[0, 2], K_3x3[1, 2]
    u = fx * cam_pts[:, 0] / (cam_pts[:, 2] + 1e-8) + cx
    v = fy * cam_pts[:, 1] / (cam_pts[:, 2] + 1e-8) + cy
    return np.stack([u, v], axis=1), depth_cam


def analyze_frame(frame_idx, world_points, extrinsic, intrinsic, images,
                  depth_conf, conf_threshold=2.0):
    """Return dict of diagnostics for one frame."""
    pts = world_points[frame_idx].reshape(-1, 3)          # (H*W, 3)
    c2w = np.eye(4); c2w[:3, :] = extrinsic[frame_idx]   # (4, 4)
    w2c = c2w_to_w2c(extrinsic[frame_idx])
    K   = intrinsic[frame_idx]
    H, W = world_points.shape[1:3]

    # Camera position and forward direction
    cam_pos     = extrinsic[frame_idx][:3, 3]
    cam_forward = extrinsic[frame_idx][:3, 2]  # 3rd column of R = Z axis

    # Transform points to camera space
    _, depth_cam = project_to_camera(pts, w2c, K)

    finite = np.isfinite(pts).all(axis=1)
    if depth_conf is not None:
        conf_mask = depth_conf[frame_idx].reshape(-1) > conf_threshold
    else:
        conf_mask = np.ones(len(pts), dtype=bool)
    valid = finite & conf_mask

    n_valid      = valid.sum()
    n_front      = (depth_cam[valid] > 0).sum()
    n_behind     = (depth_cam[valid] <= 0).sum()
    pct_front    = 100 * n_front  / max(n_valid, 1)
    pct_behind   = 100 * n_behind / max(n_valid, 1)
    depth_median = float(np.median(depth_cam[valid])) if n_valid else float("nan")
    depth_p5     = float(np.percentile(depth_cam[valid], 5))  if n_valid else float("nan")
    depth_p95    = float(np.percentile(depth_cam[valid], 95)) if n_valid else float("nan")

    # Image for display (C,H,W) → (H,W,C), clip to [0,1]
    img_display = None
    if images is not None:
        img = images[frame_idx]
        if img.shape[0] == 3:          # (3,H,W) → (H,W,3)
            img = img.transpose(1, 2, 0)
        img_display = np.clip(img, 0, 1)

    # Reprojection: project valid pts back and compare to pixel grid
    uv, _ = project_to_camera(pts[valid], w2c, K)

    return dict(
        frame=frame_idx,
        cam_pos=cam_pos,
        cam_forward=cam_forward,
        n_valid=n_valid,
        pct_front=pct_front,
        pct_behind=pct_behind,
        depth_median=depth_median,
        depth_p5=depth_p5,
        depth_p95=depth_p95,
        depth_cam_valid=depth_cam[valid],
        world_pts_valid=pts[valid],
        uv_reprojected=uv,
        img_display=img_display,
        H=H, W=W,
    )


def plot_frame(ax_row, diag):
    """Fill one row of subplots for a single frame."""
    ax_img, ax_depth, ax_reproj, ax_text = ax_row
    f = diag["frame"]

    # ── image ──────────────────────────────────────────────────────
    if diag["img_display"] is not None:
        ax_img.imshow(diag["img_display"])
    ax_img.set_title(f"frame {f}: input image", fontsize=8)
    ax_img.axis("off")

    # ── depth histogram ────────────────────────────────────────────
    dc = diag["depth_cam_valid"]
    if len(dc):
        p1, p99 = np.percentile(dc, 1), np.percentile(dc, 99)
        ax_depth.hist(np.clip(dc, p1 * 1.5, p99 * 1.5), bins=80,
                      color="steelblue" if diag["pct_front"] > 90 else "tomato",
                      alpha=0.8)
        ax_depth.axvline(0, color="red", lw=1.5, label="camera plane")
        ax_depth.set_xlabel("depth in camera space", fontsize=7)
        ax_depth.set_title(
            f"front {diag['pct_front']:.0f}%  behind {diag['pct_behind']:.0f}%\n"
            f"median={diag['depth_median']:.2f}  p5={diag['depth_p5']:.2f}  p95={diag['depth_p95']:.2f}",
            fontsize=7)
        ax_depth.legend(fontsize=6)
    ax_depth.tick_params(labelsize=6)

    # ── reprojection scatter ───────────────────────────────────────
    uv = diag["uv_reprojected"]
    H, W = diag["H"], diag["W"]
    if diag["img_display"] is not None:
        ax_reproj.imshow(diag["img_display"], alpha=0.5)
    in_frame = ((uv[:, 0] >= 0) & (uv[:, 0] < W) &
                (uv[:, 1] >= 0) & (uv[:, 1] < H))
    MAX_PTS = 2000
    if in_frame.sum():
        idx = np.random.choice(in_frame.sum(),
                               min(MAX_PTS, in_frame.sum()), replace=False)
        ax_reproj.scatter(uv[in_frame][idx, 0], uv[in_frame][idx, 1],
                          s=0.3, alpha=0.4, c="lime")
    pct_in = 100 * in_frame.mean()
    ax_reproj.set_xlim(0, W); ax_reproj.set_ylim(H, 0)
    ax_reproj.set_title(f"reprojection  {pct_in:.0f}% in frame", fontsize=8)
    ax_reproj.axis("off")

    # ── text summary ───────────────────────────────────────────────
    pos = diag["cam_pos"]
    fwd = diag["cam_forward"]
    txt = (f"frame {f}\n"
           f"pos  [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]\n"
           f"fwd  [{fwd[0]:.2f}, {fwd[1]:.2f}, {fwd[2]:.2f}]\n"
           f"valid pts: {diag['n_valid']:,}\n"
           f"front: {diag['pct_front']:.1f}%\n"
           f"behind: {diag['pct_behind']:.1f}%")
    color = "limegreen" if diag["pct_front"] > 90 else \
            "orange"    if diag["pct_front"] > 50 else "red"
    ax_text.text(0.05, 0.95, txt, transform=ax_text.transAxes,
                 fontsize=8, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor=color, alpha=0.3))
    ax_text.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Path to predictions.npz")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Frame indices to inspect (default: 0, S//4, S//2, 3*S//4)")
    parser.add_argument("--conf_threshold", type=float, default=2.0)
    parser.add_argument("--save", default=None,
                        help="Directory to save per-frame debug figures")
    args = parser.parse_args()

    print(f"Loading {args.npz} ...")
    d = np.load(args.npz, allow_pickle=False)

    world_points = d["world_points"]          # (S, H, W, 3)
    extrinsic    = d["extrinsic"]             # (S, 3, 4)  c2w
    intrinsic    = d["intrinsic"]             # (S, 3, 3)
    images       = d.get("images")            # (S, 3, H, W) or None
    depth_conf   = (d.get("depth_conf") if "depth_conf" in d
                    else d.get("world_points_conf") if "world_points_conf" in d
                    else None)

    S = world_points.shape[0]
    frames = args.frames or [0, S // 4, S // 2, 3 * S // 4, S - 1]
    frames = [min(f, S - 1) for f in frames]
    print(f"Analysing frames: {frames}\n")

    # ── Per-frame text summary ─────────────────────────────────────
    print(f"{'frame':>6}  {'front%':>7}  {'behind%':>8}  {'depth_median':>12}  {'cam_pos':>30}")
    diags = []
    for fi in frames:
        diag = analyze_frame(fi, world_points, extrinsic, intrinsic,
                             images, depth_conf, args.conf_threshold)
        diags.append(diag)
        print(f"{fi:6d}  {diag['pct_front']:7.1f}  {diag['pct_behind']:8.1f}  "
              f"{diag['depth_median']:12.3f}  "
              f"[{diag['cam_pos'][0]:.2f}, {diag['cam_pos'][1]:.2f}, {diag['cam_pos'][2]:.2f}]")

    # ── Diagnosis ─────────────────────────────────────────────────
    print()
    avg_front = np.mean([d["pct_front"] for d in diags])
    if avg_front > 90:
        print("✓ Points are mostly in front of cameras — geometry looks correct.")
        print("  Blank viewer / bad PLY is likely a density/scale issue, not a logic bug.")
    elif avg_front > 50:
        print("△ Partial front/behind mix — possible coordinate convention mismatch.")
        print("  Check if extrinsic is truly c2w (camera-to-world).")
    else:
        print("✗ Most points are BEHIND cameras — likely a coordinate convention bug.")
        print("  The c2w→w2c inversion or world_point projection may be flipped.")

    # ── Figures ───────────────────────────────────────────────────
    n = len(diags)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n),
                             gridspec_kw={"width_ratios": [2, 2, 2, 1]})
    if n == 1:
        axes = [axes]
    for row, diag in zip(axes, diags):
        plot_frame(row, diag)

    plt.suptitle("Reconstruction debug: per-frame geometry check", fontsize=11)
    plt.tight_layout()

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        path = os.path.join(args.save, "debug_frames.png")
        plt.savefig(path, dpi=120)
        print(f"\nSaved → {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
