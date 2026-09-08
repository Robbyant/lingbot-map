"""Split-screen: top-down world-space trajectory (left, with frame0 highlighted) next to
frame0's RGB with other camera centers reprojected onto it (right, the P0*Ci view).
Combines plot_camera_trajectory_topdown.py and plot_camera_trajectory.py into one image
so you can see where frame0 sits on the overall path alongside what (if anything)
reprojects into its own view.

Usage:
    python plot_trajectory_split_view.py <results_dir> --frame0 150 [--start 0] [--end -1] [--plane xz] [--out split.png]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lingbot_map.utils.geometry import closed_form_inverse_se3

PLANE_AXES = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}


def _w2c(c2w_3x4):
    m = np.eye(4)
    m[:3, :4] = np.array(c2w_3x4, dtype=np.float64)
    return closed_form_inverse_se3(m[None])[0][:3, :4]


def render_topdown(frames, frame0, plane, size, frustum_length=1.5):
    ax_a, ax_b = PLANE_AXES[plane]
    ids = [fr["frame"] for fr in frames]
    centers = np.array([fr["extrinsic"] for fr in frames], dtype=np.float64)[:, :, 3]
    pts = centers[:, [ax_a, ax_b]]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.1
    scale = (1 - 2 * margin) * size / span.max()
    center_data = (lo + hi) / 2
    center_canvas = np.array([size / 2, size / 2])

    def to_canvas(p):
        c = (p - center_data) * scale
        return (int(round(center_canvas[0] + c[0])), int(round(center_canvas[1] - c[1])))

    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    canvas_pts = [to_canvas(p) for p in pts]

    for a, b in zip(canvas_pts[:-1], canvas_pts[1:]):
        cv2.line(canvas, a, b, (140, 140, 140), 1, cv2.LINE_AA)
    n = max(len(canvas_pts) - 1, 1)
    for j, p in enumerate(canvas_pts):
        col = tuple(int(c) for c in cmap[int(j / n * 255)])
        cv2.circle(canvas, p, 3, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"f{ids[j]}", (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    if frame0 in ids:
        idx0 = ids.index(frame0)
        p0 = canvas_pts[idx0]
        cv2.drawMarker(canvas, p0, (255, 255, 255), cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame0 (f{frame0})", (p0[0] + 14, p0[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        f0_entry = frames[idx0]
        R_c2w = np.array(f0_entry["extrinsic"], dtype=np.float64)[:, :3]
        forward_world = R_c2w @ np.array([0.0, 0.0, 1.0])
        fwd2d = np.array([forward_world[ax_a], forward_world[ax_b]])
        fwd_norm = np.linalg.norm(fwd2d)
        if fwd_norm > 1e-9:
            fwd2d /= fwd_norm
            half_fov = np.radians(f0_entry.get("fov_h_deg", 60.0)) / 2
            ang0 = np.arctan2(fwd2d[1], fwd2d[0])
            length = span.max() * frustum_length
            far_pts = []
            for sign in (-1, 1):
                a = ang0 + sign * half_fov
                p_far = pts[idx0] + np.array([np.cos(a), np.sin(a)]) * length
                far_pts.append(to_canvas(p_far))
                cv2.line(canvas, p0, far_pts[-1], (120, 200, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, far_pts[0], far_pts[1], (120, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"top-down {plane.upper()} plane (f{ids[0]}-f{ids[-1]})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def render_sideview(frames, frame0, size, frustum_length=1.5):
    """Side (elevation) view in frame0's own camera space: X=forward depth (Xc.z),
    Y=up (-Xc.y). Complements render_topdown, which only shows the horizontal (X/Z
    world) field of view and misses vertical framing entirely."""
    by_idx = {fr["frame"]: fr for fr in frames}
    f0 = by_idx[frame0]
    K0 = np.array(f0["intrinsic"], dtype=np.float64)
    w2c0 = _w2c(f0["extrinsic"])
    R0, t0 = w2c0[:, :3], w2c0[:, 3]

    ids = sorted(by_idx)
    cam_pts = []
    for i in ids:
        Ci = np.array(by_idx[i]["extrinsic"], dtype=np.float64)[:, 3]
        Xc = R0 @ Ci + t0
        cam_pts.append((Xc[2], -Xc[1]))  # (forward, up)
    cam_pts = np.array(cam_pts)

    lo, hi = cam_pts.min(axis=0), cam_pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    margin = 0.1
    scale = (1 - 2 * margin) * size / span.max()
    center_data = (lo + hi) / 2
    center_canvas = np.array([size / 2, size / 2])

    def to_canvas(p):
        c = (p - center_data) * scale
        return (int(round(center_canvas[0] + c[0])), int(round(center_canvas[1] - c[1])))

    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    canvas_pts = [to_canvas(p) for p in cam_pts]

    for a, b in zip(canvas_pts[:-1], canvas_pts[1:]):
        cv2.line(canvas, a, b, (140, 140, 140), 1, cv2.LINE_AA)
    n = max(len(canvas_pts) - 1, 1)
    for j, p in enumerate(canvas_pts):
        col = tuple(int(c) for c in cmap[int(j / n * 255)])
        cv2.circle(canvas, p, 3, col, -1, cv2.LINE_AA)
        cv2.putText(canvas, f"f{ids[j]}", (p[0] + 5, p[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    idx0 = ids.index(frame0)
    p0 = canvas_pts[idx0]
    cv2.drawMarker(canvas, p0, (255, 255, 255), cv2.MARKER_STAR, 22, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"frame0 (f{frame0})", (p0[0] + 14, p0[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # frame0 sits exactly at cam-space origin (0,0): forward axis = +X in this plot,
    # so the frustum apex/direction don't need any world-space rotation math.
    cy, fy = K0[1, 2], K0[1, 1]
    half_fov_v = np.arctan(cy / fy)  # vertical half-FOV from intrinsics (image height = 2*cy)
    length = span.max() * frustum_length
    far_pts = []
    for sign in (-1, 1):
        a = sign * half_fov_v
        p_far = np.array([np.cos(a), np.sin(a)]) * length
        far_pts.append(to_canvas(p_far))
        cv2.line(canvas, p0, far_pts[-1], (120, 200, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, far_pts[0], far_pts[1], (120, 200, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"side view (forward/up, cam0 space, f{ids[0]}-f{ids[-1]})", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def render_reproj(rdir, frames, frame0, size):
    by_idx = {fr["frame"]: fr for fr in frames}
    f0 = by_idx[frame0]
    K0 = np.array(f0["intrinsic"], dtype=np.float64)
    w2c0 = _w2c(f0["extrinsic"])
    R0, t0 = w2c0[:, :3], w2c0[:, 3]

    img = cv2.imread(str(rdir / "rgb" / f"{frame0:06d}.png"))
    if img is None:
        raise SystemExit(f"No rgb/{frame0:06d}.png in {rdir}")
    H, W = img.shape[:2]

    ids = sorted(by_idx)
    cmap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    lo, hi = ids[0], ids[-1]
    span = max(hi - lo, 1)

    out = img.copy()
    hits, behind = [], []
    for i in ids:
        Ci = np.array(by_idx[i]["extrinsic"], dtype=np.float64)[:, 3]
        Xc = R0 @ Ci + t0
        if Xc[2] <= 1e-6:
            behind.append(i)
            continue
        uv = K0 @ Xc
        u, v = uv[0] / uv[2], uv[1] / uv[2]
        if 0 <= u < W and 0 <= v < H:
            hits.append((i, int(round(u)), int(round(v))))

    for a, b in zip(hits[:-1], hits[1:]):
        cv2.line(out, a[1:], b[1:], (200, 200, 200), 1, cv2.LINE_AA)
    for i, u, v in hits:
        col = tuple(int(c) for c in cmap[int((i - lo) / span * 255)])
        cv2.circle(out, (u, v), 4, col, -1, cv2.LINE_AA)
        cv2.putText(out, f"f{i}", (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    scale = size / max(H, W)
    out = cv2.resize(out, (int(W * scale), int(H * scale)))
    canvas = np.full((size, size, 3), 30, dtype=np.uint8)
    y0 = (size - out.shape[0]) // 2
    x0 = (size - out.shape[1]) // 2
    canvas[y0:y0 + out.shape[0], x0:x0 + out.shape[1]] = out
    cv2.putText(canvas, f"reprojected onto frame f{frame0} ({len(hits)}/{len(ids)} in view)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    if behind:
        note = f"{len(behind)} frame(s) behind cam0 (z<=0), can't project: {behind}"
        cv2.putText(canvas, note, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 220), 1, cv2.LINE_AA)
        print(f"  frame {frame0}: {len(behind)} frame(s) behind the camera (Xc.z<=0): {behind}")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=str)
    ap.add_argument("--frame0", type=int, default=0)
    ap.add_argument("--start", type=int, default=None,
                    help="First frame index to plot. Default: frame0 - range/2")
    ap.add_argument("--end", type=int, default=None,
                    help="Last frame index to plot (inclusive), -1 = last available frame. "
                         "Default: frame0 + range/2")
    ap.add_argument("--range", type=int, default=10,
                    help="Size of the default frame window around frame0 when --start/--end are omitted")
    ap.add_argument("--plane", choices=list(PLANE_AXES), default="xz")
    ap.add_argument("--size", type=int, default=700)
    ap.add_argument("--frustum_length", type=float, default=1.5,
                    help="Frustum wedge length as a multiple of the plotted path's span")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    with open(rdir / "poses.json") as f:
        all_frames = json.load(f)["frames"]

    last = all_frames[-1]["frame"]
    if args.start is None and args.end is None:
        start = max(0, args.frame0 - args.range // 2)
        end = min(last, start + args.range)
    else:
        start = args.start if args.start is not None else 0
        end = args.end if args.end is not None and args.end >= 0 else last
    frames = sorted((fr for fr in all_frames if start <= fr["frame"] <= end), key=lambda fr: fr["frame"])
    if not frames:
        raise SystemExit(f"No frames in range [{start}, {end}]")

    left = render_topdown(frames, args.frame0, args.plane, args.size, args.frustum_length)
    mid = render_sideview(frames, args.frame0, args.size, args.frustum_length)
    right = render_reproj(rdir, frames, args.frame0, args.size)
    split = np.hstack([left, mid, right])

    out = args.out or str(rdir / f"split_view_{args.frame0}.png")
    cv2.imwrite(out, split)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
