"""Convert NPZ (depth + camera) to GLB point cloud file.

Usage:
    python -m interactive_viewer.npz_to_glb --input_dir scene_dir/ --output scene.glb
    python -m interactive_viewer.npz_to_glb --input_npz scene.npz --output scene.glb --max_points 2000000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

# Direct imports to avoid CUDA extension dependencies
import importlib.util as _ilu


def _import_from_file(module_name: str, file_path: str):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_DEMO_RENDER_DIR = Path(__file__).resolve().parents[1]
_loader_mod = _import_from_file(
    "rgbd_render.data.loader",
    str(_DEMO_RENDER_DIR / "rgbd_render" / "data" / "loader.py"),
)
load_npz_data = _loader_mod.load_npz_data

_unproject_mod = _import_from_file(
    "rgbd_render.geometry.unproject",
    str(_DEMO_RENDER_DIR / "rgbd_render" / "geometry" / "unproject.py"),
)
unproject_depth_batch_gpu = _unproject_mod.unproject_depth_batch_gpu


def _make_gradient_trail(positions, radius=0.005, colormap="coolwarm"):
    """Create camera trajectory as gradient-colored cylinders.

    Args:
        positions: (N, 3) camera positions in world space
        radius: cylinder radius
        colormap: matplotlib colormap name

    Returns:
        trimesh.Scene with colored cylinder segments
    """
    import matplotlib.cm as cm

    N = len(positions)
    if N < 2:
        return trimesh.Scene()

    cmap = cm.get_cmap(colormap)
    trail = trimesh.Scene()

    for i in range(N - 1):
        t = i / max(N - 2, 1)
        rgba = cmap(t)
        color = (np.array(rgba[:3]) * 255).astype(np.uint8)

        p0 = positions[i]
        p1 = positions[i + 1]
        seg_len = np.linalg.norm(p1 - p0)
        if seg_len < 1e-8:
            continue

        cyl = trimesh.creation.cylinder(
            radius=radius, height=seg_len, sections=6,
        )
        # Align cylinder from p0 to p1
        direction = (p1 - p0) / seg_len
        midpoint = (p0 + p1) / 2.0

        # Build rotation: cylinder default is along Z
        z_axis = np.array([0, 0, 1], dtype=np.float64)
        v = np.cross(z_axis, direction)
        s = np.linalg.norm(v)
        c = np.dot(z_axis, direction)

        if s < 1e-8:
            R = np.eye(3) if c > 0 else np.diag([1, -1, -1])
        else:
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = midpoint
        cyl.apply_transform(T)

        cyl.visual.face_colors = np.tile(
            np.append(color, 255), (len(cyl.faces), 1)
        )
        trail.add_geometry(cyl, geom_name=f"trail_{i}")

    return trail


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ to GLB point cloud")
    parser.add_argument("--input_npz", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="scene.glb")
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--max_points", type=int, default=5_000_000,
                        help="Max points in output (random downsample if exceeded)")
    parser.add_argument("--conf_threshold", type=float, default=0.0,
                        help="Filter points with confidence < threshold")
    parser.add_argument("--show_trail", action="store_true", default=True,
                        help="Show camera trajectory as gradient line")
    parser.add_argument("--no_trail", action="store_true")
    parser.add_argument("--trail_radius", type=float, default=0.01,
                        help="Trail cylinder radius")
    parser.add_argument("--trail_colormap", type=str, default="coolwarm",
                        help="Matplotlib colormap for trail gradient")
    parser.add_argument("--trail_step", type=int, default=1,
                        help="Use every Nth camera position for trail (1=all)")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Color saturation multiplier (0.0=grayscale, 1.0=original, <1=softer)")
    parser.add_argument("--brightness", type=float, default=1.0,
                        help="Brightness multiplier (>1=brighter, <1=darker)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (>1=darker midtones, <1=brighter midtones)")
    args = parser.parse_args()

    if args.input_npz is None and args.input_dir is None:
        parser.error("Must specify --input_npz or --input_dir")

    input_path = args.input_dir or args.input_npz
    print(f"Loading {input_path}...")
    data = load_npz_data(input_path)

    images = data["images"]
    depths = data["depth"]
    c2w = data["c2w"]
    Ks = data["K"]
    confs = data["confidence"]
    S = len(images)
    H, W = depths[0].shape

    if confs is not None and args.conf_threshold > 0:
        print(f"Confidence filtering enabled (threshold={args.conf_threshold})")
    elif confs is None and args.conf_threshold > 0:
        print("Warning: --conf_threshold set but no confidence data in NPZ, ignoring")

    # Unproject all frames
    print(f"Unprojecting {S} frames...")
    all_xyz, all_rgb = [], []
    batch_size = min(8, S)
    ds = args.downsample

    for bs in range(0, S, batch_size):
        be = min(bs + batch_size, S)
        d_batch = torch.from_numpy(depths[bs:be].astype(np.float32)).cuda()
        img_batch = torch.from_numpy(images[bs:be].copy()).cuda()
        Ks_batch = torch.from_numpy(Ks[bs:be].astype(np.float32)).cuda()
        c2w_batch = torch.from_numpy(c2w[bs:be].astype(np.float32)).cuda()

        pts_xyz, pts_rgb, vcounts = unproject_depth_batch_gpu(
            d_batch, img_batch, Ks_batch, c2w_batch,
            args.max_depth, ds, False,
        )
        del d_batch, img_batch, Ks_batch, c2w_batch

        offset = 0
        for j in range(be - bs):
            cnt = vcounts[j].item()
            if cnt > 0:
                frame_xyz = pts_xyz[offset:offset + cnt].cpu().numpy()
                frame_rgb = pts_rgb[offset:offset + cnt].cpu().numpy()

                # Apply confidence filtering
                if confs is not None and args.conf_threshold > 0:
                    frame_abs = bs + j
                    c_ds = confs[frame_abs, ::ds, ::ds]
                    d_ds = depths[frame_abs, ::ds, ::ds]
                    valid = (d_ds > 0) & (d_ds < args.max_depth)
                    frame_conf = c_ds[valid].astype(np.float32)
                    conf_mask = frame_conf >= args.conf_threshold
                    frame_xyz = frame_xyz[conf_mask]
                    frame_rgb = frame_rgb[conf_mask]

                if len(frame_xyz) > 0:
                    all_xyz.append(frame_xyz)
                    all_rgb.append(frame_rgb)
            offset += cnt
        del pts_xyz, pts_rgb

    torch.cuda.empty_cache()

    xyz = np.concatenate(all_xyz)
    rgb = np.concatenate(all_rgb)
    print(f"Total: {len(xyz):,} points")

    # Downsample if needed
    if len(xyz) > args.max_points:
        idx = np.random.choice(len(xyz), args.max_points, replace=False)
        idx.sort()
        xyz, rgb = xyz[idx], rgb[idx]
        print(f"Downsampled to {len(xyz):,} points")

    # Color adjustments
    rgb = np.clip(rgb, 0, 1).astype(np.float32)

    if args.saturation != 1.0:
        # Desaturate: blend toward grayscale
        gray = 0.2989 * rgb[:, 0] + 0.5870 * rgb[:, 1] + 0.1140 * rgb[:, 2]
        rgb = gray[:, None] + args.saturation * (rgb - gray[:, None])
        rgb = np.clip(rgb, 0, 1)

    if args.brightness != 1.0:
        rgb = np.clip(rgb * args.brightness, 0, 1)

    if args.gamma != 1.0:
        rgb = np.power(rgb, 1.0 / args.gamma)

    colors_uint8 = (rgb * 255).astype(np.uint8)
    alpha = np.full((len(colors_uint8), 1), 255, dtype=np.uint8)
    colors_rgba = np.hstack([colors_uint8, alpha])

    # Build trimesh scene
    scene = trimesh.Scene()

    # Add point cloud
    pc = trimesh.PointCloud(vertices=xyz, colors=colors_rgba)
    scene.add_geometry(pc, geom_name="point_cloud")

    # Add camera trajectory as gradient-colored line
    if args.show_trail and not args.no_trail:
        print("Adding camera trail...")
        cam_positions = c2w[:, :3, 3]
        step = max(1, args.trail_step)
        positions = cam_positions[::step]
        trail_scene = _make_gradient_trail(
            positions, radius=args.trail_radius, colormap=args.trail_colormap,
        )
        for name, geom in trail_scene.geometry.items():
            scene.add_geometry(geom, geom_name=name)

    # Export
    scene.export(args.output)
    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Saved {args.output} ({file_size:.1f} MB)")


if __name__ == "__main__":
    main()
