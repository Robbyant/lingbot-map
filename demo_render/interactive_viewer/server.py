"""Interactive point cloud viewer: server-side rendering + WebSocket browser control.

Usage:
    python -m demo_render.interactive_viewer.server --input_npz scene.npz --port 8890

Then open http://localhost:8890 in a browser (use SSH port forwarding on clusters).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from demo_render.interactive_viewer.camera import KeyframeAnimator, OrbitCamera  # noqa: E402

# Import loader and unproject directly to avoid rgbd_render/__init__.py
# which pulls in CUDA extensions that may not be built.
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

_octree_mod = _import_from_file(
    "rgbd_render.geometry.octree",
    str(_DEMO_RENDER_DIR / "rgbd_render" / "geometry" / "octree.py"),
)
OctreeSPC = _octree_mod.OctreeSPC


# ---------------------------------------------------------------------------
# EDL post-processing
# ---------------------------------------------------------------------------


def _edl_shade(color: np.ndarray, depth: np.ndarray,
               strength: float = 0.5, radius: int = 2) -> np.ndarray:
    """Apply Eye-Dome Lighting post-processing for depth cues."""
    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0)
    log_d = np.where(valid, np.log2(np.where(valid, depth, 1.0)), 0.0)

    offsets = [(-radius, 0), (radius, 0), (0, -radius), (0, radius),
               (-radius, -radius), (-radius, radius),
               (radius, -radius), (radius, radius)]

    response = np.zeros_like(log_d)
    for dy, dx in offsets:
        sy = slice(max(0, -dy), H - max(0, dy))
        sx = slice(max(0, -dx), W - max(0, dx))
        ty = slice(max(0, dy), H - max(0, -dy))
        tx = slice(max(0, dx), W - max(0, -dx))

        shifted = np.zeros_like(log_d)
        shifted_v = np.zeros_like(valid)
        shifted[ty, tx] = log_d[sy, sx]
        shifted_v[ty, tx] = valid[sy, sx]

        both = valid & shifted_v
        response += np.where(both, np.maximum(0.0, log_d - shifted), 0.0)

    response /= len(offsets)
    shade = np.exp(-response * strength * 300.0)
    np.clip(shade, 0.0, 1.0, out=shade)
    shade[~valid] = 1.0

    result = color.astype(np.float32) * shade[..., None]
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Open3D import (suppress noisy output)
# ---------------------------------------------------------------------------


def _import_open3d():
    import os as _os

    old1, old2 = _os.dup(1), _os.dup(2)
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    _os.dup2(devnull, 1)
    _os.dup2(devnull, 2)
    _os.close(devnull)
    try:
        o3d = __import__("open3d")
    finally:
        _os.dup2(old1, 1)
        _os.close(old1)
        _os.dup2(old2, 2)
        _os.close(old2)
    return o3d


# ---------------------------------------------------------------------------
# Point cloud data manager
# ---------------------------------------------------------------------------


class PointCloudManager:
    """Manages per-frame point cloud data and provides filtered subsets."""

    def __init__(self):
        # Per-frame data: list of (xyz [N,3], rgb [N,3], conf [N]) numpy arrays
        self.frame_xyz: List[np.ndarray] = []
        self.frame_rgb: List[np.ndarray] = []
        self.frame_conf: List[Optional[np.ndarray]] = []
        self.num_frames = 0

    def add_frame(
        self, xyz: np.ndarray, rgb: np.ndarray, conf: Optional[np.ndarray] = None
    ):
        self.frame_xyz.append(xyz)
        self.frame_rgb.append(rgb)
        self.frame_conf.append(conf)
        self.num_frames = len(self.frame_xyz)

    def get_points(
        self,
        frame_idx: int,
        display_mode: str = "all",  # "single", "cumulative", "all"
        conf_threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (xyz, rgb) for the requested view, filtered by confidence."""
        if self.num_frames == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))

        if display_mode == "single":
            indices = [min(frame_idx, self.num_frames - 1)]
        elif display_mode == "cumulative":
            indices = list(range(min(frame_idx + 1, self.num_frames)))
        else:  # "all"
            indices = list(range(self.num_frames))

        xyz_parts, rgb_parts = [], []
        for i in indices:
            x, r, c = self.frame_xyz[i], self.frame_rgb[i], self.frame_conf[i]
            if conf_threshold > 0 and c is not None:
                mask = c >= conf_threshold
                x, r = x[mask], r[mask]
            xyz_parts.append(x)
            rgb_parts.append(r)

        if not xyz_parts:
            return np.zeros((0, 3)), np.zeros((0, 3))
        return np.concatenate(xyz_parts), np.concatenate(rgb_parts)


# ---------------------------------------------------------------------------
# Scene state
# ---------------------------------------------------------------------------


class ViewerState:
    """Holds all mutable viewer state."""

    def __init__(self):
        self.camera: Optional[OrbitCamera] = None
        self.pcm: Optional[PointCloudManager] = None
        self.animator = KeyframeAnimator()

        # Display params
        self.frame_idx = 0
        self.display_mode = "all"
        self.conf_threshold = 0.0
        self.point_size = 2.0
        self.background = [0.0, 0.0, 0.0]  # black
        self.render_width = 1280
        self.render_height = 720
        self.max_points = 5_000_000  # cap for interactive rendering

        # Octree LOD
        self.octree = None  # OctreeSPC instance
        self.lod_target_pixels = 1.5
        self.near = 0.01
        self.far = 1000.0

        # EDL post-processing
        self.edl_enabled = True
        self.edl_strength = 0.4
        self.edl_radius = 2

        # Camera trail
        self.show_trail = True
        self.cam_positions = None  # (S, 3) numpy array

        # Renderer
        self._renderer = None
        self._pcd = None
        self._mat = None
        self._dirty_geometry = True
        self._dirty_camera = True

    def init_renderer(self, width: int, height: int):
        o3d = _import_open3d()
        self.render_width = width
        self.render_height = height
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        cg = o3d.visualization.rendering.ColorGrading(
            o3d.visualization.rendering.ColorGrading.Quality.ULTRA,
            o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self._renderer.scene.view.set_color_grading(cg)

        bg = self.background
        self._renderer.scene.set_background(np.array([bg[0], bg[1], bg[2], 1.0]))

        self._mat = o3d.visualization.rendering.MaterialRecord()
        self._mat.shader = "defaultUnlit"
        self._mat.point_size = self.point_size
        self._mat.sRGB_color = True

        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(
            np.zeros((1, 3), dtype=np.float64)
        )
        self._pcd.colors = o3d.utility.Vector3dVector(
            np.zeros((1, 3), dtype=np.float64)
        )
        self._renderer.scene.add_geometry("pcd", self._pcd, self._mat)

    def update_geometry(self):
        """Rebuild point cloud geometry using octree LOD selection."""
        if self._renderer is None or self.camera is None:
            return
        o3d = _import_open3d()
        t0 = time.time()

        if self.octree is not None:
            # Use octree LOD: adaptive detail based on camera distance
            cam = self.camera
            w2c = cam.get_w2c()
            fx, fy, cx, cy = cam.fov_intrinsics(
                self.render_width, self.render_height
            )
            # frame_idx controls temporal visibility in octree
            fi = self.frame_idx if self.display_mode != "all" else (
                self.pcm.num_frames - 1 if self.pcm else 0
            )
            xyz, rgb = self.octree.lod_select(
                w2c, fx, fy, cx, cy,
                self.render_width, self.render_height,
                self.near, self.far,
                fi, self.lod_target_pixels,
            )
        else:
            # Fallback: brute force with random downsample
            xyz, rgb = self.pcm.get_points(
                self.frame_idx, self.display_mode, self.conf_threshold
            )
            if len(xyz) > self.max_points:
                idx = np.random.choice(len(xyz), self.max_points, replace=False)
                idx.sort()
                xyz, rgb = xyz[idx], rgb[idx]

        if len(xyz) == 0:
            xyz = np.zeros((1, 3), dtype=np.float64)
            rgb = np.zeros((1, 3), dtype=np.float64)

        print(f"[geometry] {len(xyz):,} points, ", end="", flush=True)

        self._pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        self._pcd.colors = o3d.utility.Vector3dVector(
            np.clip(rgb, 0, 1).astype(np.float64)
        )
        print(f"built in {time.time()-t0:.1f}s")

        if hasattr(self._renderer.scene, "update_geometry"):
            flags = (
                o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG
                | o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG
            )
            self._renderer.scene.update_geometry("pcd", self._pcd, flags)
        else:
            self._renderer.scene.remove_geometry("pcd")
            self._renderer.scene.add_geometry("pcd", self._pcd, self._mat)

        self._dirty_geometry = False

    def update_material(self):
        """Update point size and background."""
        if self._renderer is None:
            return
        self._mat.point_size = self.point_size
        # Re-add geometry with new material
        self._renderer.scene.remove_geometry("pcd")
        self._renderer.scene.add_geometry("pcd", self._pcd, self._mat)
        bg = self.background
        self._renderer.scene.set_background(np.array([bg[0], bg[1], bg[2], 1.0]))

    def _update_trail_geometry(self):
        """Add/update camera trajectory as gradient-colored line segments."""
        o3d = _import_open3d()
        # Remove old trail
        if hasattr(self, '_trail_added') and self._trail_added:
            self._renderer.scene.remove_geometry("trail")
            self._trail_added = False

        if not self.show_trail or self.cam_positions is None:
            return
        if len(self.cam_positions) < 2:
            return

        positions = self.cam_positions
        N = len(positions)

        # Build line segments with per-segment color (gradient)
        points = []
        colors = []
        indices = []
        for i in range(N - 1):
            t = i / max(N - 2, 1)
            # Cyan → Blue gradient
            r, g, b = 0.0, 1.0 - 0.5 * t, 1.0
            points.append(positions[i])
            points.append(positions[i + 1])
            colors.append([r, g, b])
            colors.append([r, g, b])
            indices.append([i * 2, i * 2 + 1])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(points, dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.array(indices, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))

        trail_mat = o3d.visualization.rendering.MaterialRecord()
        trail_mat.shader = "unlitLine"
        trail_mat.line_width = 3.0
        self._renderer.scene.add_geometry("trail", ls, trail_mat)
        self._trail_added = True

    def render(self, jpeg_quality: int = 80) -> bytes:
        """Render current view and return JPEG bytes."""
        if self._renderer is None or self.camera is None:
            return b""

        # With octree LOD, always update geometry (LOD depends on camera pos)
        if self.octree is not None or self._dirty_geometry:
            self.update_geometry()

        # Update camera trail
        self._update_trail_geometry()

        cam = self.camera
        self._renderer.setup_camera(
            cam.fov_deg,
            cam.look_at.astype(np.float64),
            cam.eye.astype(np.float64),
            cam.up.astype(np.float64),
        )

        # EDL: render depth first, then color, then shade
        if self.edl_enabled:
            depth = np.asarray(
                self._renderer.render_to_depth_image(z_in_view_space=True)
            ).astype(np.float32)
            img = np.asarray(self._renderer.render_to_image())
            img = _edl_shade(img, depth, self.edl_strength, self.edl_radius)
        else:
            img = np.asarray(self._renderer.render_to_image())

        _, buf = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        return buf.tobytes()

    def render_screenshot(self, width: int, height: int, output_path: str) -> str:
        """Render high-res screenshot and save as PNG."""
        o3d = _import_open3d()
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)

        cg = o3d.visualization.rendering.ColorGrading(
            o3d.visualization.rendering.ColorGrading.Quality.ULTRA,
            o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR,
        )
        renderer.scene.view.set_color_grading(cg)

        bg = self.background
        renderer.scene.set_background(np.array([bg[0], bg[1], bg[2], 1.0]))
        renderer.scene.add_geometry("pcd", self._pcd, self._mat)

        cam = self.camera
        renderer.setup_camera(
            cam.fov_deg,
            cam.look_at.astype(np.float64),
            cam.eye.astype(np.float64),
            cam.up.astype(np.float64),
        )

        img = np.asarray(renderer.render_to_image())
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        del renderer
        return output_path


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_scene(args) -> Tuple[ViewerState, dict]:
    """Load NPZ data and build point clouds."""
    state = ViewerState()
    state.pcm = PointCloudManager()

    # Load NPZ(s)
    # load_npz_data supports both single .npz files and per-frame directories
    # (directories with frame_*.npz files are loaded in parallel and stacked)
    input_path = args.input_dir if args.input_dir else args.input_npz
    print(f"Loading {input_path}...")
    all_data = [load_npz_data(input_path)]

    # Unproject each frame
    max_depth = args.max_depth
    downsample = args.downsample

    total_points = 0
    all_c2w = []

    for data in all_data:
        images = data["images"]
        depths = data["depth"]
        c2w = data["c2w"]
        Ks = data["K"]
        confs = data["confidence"]
        S = len(images)

        all_c2w.append(c2w)

        # Batch unproject on GPU
        batch_size = min(8, S)
        for bs in range(0, S, batch_size):
            be = min(bs + batch_size, S)

            d_batch = torch.from_numpy(depths[bs:be].astype(np.float32)).cuda()
            img_batch = torch.from_numpy(images[bs:be].copy()).cuda()
            Ks_batch = torch.from_numpy(Ks[bs:be].astype(np.float32)).cuda()
            c2w_batch = torch.from_numpy(c2w[bs:be].astype(np.float32)).cuda()



            pts_xyz, pts_rgb, vcounts = unproject_depth_batch_gpu(
                d_batch, img_batch, Ks_batch, c2w_batch, max_depth, downsample, False
            )

            del d_batch, img_batch, Ks_batch, c2w_batch

            offset = 0
            for j in range(be - bs):
                cnt = vcounts[j].item()
                frame_abs = bs + j
                if cnt > 0:
                    xyz = pts_xyz[offset : offset + cnt].cpu().numpy()
                    rgb = pts_rgb[offset : offset + cnt].cpu().numpy()

                    conf = None
                    if confs is not None:
                        # Downsample conf to match
                        c_ds = confs[frame_abs, ::downsample, ::downsample]
                        d_ds = depths[frame_abs, ::downsample, ::downsample]
                        valid = (d_ds > 0) & (d_ds < max_depth)
                        conf = c_ds[valid].astype(np.float32)

                    state.pcm.add_frame(xyz, rgb, conf)
                    total_points += cnt
                else:
                    state.pcm.add_frame(
                        np.zeros((0, 3), dtype=np.float32),
                        np.zeros((0, 3), dtype=np.float32),
                        None,
                    )
                offset += cnt

            del pts_xyz, pts_rgb

    torch.cuda.empty_cache()
    print(f"Loaded {state.pcm.num_frames} frames, {total_points:,} points total")

    # Build octree for LOD rendering
    print("Building octree for LOD...")
    all_xyz_list = [f for f in state.pcm.frame_xyz if len(f) > 0]
    all_rgb_list = [f for f in state.pcm.frame_rgb if len(f) > 0]
    all_frame_list = []
    for i, f in enumerate(state.pcm.frame_xyz):
        if len(f) > 0:
            all_frame_list.append(
                np.full(len(f), i, dtype=np.int32)
            )

    if all_xyz_list:
        cat_xyz = torch.from_numpy(np.concatenate(all_xyz_list))
        cat_rgb = torch.from_numpy(np.concatenate(all_rgb_list))
        cat_frames = torch.from_numpy(np.concatenate(all_frame_list))
        octree = OctreeSPC(max_level=10)
        octree.build(cat_xyz, cat_rgb, cat_frames, log=lambda m: print(f"  {m}"))
        state.octree = octree
        del cat_xyz, cat_rgb, cat_frames
        torch.cuda.empty_cache()
        print("Octree built.")

    # Initialize camera from scene center
    all_c2w_cat = np.concatenate(all_c2w, axis=0)
    cam_positions = all_c2w_cat[:, :3, 3]
    scene_center = cam_positions.mean(axis=0).astype(np.float32)
    scene_scale = max(
        float(np.linalg.norm(cam_positions.max(0) - cam_positions.min(0))), 1e-3
    )

    # Place camera at first pose looking at scene center
    first_pos = cam_positions[0].astype(np.float32)
    up = np.median(-all_c2w_cat[:, :3, 1], axis=0)
    up = (up / max(np.linalg.norm(up), 1e-6)).astype(np.float32)

    state.camera = OrbitCamera(
        eye=first_pos,
        look_at=scene_center,
        up=up,
        fov_deg=60.0,
    )

    # Store camera positions for trail rendering
    state.cam_positions = cam_positions.astype(np.float64)

    meta = {
        "num_frames": state.pcm.num_frames,
        "total_points": total_points,
        "scene_scale": scene_scale,
    }
    return state, meta


# ---------------------------------------------------------------------------
# aiohttp application (no WebSocket origin restrictions)
# ---------------------------------------------------------------------------


def create_app(state: ViewerState, meta: dict):
    from aiohttp import web

    static_dir = Path(__file__).parent / "static"
    move_speed_base = meta["scene_scale"] * 0.01

    async def index(request):
        return web.Response(
            text=(static_dir / "index.html").read_text(),
            content_type="text/html",
        )

    async def get_meta(request):
        return web.json_response({
            "num_frames": meta["num_frames"],
            "total_points": meta["total_points"],
            "scene_scale": meta["scene_scale"],
        })

    async def get_keyframes(request):
        return web.json_response(state.animator.to_list())

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        print("[ws] Client connected")

        # Send initial frame
        try:
            frame_data = state.render()
            if frame_data:
                await ws.send_bytes(frame_data)
                print(f"[ws] Sent initial frame ({len(frame_data)} bytes)")
        except Exception as e:
            print(f"[ws] Error sending initial frame: {e}")

        need_render = False

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                t = data.get("type")

                if t == "orbit":
                    state.camera.orbit(data["dx"], data["dy"])
                    need_render = True
                elif t == "pan":
                    state.camera.pan(data["dx"], data["dy"])
                    need_render = True
                elif t == "zoom":
                    state.camera.zoom(data["delta"])
                    need_render = True
                elif t == "move":
                    speed = move_speed_base
                    if data.get("shift"):
                        speed *= 3.0
                    d = data["dir"]
                    if d == "forward":
                        state.camera.move_forward(speed)
                    elif d == "backward":
                        state.camera.move_forward(-speed)
                    elif d == "right":
                        state.camera.move_right(speed)
                    elif d == "left":
                        state.camera.move_right(-speed)
                    elif d == "up":
                        state.camera.move_up(speed)
                    elif d == "down":
                        state.camera.move_up(-speed)
                    need_render = True
                elif t == "set_frame":
                    state.frame_idx = int(data["value"])
                    state._dirty_geometry = True
                    need_render = True
                elif t == "set_display_mode":
                    state.display_mode = data["value"]
                    state._dirty_geometry = True
                    need_render = True
                elif t == "set_conf_threshold":
                    state.conf_threshold = float(data["value"])
                    state._dirty_geometry = True
                    need_render = True
                elif t == "set_point_size":
                    state.point_size = float(data["value"])
                    state.update_material()
                    need_render = True
                elif t == "set_background":
                    hx = data["value"].lstrip("#")
                    state.background = [
                        int(hx[0:2], 16) / 255.0,
                        int(hx[2:4], 16) / 255.0,
                        int(hx[4:6], 16) / 255.0,
                    ]
                    state.update_material()
                    need_render = True
                elif t == "add_keyframe":
                    state.animator.add(state.camera, state.frame_idx)
                    await ws.send_str(json.dumps({
                        "type": "keyframes_updated",
                        "keyframes": state.animator.to_list(),
                        "count": len(state.animator.keyframes),
                    }))
                elif t == "remove_keyframe":
                    state.animator.remove_last()
                    await ws.send_str(json.dumps({
                        "type": "keyframes_updated",
                        "keyframes": state.animator.to_list(),
                        "count": len(state.animator.keyframes),
                    }))
                elif t == "clear_keyframes":
                    state.animator.clear()
                    await ws.send_str(json.dumps({
                        "type": "keyframes_updated",
                        "keyframes": state.animator.to_list(),
                        "count": 0,
                    }))
                elif t == "set_edl":
                    state.edl_enabled = bool(data["enabled"])
                    need_render = True
                elif t == "set_edl_strength":
                    state.edl_strength = float(data["value"])
                    need_render = True
                elif t == "set_trail":
                    state.show_trail = bool(data["enabled"])
                    need_render = True
                elif t == "screenshot":
                    res = data.get("resolution", "1920x1080")
                    w, h = map(int, res.split("x"))
                    out_dir = data.get("output_dir", "screenshots")
                    os.makedirs(out_dir, exist_ok=True)
                    fname = f"screenshot_{int(time.time())}.png"
                    path = state.render_screenshot(
                        w, h, os.path.join(out_dir, fname)
                    )
                    await ws.send_str(json.dumps({
                        "type": "screenshot_done", "path": path,
                    }))
                elif t == "render_video":
                    if len(state.animator.keyframes) < 2:
                        await ws.send_str(json.dumps({
                            "type": "error",
                            "message": "Need at least 2 keyframes",
                        }))
                        continue
                    fps = int(data.get("fps", 30))
                    steps = int(data.get("steps_per_segment", 60))
                    out_path = data.get("output", "camera_path.mp4")
                    await ws.send_str(json.dumps({
                        "type": "video_progress",
                        "message": "Interpolating keyframes...",
                    }))
                    interp = state.animator.interpolate(steps)
                    frames_rendered = []
                    for vi, (cam, fi) in enumerate(interp):
                        state.camera = cam
                        state.frame_idx = fi
                        state._dirty_geometry = True
                        state.update_geometry()
                        state._renderer.setup_camera(
                            cam.fov_deg,
                            cam.look_at.astype(np.float64),
                            cam.eye.astype(np.float64),
                            cam.up.astype(np.float64),
                        )
                        img = np.asarray(state._renderer.render_to_image())
                        frames_rendered.append(img)
                        if vi % 10 == 0:
                            await ws.send_str(json.dumps({
                                "type": "video_progress",
                                "message": f"Rendering {vi+1}/{len(interp)}...",
                            }))
                    frames_arr = np.stack(frames_rendered)
                    vh, vw = frames_arr[0].shape[:2]
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh)
                    )
                    for frame in frames_arr:
                        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    writer.release()
                    await ws.send_str(json.dumps({
                        "type": "video_done",
                        "path": out_path,
                        "frames": len(interp),
                    }))
                elif t == "goto_keyframe":
                    ki = int(data["index"])
                    if 0 <= ki < len(state.animator.keyframes):
                        kf = state.animator.keyframes[ki]
                        state.camera = OrbitCamera(
                            kf.eye.copy(), kf.look_at.copy(),
                            kf.up.copy(), kf.fov_deg,
                        )
                        state.frame_idx = kf.frame_idx
                        state._dirty_geometry = True
                        need_render = True

                if need_render:
                    frame_data = state.render()
                    if frame_data:
                        await ws.send_bytes(frame_data)
                    need_render = False

            elif msg.type == web.WSMsgType.ERROR:
                print(f"[ws] Error: {ws.exception()}")

        print("[ws] Client disconnected")
        return ws

    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/meta", get_meta)
    app.router.add_get("/keyframes", get_keyframes)
    app.router.add_get("/ws", ws_handler)
    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactive Point Cloud Viewer")
    parser.add_argument("--input_npz", type=str, default=None, help="Path to NPZ file")
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Directory containing per-frame NPZ files",
    )
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--max_depth", type=float, default=100.0)
    parser.add_argument("--downsample", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--max_points", type=int, default=5_000_000,
        help="Max points for interactive rendering (random downsample if exceeded)",
    )
    args = parser.parse_args()

    if args.input_npz is None and args.input_dir is None:
        parser.error("Must specify --input_npz or --input_dir")

    print("Loading scene data...")
    state, meta = load_scene(args)

    print("Initializing renderer...")
    state.init_renderer(args.width, args.height)
    state.max_points = args.max_points
    state._dirty_geometry = True

    print("Pre-rendering initial frame...")
    state.update_geometry()
    state.render()
    print("Initial frame ready.")

    print(f"Starting server at http://{args.host}:{args.port}")
    print(f"  Frames: {meta['num_frames']}, Points: {meta['total_points']:,}")

    from aiohttp import web
    app = create_app(state, meta)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
