"""Open3D offline renderer with octree LOD selection."""

from __future__ import annotations

import contextlib
import gc
import os
from typing import List, Optional

import cv2
import numpy as np

from .camera import Camera, lookat
from .config import RenderConfig
from .scene import Scene
from .overlay import Overlay, parse_color


# ---------------------------------------------------------------------------
# Open3D utilities
# ---------------------------------------------------------------------------

_OPEN3D = None


def _import_open3d():
    global _OPEN3D
    if _OPEN3D is None:
        with _suppress_c_output():
            _OPEN3D = __import__('open3d')
    return _OPEN3D


@contextlib.contextmanager
def _suppress_c_output():
    old1, old2 = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2); os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old1, 1); os.close(old1)
        os.dup2(old2, 2); os.close(old2)


def silence_process_stdio():
    """Redirect stdout/stderr to /dev/null for the process lifetime."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2); os.close(devnull)


# ---------------------------------------------------------------------------
# Open3D Renderer
# ---------------------------------------------------------------------------

def _edl_shade(color: np.ndarray, depth: np.ndarray,
               strength: float = 0.5, radius: int = 2) -> np.ndarray:
    """Apply Eye-Dome Lighting post-processing.

    For each foreground pixel, compares log2(depth) with 8 neighbours at
    the given *radius*.  Large depth jumps produce a darkening factor,
    giving the point cloud depth cues without requiring normals.

    Args:
        color:    (H, W, 3) uint8 RGB.
        depth:    (H, W) float32 view-space depth.  Background = 0 or inf.
        strength: Darkening intensity (0.1–1.0).
        radius:   Neighbour distance in pixels (1–4).

    Returns:
        (H, W, 3) uint8 EDL-shaded image.
    """
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


class Open3DRenderer:
    """Offline Open3D renderer with octree LOD selection."""

    def __init__(self, config: RenderConfig):
        o3d = _import_open3d()
        self.config = config
        self.width = config.width
        self.height = config.height

        with _suppress_c_output():
            self._renderer = o3d.visualization.rendering.OffscreenRenderer(
                config.width, config.height)

        cg = o3d.visualization.rendering.ColorGrading(
            o3d.visualization.rendering.ColorGrading.Quality.ULTRA,
            o3d.visualization.rendering.ColorGrading.ToneMapping.LINEAR)
        self._renderer.scene.view.set_color_grading(cg)

        rgb = parse_color(config.background) or [0.0, 0.0, 0.0]
        self._renderer.scene.set_background(np.array([rgb[0], rgb[1], rgb[2], 1.0]))

        self._mat = o3d.visualization.rendering.MaterialRecord()
        self._mat.shader = 'defaultUnlit'
        self._mat.point_size = config.point_size
        self._mat.sRGB_color = True

        self._pcd_buf = o3d.geometry.PointCloud()
        self._pcd_buf.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
        self._pcd_buf.colors = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
        self._renderer.scene.add_geometry('pcd', self._pcd_buf, self._mat)

        self._extra_materials = {}
        self._active_extra_geoms: set = set()

    def render_frame(self, scene: Scene, camera: Camera, frame_idx: int,
                     overlays: Optional[List[Overlay]] = None) -> np.ndarray:
        """Render one frame: LOD select -> overlays -> render -> return (H,W,3) uint8."""
        o3d = _import_open3d()

        vis_xyz, vis_rgb = scene.octree.lod_select(
            camera.w2c(),
            *camera.fov_intrinsics(self.width, self.height),
            self.width, self.height,
            self.config.near, self.config.far,
            frame_idx, self.config.lod_target_pixels)
        vis_frames = None

        extra_geoms = []
        if overlays:
            for overlay in overlays:
                vis_xyz, vis_rgb, geoms = overlay.apply(
                    scene, camera, frame_idx, vis_xyz, vis_rgb, vis_frames)
                extra_geoms.extend(geoms)

        self._pcd_buf.points = o3d.utility.Vector3dVector(vis_xyz.astype(np.float64))
        self._pcd_buf.colors = o3d.utility.Vector3dVector(
            np.clip(vis_rgb, 0, 1).astype(np.float64))

        if hasattr(self._renderer.scene, 'update_geometry'):
            flags = (o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG |
                     o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG)
            self._renderer.scene.update_geometry('pcd', self._pcd_buf, flags)
        else:
            self._renderer.scene.remove_geometry('pcd')
            self._renderer.scene.add_geometry('pcd', self._pcd_buf, self._mat)

        self._renderer.setup_camera(
            camera.fov_deg,
            camera.center.astype(np.float64),
            camera.eye.astype(np.float64),
            camera.up.astype(np.float64))

        # EDL: clear stale overlays, grab point-cloud-only depth,
        # then add current overlays
        edl_depth = None
        if self.config.edl:
            self._update_extra_geoms([])
            edl_depth = np.asarray(
                self._renderer.render_to_depth_image(z_in_view_space=True)
            ).astype(np.float32)

        self._update_extra_geoms(extra_geoms)

        return self._render_and_postprocess(edl_depth)

    def render_frame_direct(self, vis_xyz: np.ndarray, vis_rgb: np.ndarray,
                            extra_geoms: list, camera: Camera) -> np.ndarray:
        """Render pre-culled points + extra geometry. No Scene needed.

        Used by D-lite parallel workers that receive visible points
        via SharedMemory.
        """
        o3d = _import_open3d()

        if len(vis_xyz) > 0:
            self._pcd_buf.points = o3d.utility.Vector3dVector(vis_xyz.astype(np.float64))
            self._pcd_buf.colors = o3d.utility.Vector3dVector(
                np.clip(vis_rgb, 0, 1).astype(np.float64))
        else:
            self._pcd_buf.points = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))
            self._pcd_buf.colors = o3d.utility.Vector3dVector(np.zeros((1, 3), dtype=np.float64))

        if hasattr(self._renderer.scene, 'update_geometry'):
            flags = (o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG |
                     o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG)
            self._renderer.scene.update_geometry('pcd', self._pcd_buf, flags)
        else:
            self._renderer.scene.remove_geometry('pcd')
            self._renderer.scene.add_geometry('pcd', self._pcd_buf, self._mat)

        self._renderer.setup_camera(
            camera.fov_deg,
            camera.center.astype(np.float64),
            camera.eye.astype(np.float64),
            camera.up.astype(np.float64))

        # EDL: clear stale overlays, grab point-cloud-only depth,
        # then add current overlays
        edl_depth = None
        if self.config.edl:
            self._update_extra_geoms([])
            edl_depth = np.asarray(
                self._renderer.render_to_depth_image(z_in_view_space=True)
            ).astype(np.float32)

        self._update_extra_geoms(extra_geoms)

        return self._render_and_postprocess(edl_depth)

    def _render_and_postprocess(self, edl_depth: Optional[np.ndarray] = None) -> np.ndarray:
        """Render to image, apply EDL if depth provided.

        When EDL is active, uses point-cloud-only depth (edl_depth) for
        shading and a second depth pass (with overlays) to detect overlay
        pixels.  Overlay pixels are protected from EDL darkening.
        """
        if edl_depth is not None:
            depth_full = np.asarray(
                self._renderer.render_to_depth_image(z_in_view_space=True)
            ).astype(np.float32)

        color = np.asarray(self._renderer.render_to_image())

        if edl_depth is None:
            return color

        shaded = _edl_shade(color, edl_depth,
                            strength=self.config.edl_strength,
                            radius=self.config.edl_radius)

        # Protect overlay pixels: where depth changed after adding overlays,
        # keep original color instead of EDL-shaded color
        overlay_mask = np.abs(depth_full - edl_depth) > 1e-4
        shaded[overlay_mask] = color[overlay_mask]
        return shaded

    def _update_extra_geoms(self, geoms: List[dict]):
        o3d = _import_open3d()
        current_names = set()

        for g in geoms:
            name = g['name']
            current_names.add(name)

            if g['type'] == 'lines':
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(g['points'])
                ls.lines = o3d.utility.Vector2iVector(g['segments'])
                ls.colors = o3d.utility.Vector3dVector(g['colors'])
                mat = self._get_or_create_material(
                    name, shader='unlitLine', line_width=g.get('line_width', 2.0))
                if self._renderer.scene.has_geometry(name):
                    self._renderer.scene.remove_geometry(name)
                self._renderer.scene.add_geometry(name, ls, mat)

            elif g['type'] == 'points':
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(g['points'])
                pcd.colors = o3d.utility.Vector3dVector(g['colors'])
                mat = self._get_or_create_material(
                    name, shader='defaultUnlit', point_size=g.get('point_size', 8.0))
                if self._renderer.scene.has_geometry(name):
                    self._renderer.scene.remove_geometry(name)
                self._renderer.scene.add_geometry(name, pcd, mat)

            elif g['type'] == 'textured_quad':
                self._add_textured_quad(g)

        stale = self._active_extra_geoms - current_names
        for name in stale:
            if self._renderer.scene.has_geometry(name):
                self._renderer.scene.remove_geometry(name)
        self._active_extra_geoms = current_names

    def _add_textured_quad(self, g: dict):
        o3d = _import_open3d()
        name = g['name']
        corners = g['corners']
        image = g['image']
        alpha = g.get('alpha', 0.8)

        tex_h, tex_w = 64, 64
        img_small = cv2.resize(image, (tex_w, tex_h), interpolation=cv2.INTER_AREA)
        img_rgb = img_small.astype(np.float64) / 255.0

        tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]
        us = np.linspace(0, 1, tex_w + 1, dtype=np.float64)
        vs = np.linspace(0, 1, tex_h + 1, dtype=np.float64)
        ug, vg = np.meshgrid(us, vs)

        pts = ((1 - vg[..., None]) * ((1 - ug[..., None]) * tl + ug[..., None] * tr) +
               vg[..., None] * ((1 - ug[..., None]) * bl + ug[..., None] * br))

        n_rows, n_cols = tex_h + 1, tex_w + 1
        vertices = pts.reshape(-1, 3)

        vertex_colors = np.zeros((n_rows * n_cols, 3), dtype=np.float64)
        for r in range(n_rows):
            for c in range(n_cols):
                pr = min(r, tex_h - 1)
                pc = min(c, tex_w - 1)
                vertex_colors[r * n_cols + c] = img_rgb[pr, pc] * alpha

        triangles = []
        for r in range(tex_h):
            for c in range(tex_w):
                i00 = r * n_cols + c
                i01 = r * n_cols + c + 1
                i10 = (r + 1) * n_cols + c
                i11 = (r + 1) * n_cols + c + 1
                triangles.append([i00, i01, i11])
                triangles.append([i00, i11, i10])
        triangles = np.array(triangles, dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

        mat = self._get_or_create_material(name, shader='defaultUnlit')
        if self._renderer.scene.has_geometry(name):
            self._renderer.scene.remove_geometry(name)
        self._renderer.scene.add_geometry(name, mesh, mat)

    def _get_or_create_material(self, name, shader, **kwargs):
        if name not in self._extra_materials:
            o3d = _import_open3d()
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = shader
            for k, v in kwargs.items():
                setattr(mat, k, v)
            self._extra_materials[name] = mat
        return self._extra_materials[name]

    def destroy(self):
        with _suppress_c_output():
            if self._renderer is not None:
                scene = self._renderer.scene
                for name in list(self._active_extra_geoms) + ['pcd']:
                    if scene.has_geometry(name):
                        scene.remove_geometry(name)
            del self._pcd_buf, self._renderer
            self._pcd_buf = self._renderer = None
            gc.collect()
