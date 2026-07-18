"""Video encoding utilities: FFmpeg preferred, OpenCV fallback."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess

import cv2
import numpy as np


def encode_video(frame_dir: str, output_path: str, fps: int = 30):
    """Encode PNGs to MP4. Tries FFmpeg first, falls back to OpenCV."""
    pattern = os.path.join(frame_dir, 'frame_%06d.png')
    if shutil.which('ffmpeg'):
        r = subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps), '-i', pattern,
             '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
             output_path],
            capture_output=True)
        if r.returncode == 0:
            return
    # OpenCV fallback
    files = sorted(glob.glob(os.path.join(frame_dir, 'frame_*.png')))
    if not files:
        return
    first = cv2.imread(files[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))
    for p in files:
        f = cv2.imread(p)
        if f is not None:
            writer.write(f)
    writer.release()


def encode_rgb_video(images: np.ndarray, output_path: str, fps: int = 30,
                     width: int = 0, height: int = 0):
    """Encode original RGB frames (S,H,W,3 uint8) to MP4.

    If *width* or *height* is given, frames are resized before encoding.
    Providing only one dimension scales the other proportionally.
    """
    src_h, src_w = images[0].shape[:2]
    if width > 0 and height <= 0:
        height = round(src_h * width / src_w)
    elif height > 0 and width <= 0:
        width = round(src_w * height / src_h)
    elif width <= 0 and height <= 0:
        width, height = src_w, src_h
    need_resize = (width != src_w or height != src_h)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (width, height))
    for fi in range(len(images)):
        frame = cv2.cvtColor(images[fi], cv2.COLOR_RGB2BGR)
        if need_resize:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()


def encode_combined_video(render_dir: str, rgb_images: np.ndarray,
                          output_path: str, fps: int = 30):
    """Side-by-side video: rendered point cloud (left) + RGB (right).

    Both streams are resized to the same height before concatenation.
    """
    import glob as _glob
    render_files = sorted(_glob.glob(os.path.join(render_dir, 'frame_*.png')))
    if not render_files or len(rgb_images) == 0:
        return

    num_frames = min(len(render_files), len(rgb_images))
    first_render = cv2.imread(render_files[0])
    target_h = first_render.shape[0]

    # Compute RGB resize width to match render height
    rgb_h, rgb_w = rgb_images[0].shape[:2]
    rgb_new_w = round(rgb_w * target_h / rgb_h)

    out_w = first_render.shape[1] + rgb_new_w
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (out_w, target_h))
    for i in range(num_frames):
        render_frame = cv2.imread(render_files[i])
        rgb_frame = cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2BGR)
        rgb_frame = cv2.resize(rgb_frame, (rgb_new_w, target_h),
                               interpolation=cv2.INTER_AREA)
        combined = np.concatenate([render_frame, rgb_frame], axis=1)
        writer.write(combined)
    writer.release()


def colorize_depth(depth_hw: np.ndarray, depth_min: float, depth_max: float,
                   colormap: str = 'turbo') -> np.ndarray:
    """(H,W) float → (H,W,3) uint8 BGR colorized depth."""
    d = np.where(depth_hw > 0, depth_hw, depth_max)
    normed = np.clip((d - depth_min) / max(depth_max - depth_min, 1e-6), 0, 1)
    gray = (normed * 255).astype(np.uint8)
    cmap_id = getattr(cv2, f'COLORMAP_{colormap.upper()}', cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(gray, cmap_id)


def encode_depth_video(depths: np.ndarray, output_path: str, fps: int,
                       depth_range: tuple, colormap: str = 'turbo'):
    """Colorize depth frames (S,H,W) and encode to MP4."""
    depth_min, depth_max = depth_range
    h, w = depths[0].shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (w, h))
    for fi in range(len(depths)):
        frame_bgr = colorize_depth(depths[fi], depth_min, depth_max, colormap)
        writer.write(frame_bgr)
    writer.release()
