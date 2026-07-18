"""OfflinePipeline: unified entry point for serial and parallel rendering.

Usage:
    pipeline = OfflinePipeline(scene, camera_path, overlays, config, log=log)
    pipeline.run()  # renders + encodes + cleans up
"""

from __future__ import annotations

import os
import shutil
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..camera import Camera, CameraPath
from ..config import PipelineConfig
from ..overlay import Overlay, CameraOverlay, stamp_frame_tag
from ..renderer import Open3DRenderer
from ..scene import Scene
from ..video import encode_video, encode_rgb_video, encode_depth_video, encode_combined_video


class OfflinePipeline:
    """Orchestrates rendering, video encoding, and cleanup."""

    def __init__(self, scene: Scene, camera_path: CameraPath,
                 overlays: List[Overlay], config: PipelineConfig,
                 overlay_specs: Optional[list] = None,
                 log=None):
        self.scene = scene
        self.camera_path = camera_path
        self.overlays = overlays
        self.overlay_specs = overlay_specs or []
        self.cfg = config
        self._log = log or (lambda m: None)

    def run(self):
        """Full pipeline: render → encode → cleanup."""
        output_dir = os.path.splitext(self.cfg.output)[0] + '_render_frames'
        os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()
        self._log(f"[render] Output: {output_dir}, workers: {self.cfg.num_workers}")

        if self.cfg.num_workers > 1:
            self._run_parallel(output_dir)
        else:
            self._run_serial(output_dir)

        self._log(f"[render] Done in {time.time()-t0:.1f}s")

        # Encode RGB video from original frames (or HD folder if provided)
        if self.cfg.output and self.scene.images is not None:
            base = os.path.splitext(self.cfg.output)[0]
            rgb_path = f"{base}_rgb.mp4"
            hd_folder = self.cfg.hd_image_folder
            if hd_folder and os.path.isdir(hd_folder):
                self._log(f"[rgb] Encoding HD frames from {hd_folder}")
                self._encode_hd_rgb(hd_folder, rgb_path)
            else:
                self._log(f"[rgb] Encoding original frames to {rgb_path}")
                encode_rgb_video(self.scene.images, rgb_path, self.cfg.fps)

        # Encode depth visualization video
        if (self.cfg.output and self.cfg.render.depth_video
                and self.scene.depths is not None
                and self.scene.depth_range is not None):
            base = os.path.splitext(self.cfg.output)[0]
            depth_path = f"{base}_depth.mp4"
            self._log(f"[depth] Encoding depth video to {depth_path}")
            encode_depth_video(self.scene.depths, depth_path, self.cfg.fps,
                               self.scene.depth_range,
                               self.cfg.render.depth_colormap)

        # Encode rendered video
        if self.cfg.output:
            encode_video(output_dir, self.cfg.output, self.cfg.fps)

        # Encode combined side-by-side video (render + RGB)
        if (self.cfg.output and self.scene.images is not None
                and getattr(self.cfg.render, 'combined_video', True)):
            base = os.path.splitext(self.cfg.output)[0]
            combined_path = f"{base}_combined.mp4"
            self._log(f"[combined] Encoding side-by-side video to {combined_path}")
            encode_combined_video(output_dir, self.scene.images,
                                  combined_path, self.cfg.fps)

        # Cleanup render frames
        if self.cfg.output:
            if os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
                self._log(f"[cleanup] Removed {output_dir}")

        # Save config snapshot
        config_snapshot = os.path.splitext(self.cfg.output)[0] + '_config.yaml'
        self.cfg.to_yaml(config_snapshot)
        self._log(f"[config] Snapshot saved to {config_snapshot}")

    def _encode_hd_rgb(self, hd_folder: str, output_path: str):
        """Encode HD RGB video from a folder of original-resolution frames."""
        import glob as _glob
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        hd_paths = []
        for ext in exts:
            hd_paths.extend(_glob.glob(os.path.join(hd_folder, ext)))
        hd_paths = sorted(hd_paths)

        # Apply same skip_first + fast_review + stride as NPZ loading
        skip = getattr(self.cfg, 'skip_first', 0) or 0
        if skip > 0:
            hd_paths = hd_paths[skip:]
        if self.cfg.fast_review > 0:
            hd_paths = hd_paths[:self.cfg.fast_review]
        stride = self.cfg.frame_stride or 1
        if stride > 1:
            hd_paths = hd_paths[::stride]

        if not hd_paths:
            self._log(f"[rgb] No images found in {hd_folder}, falling back to NPZ frames")
            encode_rgb_video(self.scene.images, output_path, self.cfg.fps)
            return

        first = cv2.imread(hd_paths[0])
        h, w = first.shape[:2]
        self._log(f"[rgb] HD: {w}x{h}, {len(hd_paths)} frames")
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 self.cfg.fps, (w, h))
        for p in tqdm(hd_paths, desc='Encoding HD RGB', unit='frame'):
            frame = cv2.imread(p)
            if frame is not None:
                writer.write(frame)
        writer.release()

    def _run_serial(self, output_dir: str):
        """Render all frames in the main process."""
        renderer = Open3DRenderer(self.cfg.render)
        S = len(self.camera_path)
        do_tag = self.cfg.overlay.frame_tag
        tag_pos = self.cfg.overlay.frame_tag_position

        with tqdm(total=S, unit='frame', dynamic_ncols=True,
                  desc='Rendering') as pbar:
            for fi in range(S):
                camera = self.camera_path[fi]
                img = renderer.render_frame(self.scene, camera, fi,
                                            self.overlays)
                if do_tag:
                    stamp_frame_tag(img, fi, S, tag_pos)
                cv2.imwrite(
                    os.path.join(output_dir, f'frame_{fi:06d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                pbar.update(1)
                if fi < 3 or fi % 50 == 0:
                    self._log(f"[frame {fi:4d}] eye={camera.eye.round(3)}")
                if (fi + 1) % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        renderer.destroy()

    def _run_parallel(self, output_dir: str):
        """D-lite parallel rendering via SharedMemory."""
        from .parallel import run_parallel
        run_parallel(self.scene, self.camera_path, output_dir,
                     self.cfg, self.overlays, self.overlay_specs,
                     log=self._log)
