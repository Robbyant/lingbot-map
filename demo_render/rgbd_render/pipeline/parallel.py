"""D-lite parallel rendering: per-frame SharedMemory, no mmap temp files.

Architecture:
    Main process (producer):
        - Octree LOD select → visible xyz/rgb
          → create SharedMemory with exact size → write → send name via Queue

    Workers (consumers):
        - Each has an Open3DRenderer (no point cloud)
        - Read from SharedMemory → copy → close + unlink → overlay + render → write PNG
        - Increment shared counter for progress

    Progress: multiprocessing.Value shared counter (no glob polling)
    Crash: producer tracks pending shm names, atexit unlinks any leaked ones
"""

from __future__ import annotations

import atexit
import multiprocessing
import os
import warnings
from multiprocessing.shared_memory import SharedMemory
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from ..camera import CameraPath
from ..config import PipelineConfig, RenderConfig
from ..overlay import Overlay, CameraOverlay, stamp_frame_tag, TrailStyle, HeadStyle, preset_styles, parse_color
from ..renderer import Open3DRenderer, silence_process_stdio
from ..scene import Scene


# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _worker_fn(job_queue: multiprocessing.Queue,
               done_counter: multiprocessing.Value,
               init_args: dict):
    """Consumer worker: reads visible points from SharedMemory, renders."""
    import traceback
    _err_path = os.path.join(init_args['output_dir'], f'.worker_error_{os.getpid()}.txt')
    try:
        _worker_fn_inner(job_queue, done_counter, init_args)
    except Exception:
        with open(_err_path, 'w') as f:
            traceback.print_exc(file=f)


def _worker_fn_inner(job_queue, done_counter, init_args):
    silence_process_stdio()
    warnings.filterwarnings('ignore')

    config = RenderConfig.from_dict(init_args['render_config'])
    camera_path = CameraPath.load(init_args['cameras_json'])

    overlays = []
    c2w_poses = init_args['c2w_poses']
    for od in init_args.get('overlay_specs', []):
        if od['type'] == 'camera':
            oc = od.get('overlay_config', {})
            base_trail, base_head = preset_styles(od['preset'])
            base_trail.enabled = oc.get('trail_enabled', base_trail.enabled)
            base_trail.tail_len = oc.get('trail_tail_len', base_trail.tail_len)
            base_trail.line_width = oc.get('trail_line_width', base_trail.line_width)
            base_trail.color_ramp = oc.get('trail_color_ramp', base_trail.color_ramp)
            if base_head is not None:
                base_head.num_frames = oc.get('head_num_frames', base_head.num_frames)
                base_head.point_size = oc.get('head_point_size', base_head.point_size)
                base_head.frustum_scale = oc.get('head_frustum_scale', base_head.frustum_scale)
                base_head.frustum_line_width = oc.get('head_frustum_line_width', base_head.frustum_line_width)
                base_head.frustum_color = parse_color(oc.get('head_frustum_color', ''))
                base_head.texture_alpha = oc.get('head_texture_alpha', base_head.texture_alpha)
            overlays.append(CameraOverlay(
                c2w_poses, intrinsics=init_args.get('intrinsics'),
                images=init_args.get('images_thumb'),
                trail=base_trail, head=base_head))

    renderer = Open3DRenderer(config)
    output_dir = init_args['output_dir']
    do_tag = init_args.get('frame_tag', False)
    tag_pos = init_args.get('frame_tag_position', 'top_left')
    total_frames = init_args.get('total_frames', 0)

    while True:
        try:
            item = job_queue.get(timeout=300)
        except Exception:
            break
        if item is None:
            break

        frame_idx, shm_name, num_visible = item

        if num_visible == 0:
            camera = camera_path[frame_idx]
            empty = np.empty((0, 3), dtype=np.float32)
            img = _render_with_overlays(renderer, camera, frame_idx,
                                        empty, empty, overlays,
                                        c2w_poses)
        else:
            shm = SharedMemory(name=shm_name, create=False)
            xyz_bytes = num_visible * 3 * 4
            rgb_bytes = num_visible * 3 * 4

            vis_xyz = np.ndarray((num_visible, 3), dtype=np.float32,
                                 buffer=shm.buf[:xyz_bytes]).copy()
            vis_rgb = np.ndarray((num_visible, 3), dtype=np.float32,
                                 buffer=shm.buf[xyz_bytes:xyz_bytes + rgb_bytes]).copy()
            # Worker owns cleanup: close + unlink
            shm.close()
            shm.unlink()

            camera = camera_path[frame_idx]
            img = _render_with_overlays(renderer, camera, frame_idx,
                                        vis_xyz, vis_rgb,
                                        overlays, c2w_poses)

        if do_tag:
            stamp_frame_tag(img, frame_idx, total_frames, tag_pos)
        cv2.imwrite(os.path.join(output_dir, f'frame_{frame_idx:06d}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        with done_counter.get_lock():
            done_counter.value += 1

    renderer.destroy()


def _render_with_overlays(renderer, camera, frame_idx,
                          vis_xyz, vis_rgb,
                          overlays, c2w_poses):
    """Apply overlays then render."""
    extra_geoms = []
    for overlay in overlays:
        vis_xyz, vis_rgb, geoms = overlay.apply(
            None, camera, frame_idx, vis_xyz, vis_rgb, None)
        extra_geoms.extend(geoms)
    return renderer.render_frame_direct(vis_xyz, vis_rgb, extra_geoms, camera)


# ---------------------------------------------------------------------------
# Producer: GPU cull + per-frame SharedMemory
# ---------------------------------------------------------------------------

def run_parallel(scene: Scene, camera_path: CameraPath,
                 output_dir: str, config: PipelineConfig,
                 overlays: List[Overlay], overlay_specs: list,
                 log=None):
    """D-lite parallel rendering.

    Producer (main): GPU cull → gather visible points → SharedMemory (exact size)
    Consumers (workers): SharedMemory → copy → unlink → overlay + render → PNG
    """
    _log = log or (lambda m: None)
    S = len(camera_path)
    num_workers = config.num_workers
    render_cfg = config.render
    _log(f"[parallel] D-lite: {num_workers} workers, per-frame SharedMemory")

    # Save camera path for workers
    cameras_json = os.path.join(output_dir, '.cameras.json')
    camera_path.save(cameras_json)

    # Thumbnail images for textured frustum overlay
    images_thumb = None
    needs_images = any(s.get('preset') == 'textured' for s in overlay_specs
                       if s.get('type') == 'camera')
    if needs_images:
        images_thumb = scene.thumbnail_images()

    init_args = {
        'output_dir': output_dir,
        'render_config': render_cfg.to_dict(),
        'cameras_json': cameras_json,
        'c2w_poses': scene.c2w_poses,
        'intrinsics': scene.intrinsics,
        'images_thumb': images_thumb,
        'overlay_specs': overlay_specs,
        'frame_tag': config.overlay.frame_tag,
        'frame_tag_position': config.overlay.frame_tag_position,
        'total_frames': S,
    }

    # Shared progress counter
    ctx = multiprocessing.get_context('spawn')
    done_counter = ctx.Value('i', 0)
    queue_depth = min(num_workers * 2, 64)
    job_queue = ctx.Queue(maxsize=queue_depth)

    # Ensure spawned workers can find render_cuda_ext at module import time
    _ext_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', '..', 'render_cuda_ext'))
    if os.path.isdir(_ext_dir):
        pp = os.environ.get('PYTHONPATH', '')
        if _ext_dir not in pp.split(os.pathsep):
            os.environ['PYTHONPATH'] = _ext_dir + (os.pathsep + pp if pp else '')

    # Start workers
    workers = []
    for _ in range(num_workers):
        p = ctx.Process(target=_worker_fn,
                        args=(job_queue, done_counter, init_args))
        p.start()
        workers.append(p)

    # Track pending shm names for crash cleanup
    _pending_shm_names: List[str] = []

    def _cleanup_leaked_shm():
        for name in _pending_shm_names:
            try:
                shm = SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except Exception:
                pass
        _pending_shm_names.clear()

    atexit.register(_cleanup_leaked_shm)

    lod_target = render_cfg.lod_target_pixels if hasattr(render_cfg, 'lod_target_pixels') else 1.5

    # Producer loop
    with tqdm(total=S, unit='frame', dynamic_ncols=True, desc='Rendering') as pbar:
        prev_done = 0

        for fi in range(S):
            # Check worker liveness — if all dead, abort early
            alive = [w for w in workers if w.is_alive()]
            if not alive:
                import glob
                err_files = glob.glob(os.path.join(output_dir, '.worker_error_*.txt'))
                msgs = []
                for ef in err_files[:3]:
                    with open(ef) as f:
                        msgs.append(f.read())
                err_detail = '\n'.join(msgs) if msgs else '(no error logs found)'
                raise RuntimeError(
                    f"All {num_workers} render workers died.\n{err_detail}")

            camera = camera_path[fi]

            vis_xyz, vis_rgb = scene.octree.lod_select(
                camera.w2c(),
                *camera.fov_intrinsics(render_cfg.width, render_cfg.height),
                render_cfg.width, render_cfg.height,
                render_cfg.near, render_cfg.far,
                fi, lod_target)

            N = len(vis_xyz)
            if N == 0:
                job_queue.put((fi, '', 0))
            else:
                vis_xyz = np.ascontiguousarray(vis_xyz)
                vis_rgb = np.ascontiguousarray(vis_rgb)
                total_bytes = vis_xyz.nbytes + vis_rgb.nbytes

                shm = SharedMemory(create=True, size=total_bytes)
                _pending_shm_names.append(shm.name)

                offset = 0
                shm.buf[offset:offset + vis_xyz.nbytes] = vis_xyz.tobytes()
                offset += vis_xyz.nbytes
                shm.buf[offset:offset + vis_rgb.nbytes] = vis_rgb.tobytes()

                shm.close()
                job_queue.put((fi, shm.name, N))

            # Update progress from shared counter
            cur_done = done_counter.value
            if cur_done > prev_done:
                pbar.update(cur_done - prev_done)
                prev_done = cur_done

            if len(_pending_shm_names) > queue_depth * 2:
                _pending_shm_names[:] = _pending_shm_names[-queue_depth:]

            if (fi + 1) % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Send stop sentinels
        for _ in range(num_workers):
            job_queue.put(None)

        # Wait for workers
        for p in workers:
            p.join(timeout=600)

        # Final progress update
        cur_done = done_counter.value
        if cur_done > prev_done:
            pbar.update(cur_done - prev_done)

    # Cleanup
    _cleanup_leaked_shm()
    atexit.unregister(_cleanup_leaked_shm)
    if os.path.exists(cameras_json):
        os.remove(cameras_json)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log(f"[parallel] All {S} frames rendered.")
