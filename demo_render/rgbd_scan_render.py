"""
RGBD Scene Scan Visualization — CLI entry point.

Usage:
    python rgbd_scan_render.py --config config/indoor.yaml --input_npz scene.npz --output_video out.mp4
    python rgbd_scan_render.py --input_npz scene.npz --output_video out.mp4 --mask_sky
    python rgbd_scan_render.py --config config/indoor.yaml --dump_config my_params.yaml
"""

import argparse
import multiprocessing
import time
import warnings
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'render_cuda_ext'))

warnings.filterwarnings('ignore')

from tqdm import tqdm

from rgbd_render.camera import build_camera_path
from rgbd_render.config import load_config
from rgbd_render.overlay import build_overlays
from rgbd_render.pipeline.builder import SceneBuilder
from rgbd_render.pipeline.offline import OfflinePipeline


def _make_log():
    def _log(msg: str):
        tqdm.write(f"[{time.strftime('%H:%M:%S')}] {msg}")
    return _log


def parse_args():
    p = argparse.ArgumentParser(description='RGBD scan -> rendered video')
    p.add_argument('--config', type=str, default='', help='YAML config file')
    p.add_argument('--dump_config', type=str, default='', help='Save config and exit')
    p.add_argument('--input_npz', required=True)
    p.add_argument('--output_video', default='./scan_render.mp4')
    # Defaults are None so CLI args only override YAML when explicitly provided.
    # PipelineConfig dataclasses hold the real defaults.
    p.add_argument('--fps', type=int, default=None)
    p.add_argument('--render_width', type=int, default=None)
    p.add_argument('--render_height', type=int, default=None)
    p.add_argument('--point_size', type=float, default=None)
    p.add_argument('--near', type=float, default=None)
    p.add_argument('--far', type=float, default=None)
    p.add_argument('--background', type=str, default=None)
    p.add_argument('--voxel_size', type=float, default=None)
    p.add_argument('--octree_level', type=int, default=None)
    p.add_argument('--max_depth', type=float, default=None)
    p.add_argument('--downsample', type=int, default=None)
    p.add_argument('--no_jitter', action='store_true', default=None)
    p.add_argument('--color_update', choices=['first', 'latest'], default=None)
    p.add_argument('--keyframes_only_points', action='store_true', default=None,
                   help='Only unproject keyframe depth into the point cloud. '
                        'Non-keyframes still appear in the camera trajectory/overlay. '
                        'Requires is_keyframe / frame_type in the NPZ meta.')
    p.add_argument('--smooth_window', type=int, default=None)
    p.add_argument('--fov', type=float, default=None)
    p.add_argument('--back_offset', type=float, default=None)
    p.add_argument('--up_offset', type=float, default=None)
    p.add_argument('--look_offset', type=float, default=None)
    p.add_argument('--follow_scale_frames', type=int, default=None)
    p.add_argument('--mask_sky', action='store_true', default=None)
    p.add_argument('--sky_model', default=None)
    p.add_argument('--sky_batch_size', type=int, default=None)
    p.add_argument('--sky_mask_dir', type=str, default=None,
                   help='Directory for cached sky masks')
    p.add_argument('--sky_mask_visualization_dir', type=str, default=None,
                   help='Directory for sky mask visualization images')
    p.add_argument('--camera_vis', type=str, default=None)
    p.add_argument('--trail_color_ramp', type=str, default=None)
    p.add_argument('--trail_line_width', type=float, default=None)
    p.add_argument('--trail_tail_len', type=int, default=None)
    p.add_argument('--head_num_frames', type=int, default=None)
    p.add_argument('--head_point_size', type=float, default=None)
    p.add_argument('--head_frustum_scale', type=float, default=None)
    p.add_argument('--head_frustum_line_width', type=float, default=None)
    p.add_argument('--head_frustum_color', type=str, default=None)
    p.add_argument('--head_texture_alpha', type=float, default=None)
    p.add_argument('--frame_tag', action='store_true', default=None)
    p.add_argument('--frame_tag_position', type=str, default=None,
                   choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'])
    p.add_argument('--edl', action='store_true', default=None)
    p.add_argument('--edl_strength', type=float, default=None)
    p.add_argument('--edl_radius', type=int, default=None)
    p.add_argument('--lod_target_pixels', type=float, default=None)
    p.add_argument('--depth_video', action='store_true', default=None)
    p.add_argument('--no_combined_video', action='store_true', default=None,
                   help='Disable side-by-side render+RGB combined video')
    p.add_argument('--depth_colormap', type=str, default=None)
    p.add_argument('--depth_percentile_lo', type=float, default=None)
    p.add_argument('--depth_percentile_hi', type=float, default=None)
    p.add_argument('--birdeye_start', type=str, default=None)
    p.add_argument('--birdeye_duration', type=str, default=None)
    p.add_argument('--birdeye_transition', type=int, default=None)
    p.add_argument('--reveal_height_mult', type=float, default=None)
    p.add_argument('--num_workers', type=int, default=None)
    p.add_argument('--conf_threshold', type=float, default=None)
    p.add_argument('--vis_threshold', type=float, default=None)
    p.add_argument('--debug_frames', type=int, default=None, help='(legacy) alias for fast_review')
    p.add_argument('--skip_first', type=int, default=None,
                   help='Drop first K frames before any other filtering')
    p.add_argument('--fast_review', type=int, default=0)
    p.add_argument('--frame_stride', type=int, default=None,
                   help='Load every N-th frame from NPZ (1=all, 2=half, etc.)')
    p.add_argument('--hd_image_folder', type=str, default=None,
                   help='Folder with original high-res frames for HD RGB video output')
    return p.parse_args()


def main():
    args = parse_args()
    log = _make_log()

    if not args.fast_review and args.debug_frames:
        args.fast_review = args.debug_frames

    cfg = load_config(args)
    if args.input_npz:
        cfg.input = args.input_npz
    if args.output_video:
        cfg.output = args.output_video
    if args.fast_review and args.fast_review > 0:
        cfg.fast_review = args.fast_review

    if args.dump_config:
        cfg.to_yaml(args.dump_config)
        log(f"[config] Saved to {args.dump_config}")
        return

    scene = (SceneBuilder(cfg, log=log).load().preprocess().voxelize().build())
    camera_path = build_camera_path(cfg.camera, scene)
    overlays, overlay_specs = build_overlays(cfg, scene)

    log(f"[scene] {len(scene.sorted_xyz):,} voxels, {scene.num_frames} frames")
    log(f"[camera] {len(camera_path)} frames, {len(cfg.camera.segments)} segment(s)")

    OfflinePipeline(scene, camera_path, overlays, cfg,
                    overlay_specs=overlay_specs, log=log).run()
    scene.destroy()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
