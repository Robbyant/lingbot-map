"""Configuration system: nested dataclasses + YAML + CLI merge.

Usage:
    # From YAML
    cfg = PipelineConfig.from_yaml('config/indoor.yaml')

    # CLI override
    cfg = PipelineConfig.from_yaml('config/indoor.yaml')
    cfg = cfg.merge_cli(args)

    # Save snapshot
    cfg.to_yaml('run_config.yaml')

    # Pure CLI (backward compat)
    cfg = PipelineConfig()
    cfg = cfg.merge_cli(args)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, asdict
from typing import List, Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class SceneConfig:
    voxel_size: float = 0.001
    octree_level: int = 10
    max_depth: float = 100.0
    downsample: int = 2
    color_update: str = 'first'
    jitter: bool = True
    # Only unproject depth from keyframe frames when the NPZ carries a
    # per-frame ``is_keyframe`` / ``frame_type`` mask.  Non-keyframes still
    # contribute their camera pose to the trajectory/overlay.
    keyframes_only_points: bool = False


@dataclass
class PreprocessConfig:
    mask_sky: bool = False
    sky_model: str = 'skyseg.onnx'
    sky_batch_size: int = 32
    conf_threshold: float = 0.0
    vis_threshold: float = 4.0
    sky_mask_dir: Optional[str] = None
    sky_mask_visualization_dir: Optional[str] = None


@dataclass
class CameraSegment:
    mode: str = 'follow'          # follow / birdeye / static / pivot
    frames: List[int] = field(default_factory=list)  # [start, end], -1 = last frame
    # follow params
    back_offset: float = 0.4
    up_offset: float = 0.1
    look_offset: float = 0.5
    smooth_window: int = 40
    scale_frames: int = 100
    # birdeye params
    reveal_height_mult: float = 2.0
    # static / pivot params (None = auto from start frame pose)
    eye: Optional[List[float]] = None
    lookat: Optional[List[float]] = None


@dataclass
class CameraConfig:
    fov: float = 60.0
    transition: int = 30
    segments: List[CameraSegment] = field(default_factory=list)
    # empty segments → default full-range follow


@dataclass
class RenderConfig:
    width: int = 1920
    height: int = 1080
    point_size: float = 1.0
    near: float = 0.1
    far: float = 100.0
    background: str = 'black'
    depth_video: bool = False
    depth_colormap: str = 'turbo'
    depth_percentile_lo: float = 1.0
    depth_percentile_hi: float = 99.0
    edl: bool = False
    edl_strength: float = 0.5
    edl_radius: int = 2
    lod_target_pixels: float = 1.5
    combined_video: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'RenderConfig':
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


@dataclass
class OverlayConfig:
    camera_vis: str = ''

    # Trail style
    trail_enabled: bool = True
    trail_tail_len: int = 50
    trail_line_width: float = 2.0
    trail_color_ramp: str = 'cyan_blue'

    # Head style
    head_num_frames: int = 8
    head_point_size: float = 8.0
    head_frustum_scale: float = 0.05
    head_frustum_line_width: float = 1.5
    head_frustum_color: str = ''            # "" = follow trail; "#RRGGBB" = custom
    head_texture_alpha: float = 0.8

    # Frame tag
    frame_tag: bool = False
    frame_tag_position: str = 'top_left'


@dataclass
class GpuConfig:
    build_batch_size: int = 0     # 0 = auto
    cull_chunk_size: int = 0      # 0 = auto
    memory_limit_gb: float = 0    # 0 = auto (85% total)


@dataclass
class PipelineConfig:
    input: str = ''
    output: str = 'scan_render.mp4'
    fps: int = 30
    num_workers: int = 16
    skip_first: int = 0           # drop first K frames before any other filtering
    fast_review: int = 200        # 0 = render all frames
    frame_stride: int = 1         # 1 = every frame, N = every N-th frame
    hd_image_folder: str = ''     # folder with original high-res frames for RGB video

    scene: SceneConfig = field(default_factory=SceneConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)

    # -----------------------------------------------------------------
    # YAML I/O
    # -----------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> 'PipelineConfig':
        """Load config from YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls._from_dict(raw)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False,
                      sort_keys=False, allow_unicode=True)

    # -----------------------------------------------------------------
    # CLI merge (backward compatible flat args)
    # -----------------------------------------------------------------

    def merge_cli(self, args) -> 'PipelineConfig':
        """Merge argparse namespace into config. Returns new config (no mutation).

        Supports both flat CLI args (--voxel_size) and dotted (--scene.voxel_size).
        Flat args are mapped to their nested location automatically.
        Only non-default values from argparse override the config.
        """
        cfg = copy.deepcopy(self)
        if args is None:
            return cfg

        ns = vars(args) if hasattr(args, '__dict__') else dict(args)

        # Flat CLI → nested path mapping
        _FLAT_MAP = {
            # I/O
            'input_npz': 'input',
            'output_video': 'output',
            'fps': 'fps',
            'num_workers': 'num_workers',
            'skip_first': 'skip_first',
            'debug_frames': 'fast_review',
            'frame_stride': 'frame_stride',
            'hd_image_folder': 'hd_image_folder',
            # scene
            'voxel_size': 'scene.voxel_size',
            'octree_level': 'scene.octree_level',
            'max_depth': 'scene.max_depth',
            'downsample': 'scene.downsample',
            'color_update': 'scene.color_update',
            'no_jitter': '_no_jitter',  # special handling
            'keyframes_only_points': 'scene.keyframes_only_points',
            # preprocess
            'mask_sky': 'preprocess.mask_sky',
            'sky_model': 'preprocess.sky_model',
            'sky_batch_size': 'preprocess.sky_batch_size',
            'sky_mask_dir': 'preprocess.sky_mask_dir',
            'sky_mask_visualization_dir': 'preprocess.sky_mask_visualization_dir',
            'conf_threshold': 'preprocess.conf_threshold',
            'vis_threshold': 'preprocess.vis_threshold',
            # camera
            'fov': 'camera.fov',
            'smooth_window': '_follow_smooth_window',
            'back_offset': '_follow_back_offset',
            'up_offset': '_follow_up_offset',
            'look_offset': '_follow_look_offset',
            'follow_scale_frames': '_follow_scale_frames',
            'birdeye_start': '_birdeye_start',
            'birdeye_duration': '_birdeye_duration',
            'birdeye_transition': 'camera.transition',
            'reveal_height_mult': '_birdeye_reveal',
            # render
            'render_width': 'render.width',
            'render_height': 'render.height',
            'point_size': 'render.point_size',
            'near': 'render.near',
            'far': 'render.far',
            'background': 'render.background',
            'depth_video': 'render.depth_video',
            'no_combined_video': '_no_combined_video',
            'depth_colormap': 'render.depth_colormap',
            'depth_percentile_lo': 'render.depth_percentile_lo',
            'depth_percentile_hi': 'render.depth_percentile_hi',
            'edl': 'render.edl',
            'edl_strength': 'render.edl_strength',
            'edl_radius': 'render.edl_radius',
            'lod_target_pixels': 'render.lod_target_pixels',
            # overlay
            'camera_vis': 'overlay.camera_vis',
            'trail_enabled': 'overlay.trail_enabled',
            'trail_tail_len': 'overlay.trail_tail_len',
            'trail_line_width': 'overlay.trail_line_width',
            'trail_color_ramp': 'overlay.trail_color_ramp',
            'head_num_frames': 'overlay.head_num_frames',
            'head_point_size': 'overlay.head_point_size',
            'head_frustum_scale': 'overlay.head_frustum_scale',
            'head_frustum_line_width': 'overlay.head_frustum_line_width',
            'head_frustum_color': 'overlay.head_frustum_color',
            'head_texture_alpha': 'overlay.head_texture_alpha',
            'frame_tag': 'overlay.frame_tag',
            'frame_tag_position': 'overlay.frame_tag_position',
        }

        # Apply dotted args first (e.g. --scene.voxel_size 0.005)
        for key, val in ns.items():
            if '.' in key and val is not None:
                _set_nested(cfg, key, val)

        # Apply flat args via mapping
        for cli_key, cfg_path in _FLAT_MAP.items():
            if cli_key not in ns:
                continue
            val = ns[cli_key]
            if val is None:
                continue
            if cfg_path.startswith('_'):
                continue  # handled below
            _set_nested(cfg, cfg_path, val)

        # Special: no_jitter → scene.jitter
        if ns.get('no_jitter'):
            cfg.scene.jitter = False

        # Special: no_combined_video → render.combined_video
        if ns.get('no_combined_video'):
            cfg.render.combined_video = False

        # Special: build camera segments from flat birdeye/follow args
        # Only if no segments already defined via YAML
        if not cfg.camera.segments:
            cfg.camera.segments = _build_segments_from_flat_args(ns)

        return cfg

    # -----------------------------------------------------------------
    # Internal dict conversion
    # -----------------------------------------------------------------

    @classmethod
    def _from_dict(cls, d: dict) -> 'PipelineConfig':
        cfg = cls()
        # Top-level scalars
        for key in ('input', 'output', 'fps', 'num_workers', 'fast_review'):
            if key in d:
                setattr(cfg, key, d[key])

        if 'scene' in d:
            cfg.scene = _dataclass_from_dict(SceneConfig, d['scene'])
        if 'preprocess' in d:
            cfg.preprocess = _dataclass_from_dict(PreprocessConfig, d['preprocess'])
        if 'camera' in d:
            cam_d = d['camera']
            cfg.camera = CameraConfig(
                fov=cam_d.get('fov', cfg.camera.fov),
                transition=cam_d.get('transition', cfg.camera.transition),
                segments=[_dataclass_from_dict(CameraSegment, s)
                          for s in cam_d.get('segments', [])],
            )
        if 'render' in d:
            cfg.render = _dataclass_from_dict(RenderConfig, d['render'])
        if 'overlay' in d:
            cfg.overlay = _dataclass_from_dict(OverlayConfig, d['overlay'])
        if 'gpu' in d:
            cfg.gpu = _dataclass_from_dict(GpuConfig, d['gpu'])
        if 'pipeline' in d:
            # Alternative: pipeline section for fps/num_workers/fast_review
            p = d['pipeline']
            for key in ('fps', 'num_workers', 'fast_review'):
                if key in p:
                    setattr(cfg, key, p[key])
        return cfg

    def _to_dict(self) -> dict:
        d = {}
        d['input'] = self.input
        d['output'] = self.output
        d['scene'] = asdict(self.scene)
        d['preprocess'] = asdict(self.preprocess)
        d['camera'] = {
            'fov': self.camera.fov,
            'transition': self.camera.transition,
            'segments': [asdict(s) for s in self.camera.segments],
        }
        d['render'] = asdict(self.render)
        d['overlay'] = asdict(self.overlay)
        d['pipeline'] = {
            'fps': self.fps,
            'num_workers': self.num_workers,
            'fast_review': self.fast_review,
        }
        d['gpu'] = asdict(self.gpu)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataclass_from_dict(cls, d: dict):
    """Create dataclass instance from dict, ignoring unknown keys."""
    valid = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def _set_nested(obj, path: str, value):
    """Set a nested attribute via dotted path, e.g. 'scene.voxel_size'."""
    parts = path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _build_segments_from_flat_args(ns: dict) -> List[CameraSegment]:
    """Build camera segments from legacy flat CLI args.

    Handles the old --birdeye_start/--birdeye_duration/follow params.
    If no birdeye args, returns a single follow segment with the given params.
    """
    follow_kwargs = {}
    for key, attr in [('smooth_window', 'smooth_window'),
                      ('back_offset', 'back_offset'),
                      ('up_offset', 'up_offset'),
                      ('look_offset', 'look_offset'),
                      ('follow_scale_frames', 'scale_frames')]:
        if key in ns and ns[key] is not None:
            follow_kwargs[attr] = ns[key]

    birdeye_start = ns.get('birdeye_start', '')
    birdeye_duration = ns.get('birdeye_duration', '')
    reveal = ns.get('reveal_height_mult') or 2.0

    if not birdeye_start or not birdeye_duration:
        # Single follow segment covering all frames
        return [CameraSegment(mode='follow', frames=[0, -1], **follow_kwargs)]

    # Parse birdeye windows → create segment sequence
    starts = [int(x) for x in birdeye_start.split(',') if x.strip()]
    durs = [int(x) for x in birdeye_duration.split(',') if x.strip()]

    segments = []
    prev_end = 0

    for start, dur in zip(starts, durs):
        # Follow segment before birdeye
        if start > prev_end:
            segments.append(CameraSegment(
                mode='follow', frames=[prev_end, start], **follow_kwargs))
        # Birdeye segment
        segments.append(CameraSegment(
            mode='birdeye', frames=[start, start + dur],
            reveal_height_mult=reveal))
        prev_end = start + dur

    # Follow segment after last birdeye
    segments.append(CameraSegment(
        mode='follow', frames=[prev_end, -1], **follow_kwargs))

    return segments


def load_config(args=None) -> PipelineConfig:
    """Convenience: load from --config YAML if given, then merge CLI args."""
    if args is None:
        return PipelineConfig()

    config_path = getattr(args, 'config', None)
    if config_path:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()

    return cfg.merge_cli(args)
