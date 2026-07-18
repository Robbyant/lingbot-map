"""
LingbotMap method - Streaming 3D reconstruction with causal transformer.

Wraps the upstream ``lingbot-map`` package (``methods/lingbot-map_repo``,
imported as the ``lingbot_map`` Python module) for benchmark evaluation.
Both streaming and windowed inference modes are supported via ``GCTStream``.
"""

import logging
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from benchmark.method.base import BaseMethod
from benchmark.core.loader import BSSLoader


# Mirrors lingbot-map/demo.py:413-421 — above this frame count, the KV cache
# grows unbounded, so auto-bump the keyframe interval. Used as the default for
# the configurable ``auto_keyframe_threshold`` constructor argument.
_DEFAULT_AUTO_KEYFRAME_THRESHOLD = 320


def _resolve_keyframe_interval(
    cfg_val, num_frames: int, threshold: int = _DEFAULT_AUTO_KEYFRAME_THRESHOLD
) -> int:
    """Resolve a raw config value into a concrete keyframe interval.

    ``None``, ``0``, or the string ``"auto"`` triggers auto-selection:
    ``1`` when ``num_frames <= threshold`` else ``ceil(num_frames / threshold)``.
    An explicit positive int is returned as-is.
    """
    if cfg_val is None or cfg_val == 0 or (isinstance(cfg_val, str) and cfg_val.lower() == "auto"):
        if num_frames <= threshold:
            return 1
        return (num_frames + threshold - 1) // threshold
    return int(cfg_val)


class LingbotMapMethod(BaseMethod):
    """
    LingbotMap model adapter for benchmark evaluation.

    Supports streaming and windowed inference modes via the upstream
    ``GCTStream`` model exposed by the ``lingbot_map`` package.
    """

    def __init__(
        self,
        checkpoint: str = None,
        device: str = 'cuda',
        mode: str = 'streaming',
        use_amp: bool = True,
        use_sdpa: bool = False,
        image_size: int = 518,
        patch_size: int = 14,
        enable_3d_rope: bool = True,
        num_scale_frames: int = 8,
        max_frame_num: int = 1024,
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        window_size: int = 64,
        overlap_size: Optional[int] = None,
        keyframe_interval: Any = "auto",
        auto_keyframe_threshold: int = _DEFAULT_AUTO_KEYFRAME_THRESHOLD,
        flow_threshold: float = 0.0,
        max_non_keyframe_gap: int = 30,
        align: int = 14,
        area_budget: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ):
        super().__init__(
            align=align,
            area_budget=area_budget,
            logger=logger,
        )

        self.checkpoint = checkpoint
        self.device = device
        self.mode = mode
        self.use_amp = use_amp
        self.use_sdpa = use_sdpa
        self.image_size = image_size
        self.patch_size = patch_size
        self.enable_3d_rope = enable_3d_rope
        self.num_scale_frames = num_scale_frames
        self.max_frame_num = max_frame_num
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.window_size = window_size
        self.overlap_size = overlap_size
        self.keyframe_interval = keyframe_interval
        self.auto_keyframe_threshold = int(auto_keyframe_threshold)
        self.flow_threshold = flow_threshold
        self.max_non_keyframe_gap = max_non_keyframe_gap

        if self.mode not in ('streaming', 'windowed'):
            raise ValueError(f"Invalid mode '{self.mode}'. Must be 'streaming' or 'windowed'")

        if self.auto_keyframe_threshold <= 0:
            raise ValueError(
                f"auto_keyframe_threshold must be a positive int, got {self.auto_keyframe_threshold}"
            )

        self._load_model()

    def _load_model(self):
        """Load LingbotMap (GCTStream) model from checkpoint."""
        if self.mode == 'windowed':
            from lingbot_map.models.gct_stream_window import GCTStream
        else:
            from lingbot_map.models.gct_stream import GCTStream

        print(f"  → Building LingbotMap model (mode: {self.mode})")
        self.model = GCTStream(
            img_size=self.image_size,
            patch_size=self.patch_size,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=self.use_sdpa,
        )

        if self.checkpoint:
            print(f"  → Loading checkpoint: {self.checkpoint}")
            ckpt = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"    Missing keys: {len(missing)}")
            if unexpected:
                print(f"    Unexpected keys: {len(unexpected)}")
            print("    Checkpoint loaded.")

        self.model = self.model.to(self.device).eval()

    def _prepare_images(self, rgb_list):
        """Convert list of HxWx3 uint8 numpy arrays to [S, 3, H, W] tensor in [0, 1]."""
        from torchvision import transforms as TF

        to_tensor = TF.ToTensor()
        images = torch.stack([to_tensor(rgb) for rgb in rgb_list])
        return images.to(self.device)

    def _run_inference(self, images):
        """Run LingbotMap inference and return raw predictions dict."""
        if self.use_amp:
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32

        print(f"  → Running {self.mode} inference (dtype: {dtype})")

        num_frames = images.shape[0]
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            if self.mode == 'streaming':
                keyframe_interval = _resolve_keyframe_interval(
                    self.keyframe_interval, num_frames, self.auto_keyframe_threshold
                )
                if keyframe_interval != self.keyframe_interval:
                    print(
                        f"  → Auto-selected keyframe_interval={keyframe_interval} "
                        f"(num_frames={num_frames}, raw={self.keyframe_interval!r}, "
                        f"threshold={self.auto_keyframe_threshold})"
                    )
                predictions = self.model.inference_streaming(
                    images,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=keyframe_interval,
                    output_device=torch.device("cpu"),
                )
            else:
                predictions = self.model.inference_windowed(
                    images,
                    window_size=self.window_size,
                    overlap_size=self.overlap_size,
                    num_scale_frames=self.num_scale_frames,
                    keyframe_interval=self.keyframe_interval,
                    flow_threshold=self.flow_threshold,
                    max_non_keyframe_gap=self.max_non_keyframe_gap,
                    output_device=torch.device("cpu"),
                )

        return predictions

    def _process_outputs(self, predictions, image_shape):
        """Convert model predictions to benchmark output format.

        Args:
            predictions: Raw model outputs with 'pose_enc', 'depth', 'depth_conf', etc.
            image_shape: (H, W) of the processed images.

        Returns:
            Tuple of (rgb_list, depth_list, pose_list, intrinsics_list, confidence_list)
        """
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

        # Decode pose encoding to extrinsic + intrinsic
        # pose_encoding_to_extri_intri() output is C2W directly (no inverse needed)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_shape
        )

        extrinsic = extrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 4]
        intrinsic = intrinsic.float().cpu().numpy().squeeze(0)  # [S, 3, 3]
        depth = predictions["depth"].float().cpu().numpy().squeeze(0)  # [S, H, W, 1]

        # Extract processed images
        if "images" in predictions:
            images = predictions["images"].float().cpu().numpy().squeeze(0)  # [S, 3, H, W]
        else:
            images = None

        num_frames = extrinsic.shape[0]
        print(f"  → Extracting {num_frames} frames")

        rgb_list = []
        depth_list = []
        pose_list = []
        intrinsics_list = []
        confidence_list = []

        for i in range(num_frames):
            # RGB: [3, H, W] float [0,1] -> [H, W, 3] uint8
            if images is not None:
                rgb = images[i].transpose(1, 2, 0)
                rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
                rgb_list.append(rgb)

            # Pose: 3x4 C2W -> 4x4 C2W
            pose = np.eye(4, dtype=np.float32)
            pose[:3, :] = extrinsic[i].astype(np.float32)
            pose_list.append(pose)

            # Intrinsics: 3x3 K -> [fx, fy, cx, cy]
            K = intrinsic[i]
            intrinsics_list.append(np.array(
                [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32
            ))

            # Depth: [H, W, 1] -> [H, W]
            depth_frame = depth[i]
            if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
                depth_frame = depth_frame.squeeze(-1)
            depth_list.append(depth_frame.astype(np.float32))

            # Confidence
            if "depth_conf" in predictions:
                conf = predictions["depth_conf"][0, i].float().cpu().numpy()
                confidence_list.append(conf.astype(np.float32))

        return rgb_list, depth_list, pose_list, intrinsics_list, confidence_list

    def process_scene(self, gt_artifact) -> Dict[str, Any]:
        """Process a scene with LingbotMap inference."""
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        input_rgb_list = loader.load_rgb_list()
        self.logger.info(f"Image size for processing: {loader.get_processing_dimensions()} (HxW)")

        print(f"  → Processing {len(input_rgb_list)} frames with LingbotMap (mode: {self.mode})")

        # Prepare and run inference
        images = self._prepare_images(input_rgb_list)
        image_shape = images.shape[-2:]  # (H, W)
        predictions = self._run_inference(images)

        # Convert outputs
        rgb_list, depth_list, pose_list, intrinsics_list, confidence_list = \
            self._process_outputs(predictions, image_shape)

        if len(depth_list) != len(input_rgb_list):
            print(f"  → WARNING: Output frames ({len(depth_list)}) != input frames ({len(input_rgb_list)})")

        # Assemble results
        print(f"  → Assembling {len(rgb_list)} frames in standard format")
        frame_results = {
            'rgb': rgb_list,
            'depth': depth_list,
            'pose': pose_list,
            'intrinsics': intrinsics_list,
        }

        if confidence_list:
            frame_results['confidence'] = confidence_list

        return {
            'frame': frame_results,
            'global': {},
        }
