"""Install the corrected BF16 benchmark path on a fresh, private model."""
import types

from lingbot_map.optimizations.thor.options import ThorOptions


def _get_manager(self, device, dtype, tokens_per_frame=None):
    from lingbot_map.optimizations.thor.cache import FlashInferKVCacheManager
    if self.kv_cache_manager is None:
        tokens = tokens_per_frame or (self.img_size // self.patch_size) ** 2 + self.num_special_tokens
        self._thor_options.validate(self.kv_cache_sliding_window)
        self.kv_cache_manager = FlashInferKVCacheManager(
            num_blocks=self.depth,
            max_num_frames=self.kv_cache_scale_frames + self.kv_cache_sliding_window + 16,
            tokens_per_frame=tokens, num_heads=self.embed_dim // 64, head_dim=64,
            dtype=dtype, device=device, num_special_tokens=self.num_special_tokens,
            scale_frames=self.kv_cache_scale_frames, sliding_window=self.kv_cache_sliding_window,
            max_total_frames=self.max_frame_num + 100, backend="fa4",
        )
        if self._thor_options.visible_window is not None:
            self.kv_cache_manager.set_visible_window(self._thor_options.visible_window)
    return self.kv_cache_manager


def install_runtime(model, options: ThorOptions, image_shape):
    """Install only on a fresh eval model, before warmup or torch.compile.

    Rollback uses a fresh model/process. In-place training, checkpoint loading,
    device moves and concurrent sequences after installation are unsupported.
    """
    import torch
    from lingbot_map.optimizations.thor import blocks
    from .depth import patch_predict_depth_bf16
    if model.training or hasattr(model, "_thor_runtime_installed"):
        raise ValueError("Thor runtime requires a fresh eval model")
    agg = model.aggregator
    if agg.use_sdpa or agg.kv_cache_manager is not None:
        raise ValueError("Install before KV allocation on the FlashInfer model path")
    if getattr(model, "point_head", None) is not None or getattr(model, "local_point_head", None) is not None:
        raise ValueError("This runner supports camera/depth outputs, not point/local-point heads")
    if model.camera_head is None or model.depth_head is None:
        raise ValueError("Both camera and depth heads must remain enabled")
    first = next(model.parameters())
    if first.device.type != "cuda" or torch.cuda.get_device_capability(first.device) != (11, 0):
        raise ValueError("Thor runtime requires SM110 CUDA parameters")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Load original FP32 parameters; inference uses BF16 autocast")
    h, w = image_shape
    if h % 14 or w % 14:
        raise ValueError("image_shape must be divisible by patch size 14")
    if agg.depth != 24 or agg.embed_dim != 1024 or agg.num_special_tokens != 6:
        raise ValueError("Thor runtime supports 24 blocks, hidden dim 1024 and 6 special tokens")
    options.validate(agg.kv_cache_sliding_window)
    options.activate()
    agg._thor_options = options
    agg._get_flashinfer_manager = types.MethodType(_get_manager, agg)
    for block in agg.global_blocks:
        for name in ("attn_post", "attn_post_ffn", "forward"):
            setattr(block, name, types.MethodType(getattr(blocks, name), block))
        block.attn.prepare_qkv = types.MethodType(blocks.prepare_qkv, block.attn)
    for block in [*agg.frame_blocks, *agg.patch_embed.blocks]:
        block.attn.forward_with_qkv_linear = types.MethodType(blocks.forward_with_qkv_linear, block.attn)
    patch_predict_depth_bf16(model)
    from .position import install_historical_positions
    install_historical_positions(model)
    model._thor_runtime_installed = True
    model._thor_image_shape = (h, w)
    return model


def prepare_model(model, images, options: ThorOptions, *, compile_model=True):
    """Warm the explicit runtime, apply selected options, then optionally compile."""
    import torch
    from lingbot_map.optimizations.thor import weight_cache as weights
    from lingbot_map.optimizations.thor.projection import apply_projection_weight_cache, validate_housekeeping_runtime
    if images.ndim not in (4, 5) or images.shape[-3] != 3:
        raise ValueError("Expected [frames,3,H,W] or [1,frames,3,H,W]")
    if images.ndim == 5 and images.shape[0] != 1:
        raise ValueError("Thor runner supports batch one")
    available = images.shape[1] if images.ndim == 5 else images.shape[0]
    if available < 19:
        raise ValueError("At least 19 frames are required before installing the runtime")
    install_runtime(model, options, tuple(images.shape[-2:]))
    # Apply patch/global QKV caches before eager setup, preserving the established
    # warmup order. The other caches are installed before compiling hot paths.
    stats = {}
    from .execution import patch_compile_camera_trunk
    if compile_model:
        patch_compile_camera_trunk(model)
    if options.cache_mlp_weights:
        stats["patch_mlp_cache"] = vars(weights.apply_patch_mlp_weight_cache(model))
    if options.cache_qkv_weights:
        stats["patch_qkv_cache"] = vars(weights.apply_patch_qkv_weight_cache(model))
        stats["global_qkv_cache"] = vars(weights.apply_global_qkv_weight_cache(model))
    if options.cache_projection_weights:
        stats["projection_cache"] = vars(apply_projection_weight_cache(model))
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.shape[1] < 19:
        raise ValueError("Capture validation requires at least 8 scale + 10 warm + 1 replay frames")
    device = next(model.parameters()).device
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model(images[:, :8].to(device), num_frame_for_scale=8, num_frame_per_block=8, causal_inference=True)
        for index in range(8, 18):
            model(images[:, index:index+1].to(device), num_frame_for_scale=8, num_frame_per_block=1, causal_inference=True)
    torch.cuda.synchronize()
    if options.cache_mlp_weights:
        stats["global_mlp_cache"] = vars(weights.apply_global_mlp_weight_cache(model))
        stats["frame_mlp_cache"] = vars(weights.apply_frame_mlp_weight_cache(model))
    if options.cache_qkv_weights:
        stats["frame_qkv_cache"] = vars(weights.apply_frame_qkv_weight_cache(model))
    expected_counts = {
        "global_mlp_cache": len(model.aggregator.global_blocks),
        "frame_mlp_cache": len(model.aggregator.frame_blocks),
        "patch_mlp_cache": len(model.aggregator.patch_embed.blocks),
        "frame_qkv_cache": len(model.aggregator.frame_blocks),
        "patch_qkv_cache": len(model.aggregator.patch_embed.blocks),
        "global_qkv_cache": len(model.aggregator.global_blocks),
    }
    for name, expected in expected_counts.items():
        if name in stats and stats[name]["blocks_patched"] != expected:
            raise RuntimeError(f"Incomplete {name}: {stats[name]}")
    if options.cache_projection_weights and stats["projection_cache"]["modules_patched"] != sum((
        len(model.aggregator.frame_blocks), len(model.aggregator.global_blocks),
        len(model.aggregator.patch_embed.blocks),
    )):
        raise RuntimeError("Incomplete attention projection cache installation")
    validate_housekeeping_runtime(model)
    if compile_model:
        from .compile import compile_default
        torch._dynamo.config.recompile_limit = 256
        torch._dynamo.config.fail_on_recompile_limit_hit = True
        compile_default(model)
        for _ in range(3):
            model.clean_kv_cache()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model(images[:, :8].to(device), num_frame_for_scale=8, num_frame_per_block=8, causal_inference=True)
                for index in range(8, 18):
                    model(images[:, index:index+1].to(device), num_frame_for_scale=8, num_frame_per_block=1, causal_inference=True)
        torch.cuda.synchronize()
    return stats
