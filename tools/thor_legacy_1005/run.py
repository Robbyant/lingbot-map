"""Run the 1005-token streaming synthetic benchmark with static-stage CUDA Graphs.

Use from the repository root: python -m tools.thor_legacy_1005.run --help
No checkpoint is loaded. This is a synthetic workload, not demo throughput.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path

from lingbot_map.optimizations.thor.options import FLAGS, ThorOptions
from .records import RUNNER_CONTRACT_ID, summarize_timing, validate_record


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=19)
    parser.add_argument("--benchmark", action="store_true", help="Time without per-frame output collection")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--visible-window", type=int)
    for field in FLAGS:
        parser.add_argument("--" + field.replace("_", "-"), action="store_true")
    return parser


def initialize_workload(torch, model_class, frames):
    # CPU and CUDA generators are seeded together. Model creation MUST follow
    # CUDA FP32 image generation, matching the measured historical script.
    torch.manual_seed(42)
    images = torch.randn(1, frames, 3, 378, 518, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    model = model_class(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=frames + 100, kv_cache_sliding_window=64,
        kv_cache_scale_frames=8, camera_num_iterations=4, use_sdpa=False,
    ).eval().cuda()
    return model, images


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.frames < 19:
        parser.error("At least 19 frames: 8 scale + 10 warm + 1 replay")
    if args.out.exists():
        parser.error(f"Refusing to overwrite {args.out}")
    options = ThorOptions(**{field: getattr(args, field) for field in FLAGS}, visible_window=args.visible_window)
    options.validate()
    options.activate()
    os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    if importlib.metadata.version("flash-attn-4") != "4.0.0b14":
        raise RuntimeError("The all-off reference requires flash-attn-4==4.0.0b14")

    import torch
    from lingbot_map.models.gct_stream import GCTStream
    from .setup import prepare_model
    from .execution import (
        pose_depth_digests, prewarm_compiled_strict_depth_only_formal_route,
        profile_with_capture_camcompile, validate_formal_cublas_workspace_unset,
    )

    validate_formal_cublas_workspace_unset()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (11, 0):
        raise RuntimeError("This reproduction requires NVIDIA Thor SM110")
    import torch._inductor.fx_passes.joint_graph as joint_graph
    joint_graph.lazy_init()
    torch.cuda.synchronize()
    torch._dynamo.config.recompile_limit = 256
    torch._dynamo.config.fail_on_recompile_limit_hit = True
    model, images = initialize_workload(torch, GCTStream, args.frames)
    def parameter_state():
        return {name.replace("_orig_mod.", ""): (p.data_ptr(), p._version)
                for name, p in model.named_parameters()}

    def checkpoint_keys():
        return {name.replace("_orig_mod.", "") for name in model.state_dict()}

    parameters = parameter_state()
    state_keys = checkpoint_keys()
    print("Preparing BF16 streaming path and three unmeasured rehearsals", flush=True)
    stats = prepare_model(model, images, options)
    depth_impl = model.depth_head._forward_impl
    prewarm = prewarm_compiled_strict_depth_only_formal_route(
        model, images, torch.bfloat16, compiled_depth_forward_impl=depth_impl,
    )
    outputs = []

    def collect(index, output):
        values = pose_depth_digests(output)
        if not all(value["finite"] for value in values.values()):
            raise RuntimeError(f"Nonfinite output at frame {index}")
        outputs.append({"frame_index": index, **values})

    print("Timing streaming frames" if args.benchmark else "Checking streaming outputs", flush=True)
    result = profile_with_capture_camcompile(
        model, images, args.frames, torch.bfloat16,
        deterministic_captured_heads=True, compiled_depth_forward_impl=depth_impl,
        deterministic_prewarm=prewarm, output_collector=None if args.benchmark else collect,
    )
    intact = parameters == parameter_state()
    intact = intact and all(p.dtype == torch.float32 for p in model.parameters())
    model_state = {"original_fp32_parameters_intact": intact, "state_dict_keys_unchanged": checkpoint_keys() == state_keys}
    if not all(model_state.values()):
        raise RuntimeError(f"Original model state changed: {model_state}")
    report = {
        "runner_contract_id": RUNNER_CONTRACT_ID,
        "kind": "synthetic_profile" if args.benchmark else "synthetic_correctness_smoke",
        "options": options.to_dict(), "frames": args.frames, "seed": 42,
        "checkpoint": None, "initialization": "cuda_fp32_input_to_bf16_then_random_model",
        "image_shape": [378, 518], "tokens_per_frame": 1005,
        "scale_frames": 8, "warm_frames": 10, "replay_frames": args.frames - 18,
        "patch_stats": stats, "model_state": model_state, "outputs": outputs,
        "per_frame_output_collection": not args.benchmark,
        "prewarm": prewarm, "result": result,
        "timing": summarize_timing(result, args.frames) if args.benchmark else None,
        "environment": {"torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability())},
    }
    if args.benchmark or not (options.allow_approximate or options.visible_window is not None):
        validate_record(report, correctness=not args.benchmark)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    if args.benchmark:
        print(f"Streaming synthetic throughput: {report['timing']['fps']:.3f} FPS", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
