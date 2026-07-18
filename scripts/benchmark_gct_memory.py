#!/usr/bin/env python3
"""Benchmark LingBot-Map/GCT streaming inference GPU memory.

The script uses synthetic inputs and random initialized weights by default, so
it measures the architecture's inference memory behavior without checkpoint IO.
It sweeps sequence lengths and writes CSV rows with peak CUDA memory, runtime,
and OOM status.

Example:
    cd /home/clz/code/lingbot-map
    python scripts/benchmark_gct_memory.py \
        --height 384 --width 518 \
        --frame-counts 64 128 256 512 1024 2048 4096 10000 \
        --output gct_memory_h800_synthetic.csv \
        --stop-on-oom

Note:
    GCT uses patch size 14. Requested 384x518 is automatically adjusted to
    378x518 by default because 384 is not divisible by 14. The CSV records both
    requested and effective sizes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch


# Match demo.py allocator behavior unless the user has already set it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=384, help="Requested synthetic height.")
    parser.add_argument("--width", type=int, default=518, help="Requested synthetic width.")
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument(
        "--adjust-to-patch",
        choices=["floor", "ceil", "error"],
        default="floor",
        help="How to handle H/W not divisible by patch size.",
    )
    parser.add_argument(
        "--frame-counts",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512, 1024, 2048, 4096, 10000],
    )
    parser.add_argument("--num-scale-frames", type=int, default=8)
    parser.add_argument("--sliding-window", type=int, default=64)
    parser.add_argument("--keyframe-interval", type=int, default=1)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument(
        "--backend",
        choices=["flashinfer", "sdpa"],
        default="flashinfer",
        help="Attention backend. flashinfer matches the fast inference path.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
        help="Autocast dtype.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-source-images",
        type=int,
        default=8,
        help="Number of synthetic tensors to cycle when --materialize-inputs is off.",
    )
    parser.add_argument(
        "--materialize-inputs",
        action="store_true",
        help="Allocate one distinct input tensor per frame on CPU before streaming.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile hot modules as in gct_profile.py. Mostly useful for speed, not memory.",
    )
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "gct_memory_h800_synthetic.csv",
    )
    parser.add_argument("--stop-on-oom", action="store_true")
    parser.add_argument(
        "--keep-model-between-counts",
        action="store_true",
        help="Reuse one model across frame counts. Default is to rebuild per count.",
    )
    return parser.parse_args()


def patch_align(value: int, patch: int, mode: str) -> int:
    if value % patch == 0:
        return value
    if mode == "error":
        raise ValueError(f"{value} is not divisible by patch size {patch}")
    if mode == "floor":
        return (value // patch) * patch
    return ((value + patch - 1) // patch) * patch


def resolve_dtype(device: str, dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def make_autocast(dtype: torch.dtype):
    if dtype == torch.float32:
        return contextlib.nullcontext()
    return torch.amp.autocast("cuda", dtype=dtype)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_model(args: argparse.Namespace, max_frame_num: int, device: str):
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.width,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=max_frame_num,
        kv_cache_sliding_window=args.sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=(args.backend == "sdpa"),
        camera_num_iterations=args.camera_num_iterations,
    )
    return model.eval().to(device)


def compile_model(model) -> None:
    agg = model.aggregator
    for i, block in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(block, mode="reduce-overhead")
    for i, block in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(block, mode="reduce-overhead")
    for block in agg.global_blocks:
        if hasattr(block, "attn_pre"):
            block.attn_pre = torch.compile(block.attn_pre, mode="reduce-overhead")
        if hasattr(block, "ffn_residual"):
            block.ffn_residual = torch.compile(block.ffn_residual, mode="reduce-overhead")
        block.attn.proj = torch.compile(block.attn.proj, mode="reduce-overhead")


def make_source_images(
    *,
    frame_count: int,
    source_count: int,
    height: int,
    width: int,
    materialize_inputs: bool,
) -> torch.Tensor:
    count = frame_count if materialize_inputs else min(source_count, frame_count)
    torch.manual_seed(42)
    # Keep on CPU; each frame is moved to GPU right before forward. This avoids
    # counting 10k raw inputs as persistent GPU memory unless activations/cache
    # actually need them.
    return torch.randn(1, count, 3, height, width, dtype=torch.float32, device="cpu")


def get_frame(images: torch.Tensor, frame_idx: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    idx = frame_idx % images.shape[1]
    return images[:, idx : idx + 1].to(device=device, dtype=dtype, non_blocking=True)


@torch.no_grad()
def run_streaming_once(
    *,
    model,
    images: torch.Tensor,
    frame_count: int,
    dtype: torch.dtype,
    device: str,
    num_scale_frames: int,
    keyframe_interval: int,
) -> tuple[float, int, int]:
    cleanup()
    torch.cuda.synchronize(device)
    model.clean_kv_cache()
    start = time.perf_counter()

    scale_frames = min(num_scale_frames, frame_count)
    scale_batch = torch.cat(
        [get_frame(images, i, dtype, device) for i in range(scale_frames)],
        dim=1,
    )
    torch.compiler.cudagraph_mark_step_begin()
    with make_autocast(dtype):
        model.forward(
            scale_batch,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
    del scale_batch

    kf_int = max(int(keyframe_interval), 1)
    for i in range(scale_frames, frame_count):
        is_keyframe = (kf_int <= 1) or ((i - scale_frames) % kf_int == 0)
        if not is_keyframe:
            model._set_skip_append(True)
        frame = get_frame(images, i, dtype, device)
        torch.compiler.cudagraph_mark_step_begin()
        with make_autocast(dtype):
            model.forward(
                frame,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
        del frame
        if not is_keyframe:
            model._set_skip_append(False)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    model.clean_kv_cache()
    cleanup()
    return elapsed, peak_alloc, peak_reserved


def gib(num_bytes: int) -> float:
    return num_bytes / (1024**3)


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "frames",
        "status",
        "requested_height",
        "requested_width",
        "effective_height",
        "effective_width",
        "backend",
        "dtype",
        "seconds",
        "peak_allocated_gib",
        "peak_reserved_gib",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device = args.device
    dtype = resolve_dtype(device, args.dtype)
    eff_h = patch_align(args.height, args.patch_size, args.adjust_to_patch)
    eff_w = patch_align(args.width, args.patch_size, args.adjust_to_patch)
    if (eff_h, eff_w) != (args.height, args.width):
        print(
            f"Requested {args.height}x{args.width}; using patch-aligned "
            f"{eff_h}x{eff_w} (patch={args.patch_size})."
        )

    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Backend: {args.backend}")
    print(f"Dtype: {dtype}")
    print(f"Output: {args.output}")

    rows: list[dict] = []
    shared_model = None
    if args.keep_model_between_counts:
        shared_model = build_model(
            args,
            max_frame_num=max(args.frame_counts) + 100,
            device=device,
        )
        if args.compile:
            compile_model(shared_model)

    for frame_count in args.frame_counts:
        print(f"\nBenchmarking GCT {frame_count} frames")
        try:
            cleanup()
            if args.keep_model_between_counts:
                model = shared_model
                model.clean_kv_cache()
            else:
                model = build_model(args, max_frame_num=frame_count + 100, device=device)
                if args.compile:
                    compile_model(model)
            images = make_source_images(
                frame_count=frame_count,
                source_count=args.num_source_images,
                height=eff_h,
                width=eff_w,
                materialize_inputs=args.materialize_inputs,
            )

            if args.warmup_frames > 0:
                warm_frames = min(args.warmup_frames, frame_count)
                run_streaming_once(
                    model=model,
                    images=images,
                    frame_count=warm_frames,
                    dtype=dtype,
                    device=device,
                    num_scale_frames=min(args.num_scale_frames, warm_frames),
                    keyframe_interval=args.keyframe_interval,
                )

            elapsed, peak_alloc, peak_reserved = run_streaming_once(
                model=model,
                images=images,
                frame_count=frame_count,
                dtype=dtype,
                device=device,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
            del images
            if not args.keep_model_between_counts:
                del model
            cleanup()
            row = {
                "frames": frame_count,
                "status": "ok",
                "requested_height": args.height,
                "requested_width": args.width,
                "effective_height": eff_h,
                "effective_width": eff_w,
                "backend": args.backend,
                "dtype": str(dtype).replace("torch.", ""),
                "seconds": f"{elapsed:.4f}",
                "peak_allocated_gib": f"{gib(peak_alloc):.4f}",
                "peak_reserved_gib": f"{gib(peak_reserved):.4f}",
                "error": "",
            }
            print(
                f"OK frames={frame_count} time={row['seconds']}s "
                f"alloc={row['peak_allocated_gib']}GiB "
                f"reserved={row['peak_reserved_gib']}GiB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            if "model" in locals() and model is not None:
                model.clean_kv_cache()
            if "images" in locals():
                del images
            if (
                "model" in locals()
                and model is not None
                and not args.keep_model_between_counts
            ):
                del model
            cleanup()
            row = {
                "frames": frame_count,
                "status": "oom",
                "requested_height": args.height,
                "requested_width": args.width,
                "effective_height": eff_h,
                "effective_width": eff_w,
                "backend": args.backend,
                "dtype": str(dtype).replace("torch.", ""),
                "seconds": "",
                "peak_allocated_gib": "",
                "peak_reserved_gib": "",
                "error": str(exc).replace("\n", " ")[:500],
            }
            print(f"OOM frames={frame_count}: {row['error']}")
            rows.append(row)
            write_rows(args.output, rows)
            if args.stop_on_oom:
                break
            continue
        rows.append(row)
        write_rows(args.output, rows)

    print(f"\nWrote {args.output}")
    if shared_model is not None:
        shared_model.clean_kv_cache()
        del shared_model
        cleanup()


if __name__ == "__main__":
    main()
