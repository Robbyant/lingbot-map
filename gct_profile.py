"""
Profile GCTStream streaming inference FPS.

Measures only the top-level model.forward() GPU time — no per-module hooks,
no inner breakdown. One CUDA event pair per frame, a single sync at the end.

Usage:
    python gct_profile.py --backend both --dtype bf16 --num_frames 500
"""

import argparse
import contextlib
import json

import numpy as np
import torch

from lingbot_map.models.gct_stream import GCTStream


# ============================================================================
# Model loading
# ============================================================================

def load_model(backend, img_size, sliding_window, max_frame_num, camera_num_iterations, device='cuda'):
    """Build GCTStream with random weights (no checkpoint). Eval mode on device."""
    model = GCTStream(
        img_size=img_size,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=max_frame_num,
        kv_cache_sliding_window=sliding_window,
        kv_cache_scale_frames=8,
        use_sdpa=(backend == 'sdpa'),
        camera_num_iterations=camera_num_iterations,
    )
    return model.eval().to(device)


def compile_model(model):
    """
    Apply torch.compile(mode="reduce-overhead") to compute-heavy, fixed-shape
    modules and drop point_head. Matches the optimizations from the original
    --compile path; omits per-block experiments that did not pay off.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")
    model.point_head = None  # saves ~5.9 ms/frame; not needed for FPS measurement


# ============================================================================
# Profiling — reuse one CUDA event pair, sync after every frame
# ============================================================================

def profile_streaming(model, images, num_frames, dtype):
    """
    Run streaming inference. Return (per_frame_ms, scale_frames, phase1_ms).

    Reuses a single CUDA event pair across frames and syncs after every frame
    (matches the original non-lightweight path — necessary to keep GPU clock /
    memory allocator behavior comparable run-to-run).
    """
    device = next(model.parameters()).device
    if images.ndim == 4:
        images = images.unsqueeze(0)
    images = images.to(dtype)
    S = min(images.shape[1], num_frames)
    scale_frames = min(8, S)

    autocast_ctx = (
        contextlib.nullcontext() if dtype == torch.float32
        else torch.amp.autocast('cuda', dtype=dtype)
    )

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    model.clean_kv_cache()

    # ── Phase 1: scale frames, bidirectional attention among themselves ────
    # Move data onto GPU BEFORE the start event so the host→device copy is
    # excluded from the measured forward time. `.to(device)` is a no-op if
    # `images` already lives on GPU.
    scale_batch = images[:, :scale_frames].to(device)
    start_ev.record()
    torch.compiler.cudagraph_mark_step_begin()
    with torch.no_grad(), autocast_ctx:
        model.forward(
            scale_batch,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
    end_ev.record()
    torch.cuda.synchronize()
    phase1_ms = start_ev.elapsed_time(end_ev)
    print(f"  Phase 1: {phase1_ms:.1f} ms for {scale_frames} scale frames")

    # ── Phase 2: causal streaming, one frame at a time ─────────────────────
    per_frame_ms = []
    for i in range(scale_frames, S):
        frame = images[:, i:i + 1].to(device)  # outside the timed region
        start_ev.record()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), autocast_ctx:
            model.forward(
                frame,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
        end_ev.record()
        torch.cuda.synchronize()
        per_frame_ms.append(start_ev.elapsed_time(end_ev))

    return per_frame_ms, scale_frames, phase1_ms


# ============================================================================
# Reporting
# ============================================================================

def summarize(per_frame_ms, scale_frames, phase1_ms, label):
    """Print global FPS (total time / total frames) + 10/50/90% windows + trace."""
    n = len(per_frame_ms)
    if n == 0:
        print(f"  [{label}]: no frames")
        return {}

    def avg_ms(pos, window=30):
        lo = max(0, pos - window)
        hi = min(n, pos + window + 1)
        return float(np.mean(per_frame_ms[lo:hi]))

    def fps(ms):
        return 1000.0 / ms if ms > 0 else 0.0

    # Global throughput: total wall time (Phase 1 + Phase 2) / total frames.
    total_frames = scale_frames + n
    total_ms = phase1_ms + float(np.sum(per_frame_ms))
    global_ms_per_frame = total_ms / total_frames
    global_fps = fps(global_ms_per_frame)

    # Per-region windowed averages (±30 frames) for how FPS drifts over time.
    p_lo = max(10, n // 10)
    p_mid = n // 2
    p_hi = n - max(1, n // 10)
    ms_lo, ms_mid, ms_hi = avg_ms(p_lo), avg_ms(p_mid), avg_ms(p_hi)

    print(f"\n  [{label}]  ({total_frames} total frames: {scale_frames} scale + {n} streaming)")
    print(f"    ── Global FPS ─────────────────────────────────────")
    print(f"      total time: {total_ms / 1000:.2f} s  "
          f"({phase1_ms:.1f} ms phase1 + {total_ms - phase1_ms:.1f} ms phase2)")
    print(f"      per frame : {global_ms_per_frame:6.2f} ms  →  {global_fps:6.2f} FPS")
    print(f"    ── Windowed FPS (±30 streaming frames) ────────────")
    print(f"      frame {scale_frames + p_lo:>5d} (10%): {ms_lo:6.2f} ms  →  {fps(ms_lo):6.2f} FPS")
    print(f"      frame {scale_frames + p_mid:>5d} (50%): {ms_mid:6.2f} ms  →  {fps(ms_mid):6.2f} FPS")
    print(f"      frame {scale_frames + p_hi:>5d} (90%): {ms_hi:6.2f} ms  →  {fps(ms_hi):6.2f} FPS")

    # Trace at global frame indices that are multiples of 100, matching the
    # original script. This naturally skips the cold first streaming frame
    # (global index = scale_frames), whose ms is dominated by one-time CUDA
    # graph (re)capture after `clean_kv_cache()` in profile_streaming.
    print(f"    ── FPS trace (every 100 global frames) ────────────")
    first_trace = (100 - scale_frames) % 100 or 100
    for i in range(first_trace, n, 100):
        ms_i = avg_ms(i, window=3)
        print(f"      frame {scale_frames + i:>5d}: {fps(ms_i):6.2f} FPS  ({ms_i:.2f} ms)")

    return {
        'global_fps': global_fps, 'global_ms': global_ms_per_frame,
        'total_ms': total_ms, 'total_frames': total_frames,
        'phase1_ms': phase1_ms,
        'ms_lo': ms_lo, 'ms_mid': ms_mid, 'ms_hi': ms_hi,
        'fps_lo': fps(ms_lo), 'fps_mid': fps(ms_mid), 'fps_hi': fps(ms_hi),
    }


def print_comparison(results):
    """Side-by-side FPS / ms table across all variants."""
    if len(results) < 2:
        return
    keys = sorted(results.keys())
    col = 14
    width = 18 + col * len(keys)
    print(f"\n{'=' * width}\n  Comparison\n{'=' * width}")
    print(f"  {'Metric':<18s}" + "".join(f"{k:>{col}s}" for k in keys))
    print("  " + "-" * (width - 2))
    rows = [
        ('Global FPS', 'global_fps'), ('Global ms/frame', 'global_ms'),
        ('FPS @10%', 'fps_lo'), ('FPS @50%', 'fps_mid'), ('FPS @90%', 'fps_hi'),
        ('ms @10%', 'ms_lo'),   ('ms @50%', 'ms_mid'),   ('ms @90%', 'ms_hi'),
    ]
    for label, field in rows:
        vals = "".join(f"{results[k].get(field, 0):>{col}.2f}" for k in keys)
        print(f"  {label:<18s}{vals}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GCTStream end-to-end FPS profiling (no module breakdown)."
    )
    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--img_h', type=int, default=378, help='Must be divisible by 14')
    parser.add_argument('--img_w', type=int, default=504, help='Must be divisible by 14')
    parser.add_argument('--num_frames', type=int, default=500)
    parser.add_argument('--sliding_window', type=int, default=64)
    parser.add_argument('--camera_num_iterations', type=int, default=4,
                        help='Camera head iterative-refinement steps. Default 4; '
                             'set 1 for faster inference (skips 3 refinement passes '
                             'at a small accuracy cost).')
    parser.add_argument('--backend', choices=['sdpa', 'flashinfer', 'both'], default='flashinfer')
    parser.add_argument('--dtype', choices=['bf16', 'fp32', 'both'], default='bf16')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='torch.compile hot modules (reduce-overhead) and drop point_head. '
                             'Typically ~5 FPS faster at 518×378.')
    parser.add_argument('--fa3', action='store_true',
                        help='Use FlashInfer FA3 (SM90) kernel instead of FA2 (requires power-of-2 page_size)')

    args = parser.parse_args()

    dtype_map = {'bf16': torch.bfloat16, 'fp32': torch.float32}
    backends = ['sdpa', 'flashinfer'] if args.backend == 'both' else [args.backend]
    dtypes = ['bf16', 'fp32'] if args.dtype == 'both' else [args.dtype]
    device = 'cuda'

    print("=" * 72)
    print(f"GCTStream FPS profiling  |  {args.img_h}×{args.img_w}  |  "
          f"{args.num_frames} frames  |  sw={args.sliding_window}")
    print(f"  backends={backends}  dtypes={dtypes}")
    print("=" * 72)

    # Synthetic images — keep on CPU for long runs to avoid OOM.
    img_device = device if args.num_frames <= 500 else 'cpu'
    print(f"Generating {args.num_frames} synthetic images on {img_device.upper()}...")
    torch.manual_seed(42)
    images_master = torch.randn(
        1, args.num_frames, 3, args.img_h, args.img_w,
        device=img_device, dtype=torch.float32,
    )

    results = {}
    for backend in backends:
        for dtype_str in dtypes:
            dtype = dtype_map[dtype_str]
            key = f'{backend}_{dtype_str}'
            print(f"\n{'=' * 72}\n  Run: {key}\n{'=' * 72}")

            model = load_model(
                backend,
                img_size=args.img_size,
                sliding_window=args.sliding_window,
                max_frame_num=args.num_frames + 100,
                camera_num_iterations=args.camera_num_iterations,
                device=device,
            )

            # FlashInfer FA2 only supports fp16/bf16; fall back to gather+SDPA for fp32.
            if backend == 'flashinfer' and dtype == torch.float32:
                model.aggregator.kv_cache_force_fp32 = True
            if backend == 'flashinfer' and args.fa3:
                model.aggregator.kv_cache_fa3 = True

            autocast_ctx = (
                contextlib.nullcontext() if dtype == torch.float32
                else torch.amp.autocast('cuda', dtype=dtype)
            )
            # N streaming frames in warmup so CUDA graphs / cuDNN autotune /
            # FlashInfer lazy init / allocator growth all complete before the
            # measured profile begins. Profile's `clean_kv_cache → P1 → stream`
            # opening then hits already-captured graphs with stable addresses.
            WARMUP_STREAM = 10
            warm_scale = images_master[:1, :8].to(device=device, dtype=dtype)
            warm_stream = images_master[:1, 8:8 + WARMUP_STREAM].to(device=device, dtype=dtype)

            def _warm(m, passes=1):
                """Run `passes` full `clean → Phase-1 → N-stream` sequences."""
                for _ in range(passes):
                    m.clean_kv_cache()
                    torch.compiler.cudagraph_mark_step_begin()
                    with torch.no_grad(), autocast_ctx:
                        m.forward(warm_scale, num_frame_for_scale=8,
                                  num_frame_per_block=8, causal_inference=True)
                    for i in range(WARMUP_STREAM):
                        torch.compiler.cudagraph_mark_step_begin()
                        with torch.no_grad(), autocast_ctx:
                            m.forward(warm_stream[:, i:i + 1], num_frame_for_scale=8,
                                      num_frame_per_block=1, causal_inference=True)
                torch.cuda.synchronize()

            # Eager warmup populates RoPE / kernel caches BEFORE torch.compile
            # captures CUDA graphs (otherwise capture would bake in a cache-miss
            # tensor-allocation path).
            print(f"  Warmup eager (scale + {WARMUP_STREAM} streaming)...")
            _warm(model)

            if args.compile:
                print(f"  Compiling hot modules...")
                compile_model(model)
                # Three passes under compile: 1st captures CUDA graphs, 2nd/3rd
                # replay so the caching allocator and graph-address map converge
                # on the exact state the subsequent profile will see.
                print(f"  Warmup compiled (3× dress rehearsal)...")
                _warm(model, passes=3)
            else:
                # No compile → a single dress-rehearsal pass is enough to
                # settle cuDNN / allocator for the first Phase-2 frame.
                _warm(model)

            images = images_master.to(dtype=dtype)
            per_frame_ms, scale_frames, phase1_ms = profile_streaming(
                model, images, args.num_frames, dtype,
            )
            results[key] = summarize(per_frame_ms, scale_frames, phase1_ms, key)

            del model
            torch.cuda.empty_cache()

    print_comparison(results)

    out_path = (
        f'/tmp/profile_results_{args.img_h}x{args.img_w}_'
        f'{args.num_frames}f_{args.dtype}.json'
    )
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
