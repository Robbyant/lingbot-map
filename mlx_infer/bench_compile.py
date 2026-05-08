"""
Benchmark: measure per-component timing before/after backbone mx.compile.
Run with:
    conda run -n lingbot python mlx_infer/bench_compile.py \
        --checkpoint /Users/dan/.cache/modelscope/hub/models/Robbyant/lingbot-map/lingbot-map-long.pt \
        --images example/courthouse
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_infer.model import GCTStreamMLX
from mlx_infer.weights import load_checkpoint
from mlx_infer.demo import _load_images_from_dir

CKPT  = "/Users/dan/.cache/modelscope/hub/models/Robbyant/lingbot-map/lingbot-map-long.pt"
IMGS  = "example/courthouse"
SW    = 16
SF    = 8
MSF   = 10
DTYPE = mx.float16
N_WARMUP = 30   # need 16 (fill cache) + max_special_frames=10 (fill k_special cap) + buffer
N_BENCH  = 40

def tsync(arr):
    mx.eval(arr)
    return time.perf_counter()

def build_model():
    m = GCTStreamMLX(
        img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        num_register_tokens=4,
        kv_cache_sliding_window=SW,
        kv_cache_scale_frames=SF,
        kv_cache_max_special_frames=MSF,
        camera_num_iterations=4,
        enable_depth=True,
        enable_point=False,
    )
    load_checkpoint(m, CKPT, verbose=False)
    m.apply(lambda x: x.astype(mx.float16) if isinstance(x, mx.array) else x)
    return m

def run_bench(args):
    model = build_model()

    images_np = _load_images_from_dir(IMGS, img_size=518)[:SF + N_WARMUP + N_BENCH]
    images = mx.array(images_np).astype(DTYPE)
    print(f"Image shape: {images.shape}  dtype={images.dtype}")
    print(f"KV cache: scale={SF} + sliding={SW}  max_special_frames={MSF}")

    model.clean_kv_cache()

    # Phase 1
    scale_out = model(
        images[None, :SF],
        num_frame_for_scale=SF,
        num_frame_per_block=SF,
        causal_inference=True,
    )
    mx.eval(scale_out)
    print("Phase 1 done")

    # Warmup (fill cache to steady state)
    for i in range(SF, SF + N_WARMUP):
        out = model(images[None, i:i+1], num_frame_for_scale=SF,
                    num_frame_per_block=1, causal_inference=True)
        mx.eval(out)
    print(f"Warmup ({N_WARMUP} frames) done, measuring {N_BENCH} frames...")

    # Benchmark loop
    agg_ms = []; cam_ms = []; depth_ms = []; total_ms = []

    for i in range(SF + N_WARMUP, SF + N_WARMUP + N_BENCH):
        frame = images[None, i:i+1]

        t0 = time.perf_counter()
        agg_list, psi = model.aggregator(
            frame, selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=SF, num_frame_per_block=1,
        )
        t1 = tsync(agg_list)

        pose_list = model.camera_head(
            agg_list, causal_inference=True,
            num_frame_per_block=1, num_frame_for_scale=SF,
        )
        t2 = tsync(pose_list[-1])

        depth, dconf = model.depth_head(agg_list, frame, psi)
        t3 = tsync(depth)

        agg_ms.append((t1-t0)*1e3)
        cam_ms.append((t2-t1)*1e3)
        depth_ms.append((t3-t2)*1e3)
        total_ms.append((t3-t0)*1e3)

    def stats(name, arr):
        a = np.array(arr)
        print(f"  {name:8s}  mean={a.mean():.1f}ms  min={a.min():.1f}ms  std={a.std():.1f}ms")

    print(f"\n--- Steady-state component breakdown ({N_BENCH} frames) ---")
    stats("agg",   agg_ms)
    stats("cam",   cam_ms)
    stats("depth", depth_ms)
    stats("total", total_ms)
    fps = 1000 / np.mean(total_ms)
    print(f"  => {fps:.2f} fps")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--images", default=IMGS)
    args = parser.parse_args()
    CKPT = args.checkpoint
    IMGS = args.images
    run_bench(args)
