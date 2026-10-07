"""End-to-end per-frame timing through model() — measures the fused aggregator+camera+depth graph."""
import argparse, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_infer.model import GCTStreamMLX
from mlx_infer.weights import load_checkpoint
from mlx_infer.demo import _load_images_from_dir

CKPT = "/Users/dan/.cache/modelscope/hub/models/Robbyant/lingbot-map/lingbot-map-long.pt"

def run(sw, sf, msf, n_wu, n_b, ckpt, imgs_dir):
    model = GCTStreamMLX(
        img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        num_register_tokens=4, kv_cache_sliding_window=sw, kv_cache_scale_frames=sf,
        kv_cache_max_special_frames=msf, camera_num_iterations=4,
        enable_depth=True, enable_point=False,
    )
    load_checkpoint(model, ckpt, verbose=False)
    model.apply(lambda x: x.astype(mx.float16) if isinstance(x, mx.array) else x)
    print(f"Model loaded  sw={sw} sf={sf} msf={msf}")

    imgs_np = _load_images_from_dir(imgs_dir, img_size=518)[:sf + n_wu + n_b]
    imgs = mx.array(imgs_np).astype(mx.float16)

    model.clean_kv_cache()
    s = model(imgs[None,:sf], num_frame_for_scale=sf, num_frame_per_block=sf, causal_inference=True)
    mx.eval(s); del s

    for i in range(sf, sf + n_wu):
        o = model(imgs[None,i:i+1], num_frame_for_scale=sf, num_frame_per_block=1, causal_inference=True)
        mx.eval(o)
    print(f"Warmup ({n_wu} frames) done")

    times = []
    for i in range(sf + n_wu, sf + n_wu + n_b):
        t0 = time.perf_counter()
        o = model(imgs[None,i:i+1], num_frame_for_scale=sf, num_frame_per_block=1, causal_inference=True)
        mx.eval(o)
        times.append((time.perf_counter() - t0) * 1e3)

    a = np.array(times)
    print(f"\n--- End-to-end (N={n_b}, sw={sw}, sf={sf}) ---")
    print(f"  mean={a.mean():.1f}ms  min={a.min():.1f}ms  std={a.std():.1f}ms  => {1000/a.mean():.2f} fps")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sw",   type=int, default=16)
    p.add_argument("--sf",   type=int, default=8)
    p.add_argument("--msf",  type=int, default=10)
    p.add_argument("--n-wu", type=int, default=30)
    p.add_argument("--n-b",  type=int, default=40)
    p.add_argument("--checkpoint", default=CKPT)
    p.add_argument("--images", default="example/courthouse")
    args = p.parse_args()
    run(args.sw, args.sf, args.msf, args.n_wu, args.n_b, args.checkpoint, args.images)
