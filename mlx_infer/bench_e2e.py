"""End-to-end per-frame timing through model() — measures the fused aggregator+camera+depth graph."""
import sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_infer.model import GCTStreamMLX
from mlx_infer.weights import load_checkpoint
from mlx_infer.demo import _load_images_from_dir

CKPT = "/Users/dan/.cache/modelscope/hub/models/Robbyant/lingbot-map/lingbot-map-long.pt"
SW, SF, MSF = 16, 8, 10
N_WU, N_B = 30, 40

model = GCTStreamMLX(
    img_size=518, patch_size=14, embed_dim=1024, depth=24, num_heads=16,
    num_register_tokens=4, kv_cache_sliding_window=SW, kv_cache_scale_frames=SF,
    kv_cache_max_special_frames=MSF, camera_num_iterations=4,
    enable_depth=True, enable_point=False,
)
load_checkpoint(model, CKPT, verbose=False)
model.apply(lambda x: x.astype(mx.float16) if isinstance(x, mx.array) else x)
print("Model loaded")

imgs_np = _load_images_from_dir("example/courthouse", img_size=518)[:SF + N_WU + N_B]
imgs = mx.array(imgs_np).astype(mx.float16)

model.clean_kv_cache()
s = model(imgs[None,:SF], num_frame_for_scale=SF, num_frame_per_block=SF, causal_inference=True)
mx.eval(s); del s

for i in range(SF, SF+N_WU):
    o = model(imgs[None,i:i+1], num_frame_for_scale=SF, num_frame_per_block=1, causal_inference=True)
    mx.eval(o)
print(f"Warmup ({N_WU} frames) done")

times = []
for i in range(SF+N_WU, SF+N_WU+N_B):
    t0 = time.perf_counter()
    o = model(imgs[None,i:i+1], num_frame_for_scale=SF, num_frame_per_block=1, causal_inference=True)
    mx.eval(o)
    times.append((time.perf_counter() - t0) * 1e3)

a = np.array(times)
print(f"\n--- End-to-end (N={N_B}, sw={SW}) ---")
print(f"  mean={a.mean():.1f}ms  min={a.min():.1f}ms  std={a.std():.1f}ms  => {1000/a.mean():.2f} fps")
