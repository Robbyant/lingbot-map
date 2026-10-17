"""Profile depth head component timing in isolation."""
import sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_infer.model import GCTStreamMLX
from mlx_infer.weights import load_checkpoint
from mlx_infer.demo import _load_images_from_dir
from mlx_infer.heads import _bilinear, activate_head

CKPT = "/Users/dan/.cache/modelscope/hub/models/Robbyant/lingbot-map/lingbot-map-long.pt"
SW, SF, MSF = 16, 8, 10
N_WU, N_B = 30, 20

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
scale_out = model(imgs[None,:SF], num_frame_for_scale=SF, num_frame_per_block=SF, causal_inference=True)
mx.eval(scale_out); del scale_out

for i in range(SF, SF+N_WU):
    out = model(imgs[None,i:i+1], num_frame_for_scale=SF, num_frame_per_block=1, causal_inference=True)
    mx.eval(out)
print(f"Warmup done ({N_WU} frames)")

dh = model.depth_head
B, S, H, W = 1, 1, 294, 518
patch_h, patch_w, psi = H//14, W//14, model.aggregator.patch_start_idx

buckets = {k: [] for k in ['norm+proj+posemb', 'resize', 'scratch_rn', 'refinenets', 'output_conv', 'total']}

for i in range(SF+N_WU, SF+N_WU+N_B):
    frame = imgs[None, i:i+1]
    agg_list, _ = model.aggregator(frame, selected_idx=[4,11,17,23],
                                   num_frame_for_scale=SF, num_frame_per_block=1)
    mx.eval(agg_list)

    def ts(a):
        mx.eval(a); return time.perf_counter()

    # ---- norm + project + pos_embed (4 levels) ----
    t0 = time.perf_counter()
    out_feats = []
    for level, li in enumerate([0,1,2,3]):
        x = agg_list[li][:,:,psi:].reshape(B*S, patch_h*patch_w, -1)
        x = dh.norm(x).reshape(B*S, patch_h, patch_w, -1)
        x = dh.projects[level](x)
        x = dh._apply_pos_embed(x, W, H)
        out_feats.append(x)
    t1 = ts(out_feats)

    # ---- resize (ConvTranspose2d × 2 + Conv2d × 1) ----
    r0 = dh.resize_conv0(out_feats[0])
    r1 = dh.resize_conv1(out_feats[1])
    r2 = out_feats[2]
    r3 = dh.resize_conv3(out_feats[3])
    t2 = ts([r0, r1, r3])

    # ---- scratch layer_rn (4 × 3×3 Conv2d) ----
    l1 = dh.layer1_rn(r0); l2 = dh.layer2_rn(r1)
    l3 = dh.layer3_rn(r2); l4 = dh.layer4_rn(r3)
    t3 = ts([l1, l2, l3, l4])

    # ---- refinenets (4 FeatureFusionBlocks) ----
    o = dh.refinenet4(l4, target_hw=(l3.shape[1], l3.shape[2]))
    o = dh.refinenet3(o, skip=l3, target_hw=(l2.shape[1], l2.shape[2]))
    o = dh.refinenet2(o, skip=l2, target_hw=(l1.shape[1], l1.shape[2]))
    o = dh.refinenet1(o, skip=l1)
    t4 = ts(o)

    # ---- output convolutions + final bilinear ----
    o = dh.output_conv1(o)
    o = _bilinear(o, (patch_h*14, patch_w*14))
    o = dh._apply_pos_embed(o, W, H)
    o = nn.relu(dh.output_conv2a(o))
    o = dh.output_conv2b(o)
    o, _ = activate_head(o, dh.activation, dh.conf_activation)
    t5 = ts(o)

    model.camera_head(agg_list, causal_inference=True, num_frame_per_block=1, num_frame_for_scale=SF)

    buckets['norm+proj+posemb'].append((t1-t0)*1e3)
    buckets['resize'].append((t2-t1)*1e3)
    buckets['scratch_rn'].append((t3-t2)*1e3)
    buckets['refinenets'].append((t4-t3)*1e3)
    buckets['output_conv'].append((t5-t4)*1e3)
    buckets['total'].append((t5-t0)*1e3)

print("\n--- Depth head breakdown ---")
for k, v in buckets.items():
    a = np.array(v)
    print(f"  {k:20s}  mean={a.mean():.1f}ms  min={a.min():.1f}ms  std={a.std():.1f}ms")
