# mlx_infer

MLX inference backend for GCTStream on Apple Silicon. Runs the full streaming 3D reconstruction pipeline (pose + depth) natively on Metal via the unified memory architecture, eliminating the Metal OOM accumulation that occurs with the PyTorch backend on long sequences.

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.10+
- A PyTorch `.pt` checkpoint (same file used by the main `demo.py`)

## Installation

```bash
pip install lingbot-map[mlx]
```

This installs `mlx==0.31.2` and `mlx-metal==0.31.2`. The rest of the dependencies (`torch`, `numpy`, etc.) are part of the base install.

## Demo

```bash
python mlx_infer/demo.py \
    --checkpoint /path/to/lingbot-map-long.pt \
    --images     /path/to/image_dir \
    --output     out.npz
```

Images are loaded from the directory in sorted order (`.jpg` and `.png`). The script resizes each frame to `--img-size` width (default 518), rounds height to the nearest 14-pixel multiple, and center-crops if the result is taller than `--img-size`.

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--scale-frames` | 8 | Phase 1 bidirectional frames. Lower = faster, lower initialization quality. |
| `--kv-sliding-window` | 64 | Sliding window size for KV cache eviction. Lower = faster, shorter temporal context. |
| `--dtype` | `float32` | `float16` is ~2× faster and recommended for speed benchmarks. |
| `--max-frames` | all | Truncate the sequence to N frames. |
| `--keyframe-interval` | 1 | Only append every Nth frame to KV cache (1 = every frame). |
| `--max-special-frames` | none | Cap the number of evicted-frame tokens retained. Set to ~100 for sequences >300 frames with a small sliding window. |
| `--no-vis` | off | Skip the viser 3D viewer; just save the `.npz`. |

### Speed vs quality tradeoffs (float16, 294×518 px)

| `--scale-frames` | `--kv-sliding-window` | fps |
|---|---|---|
| 8 | 64 | ~1.5 |
| 8 | 16 | ~1.9 |
| 8 | 8 | ~2.1 |
| 8 | 4 | ~2.4 |
| 1 | 4 | ~2.8 |
| 1 | 1 | ~3.0 |

## Streaming inference API

```python
from mlx_infer import GCTStreamMLX, load_checkpoint
import mlx.core as mx
import numpy as np

model = GCTStreamMLX(
    kv_cache_sliding_window=16,
    kv_cache_scale_frames=8,
    enable_depth=True,
    enable_point=False,
)
load_checkpoint(model, "lingbot-map-long.pt")

images = mx.array(np.load("frames.npy"))   # [S, 3, H, W] in [0, 1]
predictions = model.inference_streaming(images, num_scale_frames=8)
mx.eval(predictions)
# keys: pose_enc, depth, depth_conf, images
```

`inference_streaming` runs two phases:

1. **Scale phase** — the first `num_scale_frames` frames are processed together bidirectionally via a scale token, establishing the scene geometry baseline.
2. **Streaming phase** — remaining frames are processed one at a time with a sliding KV cache, enabling unbounded sequences at bounded memory.

## Benchmarking

```bash
# End-to-end per-frame timing
python mlx_infer/bench_e2e.py \
    --checkpoint /path/to/checkpoint.pt \
    --images     /path/to/images \
    --sw 16 --sf 8 --msf 10

# Depth head component breakdown
python mlx_infer/bench_depth.py

# MLX graph compile timing
python mlx_infer/bench_compile.py
```

## Checkpoint loading

`load_checkpoint` converts a PyTorch `.pt` checkpoint to MLX in-memory — no separate conversion step is needed. Weight remapping handles the structural differences between the PyTorch and MLX model definitions (Conv2d/ConvTranspose2d axis permutations, renamed submodules).

## Output format

`inference_streaming` returns an `mx.array` dict:

| Key | Shape | Description |
|---|---|---|
| `pose_enc` | `[B, S, 9]` | Encoded camera pose (FoV + rotation + translation) |
| `depth` | `[B, S, H, W, 1]` | Metric depth |
| `depth_conf` | `[B, S, H, W]` | Depth confidence |
| `world_points` | `[B, S, H, W, 3]` | 3D world points (requires `enable_point=True`) |
| `world_points_conf` | `[B, S, H, W]` | World point confidence |
| `images` | `[B, S, 3, H, W]` | Input images (pass-through) |

Use `demo.postprocess()` to convert `pose_enc` to `(extrinsic, intrinsic)` numpy arrays compatible with `PointCloudViewer`.
