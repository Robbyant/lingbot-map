<div align="center">
  <img src="assets/teaser.png" width="100%">

<h1>LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction</h1>

Robbyant Team

</div>

<div align="center">

[![Paper](https://img.shields.io/static/v1?label=Paper&message=arXiv&color=red&logo=arxiv)](https://arxiv.org/abs/2604.14141)
[![PDF](https://img.shields.io/static/v1?label=Paper&message=PDF&color=red&logo=adobeacrobatreader)](lingbot-map_paper.pdf)
[![Project](https://img.shields.io/badge/Project-Website-blue)](https://technology.robbyant.com/lingbot-map)
[![HuggingFace](https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&message=HuggingFace&color=orange)](https://huggingface.co/robbyant/lingbot-map)
[![ModelScope](https://img.shields.io/static/v1?label=%F0%9F%A4%96%20Model&message=ModelScope&color=purple)](https://www.modelscope.cn/models/Robbyant/lingbot-map)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE.txt)

</div>

https://github.com/user-attachments/assets/fe39e095-af2c-4ec9-b68d-a8ba97e505ab

-----

### 🗺️ Meet LingBot-Map! We've built a feed-forward 3D foundation model for streaming 3D reconstruction! 🏗️🌍

LingBot-Map has focused on:

- **Geometric Context Transformer**: Architecturally unifies coordinate grounding, dense geometric cues, and long-range drift correction within a single streaming framework through anchor context, pose-reference window, and trajectory memory.
- **High-Efficiency Streaming Inference**: A feed-forward architecture with paged KV cache attention, enabling stable inference at ~20 FPS on 518×378 resolution over long sequences exceeding 10,000 frames.
- **State-of-the-Art Reconstruction**: Superior performance on diverse benchmarks compared to both existing streaming and iterative optimization-based approaches.

---

# 📰 News

- **2026-04-27** — 🚀 **LingBot-Map accelerated**. Pull the latest `main` and run `python demo.py --compile ...` or `python gct_profile.py --backend flashinfer --dtype bf16 --compile` to verify on your hardware.
- **2026-04-24** — Fixed a FlashInfer KV cache bug where `--keyframe_interval > 1` silently cached non-keyframes. **You should now see better pose and reconstruction quality when running with more than 320 frames**.
---

# 📋 TODO

- [ ] Release evaluation benchmark
- [ ] Release demo scripts

---

# ⚙️ Quick Start

## Installation

**1. Create conda environment**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. Install PyTorch (CUDA 12.8)**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

> PyTorch 2.8.0 is the recommended version because NVIDIA Kaolin (required by the batch rendering pipeline) has prebuilt wheels for `torch-2.8.0_cu128`. If you only need `demo.py` you may use a newer PyTorch, but the batch renderer then requires building Kaolin from source.
> For other CUDA versions, see [PyTorch Get Started](https://pytorch.org/get-started/locally/).

**3. Install lingbot-map**

```bash
pip install -e .
```

**4. Install FlashInfer (recommended)**

FlashInfer provides paged KV cache attention for efficient streaming inference. It is a pure-Python package that JIT-compiles CUDA kernels on first use, so a single wheel works across CUDA/PyTorch versions:

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

> `--index-url https://pypi.org/simple` is only needed if your default pip index is an internal mirror that doesn't have `flashinfer-python`.
> (Optional) For faster first-use, you can additionally install a CUDA-specific JIT cache: `pip install flashinfer-jit-cache -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/`.
> See [FlashInfer installation](https://docs.flashinfer.ai/installation.html) for details. If FlashInfer is not installed, the model falls back to SDPA (PyTorch native attention) via `--use_sdpa`.

**5. Visualization dependencies (optional)**

```bash
pip install -e ".[vis]"
```

# 📦 Model Download

| Model Name | Huggingface Repository | ModelScope Repository | Description |
| :--- | :--- | :--- | :--- |
| lingbot-map-long | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Better suited for long sequences and large scale scenes (Recommend). |
| lingbot-map | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Balanced checkpoint — trade off all-around performance across short and long sequences. |
| lingbot-map-stage1 | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | Stage-1 training checkpoint of lingbot-map — can be loaded into the VGGT model for bidirectional inference (c2w). |

> 🚧 **Coming soon:** we're training an stronger model that supports longer sequences — stay tuned.

# 🎬 Demo

Run `demo.py` for interactive 3D visualization via a browser-based [viser](https://github.com/nerfstudio-project/viser) viewer (default `http://localhost:8080`).

### Try the Example Scenes

We provide four example scenes in `example/` that you can run out of the box:
```bash
# courthouse scene
python demo.py --model_path /path/to/lingbot-map-long.pt \
    --image_folder example/courthouse --mask_sky
```


https://github.com/user-attachments/assets/aa10f7ab-8024-43c7-92f8-d56159ec85c8






```bash
# University scene
python demo.py --model_path /path/to/lingbot-map-long.pt \
    --image_folder example/university --mask_sky
```


https://github.com/user-attachments/assets/212a1744-6ff5-4ccf-9bd4-728608248b57







```bash
# Loop scene (loop closure trajectory)
python demo.py --model_path /path/to/lingbot-map-long.pt \
    --image_folder example/loop
```


https://github.com/user-attachments/assets/5ae0a292-b081-40c6-838c-b7c1a0538d75





```bash
# Oxford scene with sky masking (outdoor, large scale scene)
python demo.py --model_path /path/to/lingbot-map-long.pt \
    --image_folder example/oxford --mask_sky
```


https://github.com/user-attachments/assets/6b8daa95-9ed4-40b2-9902-7435779b886d






We will provide more examples in the follow-up.
### Streaming with Keyframe Interval

Use `--keyframe_interval` to reduce KV cache memory by only keeping every N-th frame as a keyframe. Non-keyframe frames still produce predictions but are not stored in the cache. This is useful for long sequences which exceed 320 frames (We train with video RoPE on 320 views, so performance degrades when the KV cache stores more than 320 views. Using a keyframe strategy allows inference over longer sequences.).

**Dataset:** Download the demo sequences from [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) on Hugging Face.

Example run on the `travel` sequence from the dataset above (sky masking on, 4 camera optimization iterations, keyframe every 2 frames):

```bash
python demo.py \
    --image_folder /path/to/lingbot-map-demo/travel/ \
    --model_path /path/to/lingbot-map-long.pt \
    --mask_sky \
    --camera_num_iterations 4 \
    --keyframe_interval 2
```


https://github.com/user-attachments/assets/d350b590-d036-4363-af8c-7af3918338ef






### Windowed Inference (for long sequences, >3000 frames)

```bash
python demo.py --model_path /path/to/lingbot-map-long.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128
```


### Sky Masking

Sky masking uses an ONNX sky segmentation model to filter out sky points from the reconstructed point cloud, which improves visualization quality for outdoor scenes.

**Setup:**

```bash
# Install onnxruntime (required)
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # GPU (faster for large image sets)
```

The sky segmentation model (`skyseg.onnx`) will be automatically downloaded from [HuggingFace](https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx) on first use.

**Usage:**

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky
```

Sky masks are cached in `<image_folder>_sky_masks/` so subsequent runs skip regeneration. You can also specify a custom cache directory with `--sky_mask_dir`, or save side-by-side mask visualizations with `--sky_mask_visualization_dir`:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --sky_mask_dir /path/to/cached_masks/ \
    --sky_mask_visualization_dir /path/to/mask_viz/
```

### Visualization Options

| Argument | Default | Description |
|:---|:---|:---|
| `--port` | `8080` | Viser viewer port |
| `--conf_threshold` | `1.5` | Visibility threshold for filtering low-confidence points |
| `--point_size` | `0.00001` | Point cloud point size |
| `--downsample_factor` | `10` | Spatial downsampling for point cloud display |

### Without FlashInfer (SDPA fallback)

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --use_sdpa
```

### Running on Limited GPU Memory

If you run into out-of-memory issues, try one (or both) of the following:

- **`--offload_to_cpu`** — offload per-frame predictions to CPU during inference (on by default; use `--no-offload_to_cpu` only if you have memory to spare).
- **`--num_scale_frames 2`** — reduce the number of bidirectional scale frames from the default 8 down to 2, which shrinks the activation peak of the initial scale phase.

### Faster Inference

Lower the number of iterative refinement steps in the camera head to trade a small amount of pose accuracy for wall-clock speed:

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --camera_num_iterations 1
```

`--camera_num_iterations` defaults to `4`; setting it to `1` skips three refinement passes in the camera head (and shrinks its KV cache by 4×).

<!-- # 🎥 Batch Inference & Offline Video Rendering

`demo_render/` produces headless point-cloud flythrough videos from a video or image folder — a two-phase pipeline that shares the same PyTorch / FlashInfer / checkpoint stack as `demo.py`:

```
Video / Images ──► batch_demo.py (inference) ──► NPZ ──► rgbd_scan_render.py (rendering) ──► MP4
                   Phase 1: model prediction         Phase 2: point cloud rendering
```

- **Phase 1** — `batch_demo.py`: run model inference, save per-frame NPZs (depth, poses, images).
- **Phase 2** — `rgbd_scan_render.py`: build a voxelized point cloud from NPZ, render a camera flythrough with trajectory overlays.
- **Combined** — `process_videos.sh`: batch-process a folder of videos through both phases, skipping those that already have NPZ output.

## Install (extends the main install)

**1. Rendering Python dependencies**

```bash
pip install -e ".[vis,render]"
```

`render` pulls in `open3d>=0.19` and `pyyaml` (the core `numpy<2` constraint comes from the base `lingbot-map` install). Sky masking in this pipeline uses `onnxruntime-gpu` for batched segmentation; install it if you don't already have the CPU `onnxruntime`:

```bash
pip install onnxruntime-gpu
```

**2. Kaolin** — matches the PyTorch 2.8.0 + CUDA 12.8 recommended above:

```bash
pip install --index-url https://pypi.org/simple \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

> `--index-url https://pypi.org/simple` bypasses any internal mirror that might otherwise serve the PyPI placeholder wheel (which raises `ImportError` on import).
> NVIDIA Kaolin does not publish prebuilt wheels for PyTorch 2.9.x — if you're on 2.9 for other reasons, build Kaolin from source (`pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git`, needs local CUDA toolkit). For other torch/CUDA combinations see [NVIDIA Kaolin installation](https://kaolin.readthedocs.io/en/latest/notes/installation.html).

**3. ffmpeg**

```bash
sudo apt install ffmpeg    # or: brew install ffmpeg
```

**4. CUDA extensions** (required before first run)

```bash
cd demo_render/render_cuda_ext && python setup.py build_ext --inplace && cd ../..
```

This builds `voxel_morton_ext` and `frustum_cull_ext` in place — both are imported by `rgbd_render` for GPU voxelization and frustum culling.

## Quick Start

### Single video (both phases)

```bash
# Phase 1: inference → per-frame NPZ
CUDA_VISIBLE_DEVICES=0 python demo_render/batch_demo.py \
    --video_path /path/to/video.mp4 \
    --output_folder /path/to/output/ \
    --model_path /path/to/lingbot-map-long.pt \
    --mode windowed --window_size 64 --fps 20 \
    --save_predictions --no_render

# Phase 2: NPZ → rendered video
CUDA_VISIBLE_DEVICES=0 python demo_render/rgbd_scan_render.py \
    --input_npz /path/to/output/video_name/ \
    --output_video /path/to/output/video_name.mp4 \
    --mask_sky --draw_traj --fps 60
```

### Batch a folder of videos

Edit the config variables at the top of `demo_render/process_videos.sh`, then:

```bash
bash demo_render/process_videos.sh
```

Runs Phase 1 on all videos, then Phase 2. Skips videos that already have NPZ output (safe to re-run).

## Phase 1 — `batch_demo.py`

| Mode | Flag | Description |
|------|------|-------------|
| **Streaming** | `--mode streaming` | Frame-by-frame with KV cache. Fast, lower memory. |
| **Windowed** | `--mode windowed` | Sliding window with overlap alignment. Better quality for long sequences. |

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | required | Path to model checkpoint |
| `--video_path` / `--input_folder` | — | Input video, or folder of scene directories |
| `--fps` / `--target_frames` | — | Extraction rate, or target total frames (auto-computes fps) |
| `--first_k` / `--image_stride` | — / 1 | Only use first K frames / every N-th frame |
| `--window_size` | 64 | Frames per window (windowed mode) |
| `--flow_threshold` | 0.0 | Flow-based keyframe threshold in px; `>0` enables adaptive keyframes |
| `--max_non_keyframe_gap` | 100 | Max consecutive non-keyframes before forcing one |
| `--mask_sky` | off | Run sky segmentation during inference |
| `--compile` | off | `torch.compile` CUDA graph acceleration (FlashInfer backend only) |
| `--save_predictions` | off | Save per-frame NPZ files |
| `--no_render` | off | Skip video rendering (inference only) |

Example — long video with flow-based keyframes:

```bash
CUDA_VISIBLE_DEVICES=0 python demo_render/batch_demo.py \
    --video_path video.mp4 --output_folder outputs/ \
    --model_path lingbot-map-long.pt \
    --mode windowed --window_size 320 \
    --flow_threshold 25.0 --max_non_keyframe_gap 100 \
    --target_frames 4000 --vis_threshold 2.0 \
    --mask_sky --save_predictions --no_render
```

## Phase 2 — `rgbd_scan_render.py`

Configuration is loaded in order **YAML → CLI flags** (CLI wins). Presets live in `demo_render/config/`:

| Preset | Scene Type | Notes |
|--------|------------|-------|
| `config/default.yaml` | General | Renders first 200 frames for quick preview |
| `config/indoor.yaml` | Indoor | Short depth (10m), tighter camera follow |
| `config/outdoor_large.yaml` | Large outdoor | Sky masking on, coarser voxels, full render |

```bash
# YAML preset
python demo_render/rgbd_scan_render.py \
    --config demo_render/config/indoor.yaml \
    --input_npz scene_dir/ --output_video output.mp4

# Pure CLI
python demo_render/rgbd_scan_render.py \
    --input_npz scene_dir/ --output_video output.mp4 \
    --voxel_size 0.001 --mask_sky --draw_traj \
    --back_offset 0.6 --up_offset 0.3 --look_offset 0.3 \
    --num_workers 16 --fps 60
```

Each run produces (assuming `--output_video output.mp4`):

| File | Description |
|------|-------------|
| `output.mp4` | Rendered point-cloud flythrough |
| `output_rgb.mp4` | Original RGB frames encoded as video |
| `output_depth.mp4` | Depth visualization (with `--depth_video`) |
| `output_config.yaml` | Full config snapshot of this run |

## NPZ Input Format

`--input_npz` accepts either a single `.npz` with all frames stacked (`images (S,H,W,3)`, `depth (S,H,W)`, `intrinsic (S,3,3)`, `extrinsic (S,4,4)` world-to-camera, optional `depth_conf`), or a per-frame directory produced by `batch_demo.py --save_predictions`:

```
scene_name/
  frame_000000.npz    # per-frame slice of each key
  frame_000001.npz
  ...
  meta.npz            # optional non-sequence metadata
```

Per-frame files are loaded in parallel and stacked automatically — recommended for 500+ frame sequences.

See [`demo_render/README.md`](demo_render/README.md) for the full parameter reference (scene / preprocess / camera segments / render / overlay / pipeline / gpu), multi-segment camera path examples, and library-style usage. -->

# 📜 License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE.txt) file for details.

# 📖 Citation

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

# ✨ Acknowledgments

We thank Shangzhan Zhang, Jianyuan Wang, Yudong Jin, Christian Rupprecht, and Xun Cao for their helpful discussions and support.

This work builds upon several excellent open-source projects:

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer)

---
