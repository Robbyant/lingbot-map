<div align="center">
  <img src="assets/teaser.webp" width="100%">

<h1>LingBot-Map：用于流式三维重建的几何上下文 Transformer</h1>

Robbyant Team

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md)

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

### 🗺️ 认识 LingBot-Map！我们构建了一个用于流式三维重建的前馈式三维基础模型！🏗️🌍

LingBot-Map 重点解决以下问题：

- **几何上下文 Transformer**：通过锚点上下文、位姿参考窗口和轨迹记忆，在单一流式框架中从架构层面统一坐标锚定、稠密几何线索与长距离漂移校正。
- **高效流式推理**：采用带分页 KV 缓存注意力的前馈架构，在 518×378 分辨率、超过 10,000 帧的长序列上实现约 20 FPS 的稳定推理。
- **先进的重建效果**：与现有流式方法和基于迭代优化的方法相比，在多种基准测试上均表现更优。

---

## 📑 目录

<details>
<summary>点击展开</summary>

- [📰 新闻](#-新闻)
- [📋 待办事项](#-待办事项)
- [⚙️ 安装](#️-安装)
- [📦 模型下载](#-模型下载)
- [🚀 快速开始](#-快速开始)
- [🎬 交互式演示（`demo.py`）](#-交互式演示demopy)
  - [尝试示例场景](#尝试示例场景)
  - [按关键帧间隔进行流式推理](#按关键帧间隔进行流式推理)
  - [窗口化推理（用于超过 3000 帧的长序列）](#窗口化推理用于超过-3000-帧的长序列)
  - [天空遮罩](#天空遮罩)
  - [可视化选项](#可视化选项)
  - [性能与显存](#性能与显存)
- [🎥 离线渲染流水线（`demo_render/batch_demo.py`）](#-离线渲染流水线demo_renderbatch_demopy)
- [📜 许可证](#-许可证)
- [📖 引用](#-引用)
- [✨ 致谢](#-致谢)

</details>

---

## 📰 新闻

- **2026-06-28** — 修复了一个 SDPA KV 缓存错误。**SDPA 后端现在处理长序列时表现更好**。为获得最佳性能，我们仍建议使用 FlashInfer 后端。
- **2026-05-25** — 📊 **发布评测基准**。我们发布了 KITTI 和 Oxford Spires 的评测脚本——流水线请参阅 [benchmark/](benchmark/)，并在评测前运行 [`preprocess/oxford.py`](preprocess/oxford.py) 准备 Oxford Spires 数据。
- **2026-04-29** — 📹 **发布长视频演示**。我们发布了一个由离线流水线渲染的超长视频示例（约 25,000 帧、13 分钟室内漫游）——命令、参数说明和渲染结果请参阅[完整示例](#完整示例长室内漫游约-25000-帧13-分钟)。
- **2026-04-27** — 🚀 **LingBot-Map 加速完成**。拉取最新的 `main`，然后运行 `python demo.py --compile ...` 或 `python gct_profile.py --backend flashinfer --dtype bf16 --compile`，即可在你的硬件上验证。
- **2026-04-24** — 修复了 `--keyframe_interval > 1` 时 FlashInfer KV 缓存会静默缓存非关键帧的问题。**现在处理超过 320 帧时，位姿与重建质量应当会更好**。

---

## 📋 待办事项

- ✅ 发布评测基准
  - ✅ Oxford Spires 数据集
  - ✅ KITTI 数据集
  - ✅ VBR 数据集
  - ✅ Droid-W 数据集
  - ✅ TUM-D 数据集
  - ✅ 7-scenes 数据集
  - ✅ ETH3D 数据集
  - ✅ Tanks and Temples 数据集
  - ✅ NRGBD 数据集
- ✅ 发布演示脚本
  - ✅ 室内长视频演示（[精选室内漫游](#-精选室内漫游约-25000-帧13-分钟)）
  - ✅ 室外长视频演示
  - ✅ LingBot-World 演示（[完整示例](#完整示例lingbot-world-场景)）
  - ✅ 航拍长视频演示

---

## ⚙️ 安装

**1. 创建 conda 环境**

```bash
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
```

**2. 安装 PyTorch（CUDA 12.8）**

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

> 推荐使用 PyTorch 2.8.0，因为 NVIDIA Kaolin（批量渲染流水线所必需）为 `torch-2.8.0_cu128` 提供了预编译 wheel。如果只需要 `demo.py`，可以使用更新的 PyTorch，但批量渲染器届时需要从源码构建 Kaolin。
> 其他 CUDA 版本请参阅 [PyTorch 入门指南](https://pytorch.org/get-started/locally/)。

**3. 安装 lingbot-map**

```bash
pip install -e .
```

**4. 安装 FlashInfer（推荐）**

FlashInfer 提供分页 KV 缓存注意力，以实现高效的流式推理。它是一个纯 Python 包，在首次使用时对 CUDA 内核进行 JIT 编译，因此一个 wheel 即可适配多个 CUDA/PyTorch 版本：

```bash
pip install --index-url https://pypi.org/simple flashinfer-python
```

> `--index-url https://pypi.org/simple` 仅在默认 pip 索引是缺少 `flashinfer-python` 的内部镜像时才需要使用。
> （可选）为加快首次使用速度，还可以安装 CUDA 专用的 JIT 缓存：`pip install flashinfer-jit-cache -f https://flashinfer.ai/whl/cu128/flashinfer-jit-cache/`。
> 详情请参阅 [FlashInfer 安装文档](https://docs.flashinfer.ai/installation.html)。如果未安装 FlashInfer，可通过 `--use_sdpa` 让模型回退到 SDPA（PyTorch 原生注意力）。

**5. 可视化依赖（可选）**

```bash
pip install -e ".[vis]"
```

## 📦 模型下载

| 模型名称 | Hugging Face 仓库 | ModelScope 仓库 | 说明 |
| :--- | :--- | :--- | :--- |
| lingbot-map-long | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | 更适合长序列和大尺度场景。 |
| lingbot-map | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | 均衡检查点（用于论文、基准测试和离线演示），兼顾短序列与长序列的综合性能。 |
| lingbot-map-stage1 | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | [Robbyant/lingbot-map](https://www.modelscope.cn/models/Robbyant/lingbot-map) | lingbot-map 第一阶段训练检查点，可加载到 VGGT 模型中进行双向推理（c2w）。 |

> 🚧 **即将推出：**我们正在训练一个支持更长序列的更强模型，敬请期待。

## 🚀 快速开始

安装完成后，用一条命令运行你的第一个场景：

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```

这会在 `http://localhost:8080` 启动交互式 [viser](https://github.com/nerfstudio-project/viser) 查看器。完整的场景和参数请参阅下方的[交互式演示](#-交互式演示demopy)，长序列批量渲染请直接跳转到[离线渲染流水线](#-离线渲染流水线demo_renderbatch_demopy)。

## 🎬 交互式演示（`demo.py`）

运行 `demo.py`，通过基于浏览器的 [viser](https://github.com/nerfstudio-project/viser) 查看器（默认地址为 `http://localhost:8080`）进行交互式三维可视化。

### 尝试示例场景

我们在 `example/` 中提供了三个开箱即用的示例场景：
```bash
# courthouse scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/courthouse --mask_sky
```


https://github.com/user-attachments/assets/aa10f7ab-8024-43c7-92f8-d56159ec85c8






```bash
# University scene
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/university --mask_sky
```


https://github.com/user-attachments/assets/212a1744-6ff5-4ccf-9bd4-728608248b57







```bash
# Loop scene (loop closure trajectory)
python demo.py --model_path /path/to/lingbot-map.pt \
    --image_folder example/loop
```


https://github.com/user-attachments/assets/5ae0a292-b081-40c6-838c-b7c1a0538d75





#### 🎯 精选：室内漫游（约 25,000 帧，13 分钟）


*该序列对交互式 viser 查看器而言过长——此视频使用[离线渲染流水线](#-离线渲染流水线demo_renderbatch_demopy)渲染。完整命令请参阅对应章节。*

后续我们将提供更多示例。

### 动态场景演示（来自 Droid-W）

**数据集：**从 Hugging Face 上的 [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) 下载演示序列。

在上述数据集的 `dynamic` 序列上运行示例（启用天空遮罩、相机优化迭代 4 次、每 2 帧设一个关键帧）：

使用天空遮罩、4 次相机优化迭代和 2 的输入步长运行 `dynamic` 序列：

```bash
python demo.py \
    --image_folder /path/to/dynamic\
    --model_path ../../Lingbot-Map/lingbot-map.pt \
    --camera_num_iterations 4 \
    --mask_sky \
    --stride 2
```



https://github.com/user-attachments/assets/567b6e9b-1cbf-402a-96be-9bab70715ec3

<img width="1453" height="1195" alt="image" src="https://github.com/user-attachments/assets/27f8c6b7-339e-4e5f-9776-7cb577147401" />





### 按关键帧间隔进行流式推理

使用 `--keyframe_interval` 仅将每第 N 帧保留为关键帧，从而减少 KV 缓存占用。非关键帧仍会生成预测，但不会存入缓存。这对超过 320 帧的长序列很有用（我们使用 320 个视图的视频 RoPE 进行训练，因此 KV 缓存存储超过 320 个视图时性能会下降；采用关键帧策略可以对更长序列进行推理）。在 demo.py 中，关键帧间隔会自动计算。

> **关于推理范围。**本方法默认不重置状态，因此最大推理范围受训练数据集中所见最长距离限制。超过该距离后，就必须重置状态。如果观察到位姿崩溃，请切换到窗口模式（`--mode windowed`）——大多数情况下，只需调整 `--keyframe_interval`，其余窗口参数保持默认值即可。


### 窗口化推理（用于超过 3000 帧的长序列）

```bash
python demo.py --model_path /path/to/lingbot-map.pt \
    --video_path video.mp4 --fps 10 \
    --mode windowed --window_size 128 --overlap_keyframes 16 --keyframe_interval 2 
```


### 天空遮罩

天空遮罩使用 ONNX 天空分割模型，从重建点云中过滤天空点，从而提升室外场景的可视化质量。

**设置：**

```bash
# Install onnxruntime (required)
pip install onnxruntime        # CPU
# or
pip install onnxruntime-gpu    # GPU (faster for large image sets)
```

首次使用时会从 [Hugging Face](https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx) 自动下载天空分割模型（`skyseg.onnx`）。

**用法：**

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky
```

天空遮罩会缓存在 `<image_folder>_sky_masks/` 中，后续运行可跳过重新生成。也可以用 `--sky_mask_dir` 指定自定义缓存目录，或用 `--sky_mask_visualization_dir` 保存并排对比的遮罩可视化结果：

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --mask_sky \
    --sky_mask_dir /path/to/cached_masks/ \
    --sky_mask_visualization_dir /path/to/mask_viz/
```

### 可视化选项

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--port` | `8080` | Viser 查看器端口 |
| `--conf_threshold` | `1.5` | 过滤低置信度点的可见性阈值 |
| `--point_size` | `0.00001` | 点云中的点大小 |
| `--downsample_factor` | `10` | 点云显示的空间下采样系数 |

### 性能与显存

#### 不使用 FlashInfer（回退到 SDPA）

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --use_sdpa
```

#### 在 GPU 显存有限时运行

如果遇到显存不足问题，请尝试以下一种或两种方法：

- **`--offload_to_cpu`**——推理期间将逐帧预测卸载到 CPU（默认启用；仅在显存充足时使用 `--no-offload_to_cpu`）。
- **`--num_scale_frames 2`**——将双向尺度帧数量从默认的 8 减少到 2，从而降低初始尺度阶段的激活峰值。

#### 加快推理

减少相机头中的迭代细化次数，以少量位姿精度换取实际运行速度：

```bash
python demo.py --model_path /path/to/checkpoint.pt \
    --image_folder /path/to/images/ --camera_num_iterations 1
```

`--camera_num_iterations` 默认为 `4`；将其设为 `1` 会跳过相机头中的三次细化过程（并将其 KV 缓存缩小为四分之一）。

## 🎥 离线渲染流水线（`demo_render/batch_demo.py`）

当序列对交互式 viser 查看器而言过长时，请使用此流水线，例如[上方精选的室内漫游](#-精选室内漫游约-25000-帧13-分钟)。`demo_render/batch_demo.py` 是一体化离线入口：向它提供视频或图像文件夹，一条命令即可运行模型推理并生成无需图形界面的点云穿行 MP4。它与 `demo.py` 共用同一套 PyTorch / FlashInfer / 检查点栈。

如果受到显存容量或 GPU 使用限制，也可以参考此实现：https://github.com/ureeey/lingbot-map-rtx4060-8g/commit/eeee84a89cc97c1e39b736b46df4ee315275700b

### 安装（在主安装流程基础上扩展）

**1. 渲染所需的 Python 依赖**

```bash
pip install -e ".[vis,render]"
```

`render` 会引入 `open3d>=0.19` 和 `pyyaml`（核心的 `numpy<2` 约束来自 `lingbot-map` 基础安装）。此流水线的天空遮罩使用 `onnxruntime-gpu` 进行批量分割；如果尚未安装 CPU 版 `onnxruntime`，请安装它：

```bash
pip install onnxruntime-gpu
```

**2. Kaolin**——与上方推荐的 PyTorch 2.8.0 + CUDA 12.8 匹配：

```bash
pip install --index-url https://pypi.org/simple \
    kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.8.0_cu128.html
```

> `--index-url https://pypi.org/simple` 可绕过内部镜像；否则内部镜像可能提供 PyPI 占位 wheel，导入时会引发 `ImportError`。
> NVIDIA Kaolin 不为 PyTorch 2.9.x 发布预编译 wheel——如果因其他原因使用 2.9，请从源码构建 Kaolin（`pip install --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git`，需要本地 CUDA 工具包）。其他 torch/CUDA 组合请参阅 [NVIDIA Kaolin 安装文档](https://kaolin.readthedocs.io/en/latest/notes/installation.html)。

**3. ffmpeg**

```bash
sudo apt install ffmpeg    # or: brew install ffmpeg
```

**4. CUDA 扩展**（首次运行前必需）

```bash
cd demo_render/render_cuda_ext && python setup.py build_ext --inplace && cd ../..
```

这会在原位置构建 `voxel_morton_ext` 和 `frustum_cull_ext`——`rgbd_render` 会导入二者，用于 GPU 体素化和视锥剔除。

### 完整示例——长室内漫游（约 25,000 帧，13 分钟）

**数据集：**从 Hugging Face 上的 [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) 下载示例视频。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/indoor_travel.MP4 \
    --output_folder /data/outputs/indoor_travel/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/indoor.yaml \
    --mode windowed --window_size 128 \
    --keyframe_interval 10 --overlap_keyframes 8 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f4f5e555-22a8-4cc9-b380-dfde5fe1c809" />

各参数的设置理由：

| 参数 | 设置理由 |
|---|---|
| `--mode windowed --window_size 128` | 序列超过约 320 帧的 RoPE 训练范围后，需要使用滑动窗口推理；每个窗口都会重置 KV 缓存。**`window_size` 计算的是 KV 缓存槽位，而不是实际帧数**——前 `num_scale_frames`（=8）个槽位保存尺度帧，其余 `128 − 8 = 120` 个槽位保存关键帧。因此，当 `keyframe_interval = 13` 时，一个窗口覆盖 `8 + 120 × 13 = 1568` 个实际帧。 |
| `--keyframe_interval 10` | 每 10 帧仅缓存一帧作为关键帧。非关键帧仍会生成逐帧预测，但不会增大 KV 缓存。|
| `--overlap_keyframes 8` | 相邻窗口共享 8 个关键帧的上下文，内部解析为 `max(num_scale_frames, 8 × keyframe_interval) = 8 × 13 = 104` 个实际重叠帧。只要 `keyframe_interval > 1` 就建议使用，以保持跨窗口位姿对齐稳定。 |
| `--config demo_render/config/indoor.yaml` | 从室内预设初始化渲染、场景、相机和叠加层的默认值（较短深度、更贴近的跟随相机）。用户显式传入的任何 CLI 参数仍会覆盖 YAML 值。 |
| `--sky_mask_dir` / `--sky_mask_visualization_dir` | 将天空遮罩及其并排可视化结果持久化到磁盘，使后续重新运行时可以复用，而无需再次执行 ONNX 分割。（仅当 YAML 预设或 `--mask_sky` 启用天空遮罩时，渲染流水线才会使用它们。） |
| `--camera_vis default` | 在渲染视频上叠加轨迹路径和最近帧的点。 |
| `--keyframes_only_points` | 只将关键帧深度反投影到点云；非关键帧的位姿仍用于轨迹/视锥叠加层。这样可让超长序列的点云保持稀疏。 |
| `--frame_tag --frame_tag_position top_right` | 在 MP4 右上角标注 `<i> / <N> Frames` 计数器。 |
| `--save_predictions` | 在 MP4 旁持久化逐帧 NPZ 文件，便于检查，或稍后使用不同相机/叠加层设置重新渲染。 |


将 keyframe_interval = 10 替换为 image_stride = 10 可以加快渲染。然后取消注释 demo_render/config/indoor.yaml 中的相机跟随部分，并将鸟瞰范围设为 [2000, 2500]，即可复现演示中的室内穿行效果：

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/5581d2b2-cb86-4187-a13d-46ac9a22ce99" />





https://github.com/user-attachments/assets/21b444ea-e6b6-48f0-8b34-3acad41166ac







### 完整示例——室外行车场景

**数据集：**从 Hugging Face 上的 [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) 下载示例视频。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/drive_frames.mp4 \
    --output_folder /data/outputs/drive/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

<img width="3822" height="1080" alt="image" src="https://github.com/user-attachments/assets/3c26afdb-6bb8-4d20-a7e0-f5a220382662" />


与上方室内漫游示例的不同之处：

| 参数 | 设置理由 |
|---|---|
| `--config demo_render/config/outdoor_drive.yaml` | 从室外预设初始化默认值：启用天空遮罩、使用更深的渲染范围（`max_depth: 250`），并采用针对车辆轨迹调优的跟随相机，最后以鸟瞰画面收尾。 |
| `--image_stride 1` | 使用视频的每一帧。增大该值可对长视频或高帧率行车视频进行下采样。 |
| `--max_non_keyframe_gap 100` | 强制设置关键帧前允许连续非关键帧的上限。仅在基于光流选择关键帧（`--flow_threshold > 0`）时生效；在默认的固定间隔模式下不起作用。 |

其余参数（`--mode windowed --window_size 128`、`--overlap_keyframes 8`、天空遮罩缓存、叠加层、`--save_predictions`）与室内示例保持一致——请参阅上方逐项参数表。

### 完整示例——LingBot-World 场景

重建由我们的世界模型 LingBot-World 生成的视频——同一流水线可直接处理生成式视频。

**数据集：**从 Hugging Face 上的 [robbyant/lingbot-map-demo](https://huggingface.co/datasets/robbyant/lingbot-map-demo/tree/main) 下载示例视频（`lingbo_world_frames.mp4`、`lingbo_world2_frames.mp4`）。

```bash
    python demo_render/batch_demo.py \
    --video_path /data/demo_videos/lingbo_world_frames.mp4 \
    --output_folder /data/outputs/lingbo_world/ \
    --model_path /path/to/lingbot-map.pt \
    --config demo_render/config/outdoor_drive.yaml \
    --mode windowed --window_size 128 \
    --max_non_keyframe_gap 100 --overlap_keyframes 8 \
    --image_stride 1 \
    --sky_mask_dir /data/outputs/sky_masks \
    --sky_mask_visualization_dir /data/outputs/sky_mask_viz \
    --camera_vis default --keyframes_only_points \
    --frame_tag --frame_tag_position top_right \
    --save_predictions
```

对于第二段视频，使用 `--video_path /data/demo_videos/lingbo_world2_frames.mp4 --output_folder /data/outputs/lingbo_world2/` 运行同一命令（如果希望分别保存缓存的遮罩，请为 `--sky_mask_dir` / `--sky_mask_visualization_dir` 使用不同文件夹）。

所有参数都与上方的[室外行车场景](#完整示例室外行车场景)相同——仅输入视频和输出文件夹不同。各参数的设置理由请参阅行车场景和室内漫游表格。

<img width="3736" height="1080" alt="image" src="https://github.com/user-attachments/assets/1f60d505-1407-482c-9b5d-57c7145c0b7d" />

<img width="1200" height="339" alt="image" src="https://github.com/user-attachments/assets/e62bedaa-1e61-40b3-8fea-01c8a15355f0" />



### 相机路径（YAML）

虚拟相机路径由 YAML 预设中的 `camera.segments` 列表描述，该预设通过 `--config` 传入。编辑 YAML 即可设计自己的镜头，无需修改 CLI 参数。

内置预设位于 `demo_render/config/`：`default.yaml`、`indoor.yaml`、`outdoor_drive.yaml`。复制其中一个并编辑 `camera:` 块。

#### YAML 结构

```yaml
camera:
  fov: 60.0          # camera field of view in degrees
  transition: 30     # frames blended between adjacent segments
  segments:
    - mode: follow            # chase cam following the input trajectory
      frames: [0, 1500]       # rendered-frame range this segment covers (-1 = end)
      back_offset: 0.3        # how far behind the input camera (fraction of scene scale)
      up_offset: 0.08         # vertical lift above the input camera
      look_offset: 0.4        # how far ahead the lookat target points
      smooth_window: 30       # trajectory smoothing window in frames
    - mode: birdeye           # rise up for a top-down reveal of the whole scene
      frames: [1500, 1800]
      reveal_height_mult: 2.5 # birdeye height = scene scale × this factor
    - mode: follow            # drop back into chase cam
      frames: [1800, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
```

`transition` 控制相邻片段之间混合多少帧；`frames: [0, -1]` 表示“整个序列”。

#### 可用模式

| `mode` | 行为 | 可调字段 |
|---|---|---|
| `follow` | 追踪相机以平滑偏移跟随输入轨迹，是漫游中最具电影感的选择。 | `back_offset`、`up_offset`、`look_offset`、`smooth_window`、`scale_frames` |
| `birdeye` | 从俯视视角展示整个场景，适合主视觉或全景镜头。 | `reveal_height_mult` |
| `static` | 固定 eye + lookat，由片段起始帧自动推导。 | — |
| `pivot` | 固定 eye，lookat 沿轨迹扫动。 | — |

#### 单镜头 YAML 示例

**纯跟随**（最常用）：

```yaml
camera:
  fov: 60.0
  segments:
    - mode: follow
      frames: [0, -1]
      back_offset: 0.3
      up_offset: 0.08
      look_offset: 0.4
      smooth_window: 30
```

**全程鸟瞰**（适合全景/主视觉镜头）：

```yaml
camera:
  fov: 60.0
  segments:
    - mode: birdeye
      frames: [0, -1]
      reveal_height_mult: 2.5
```

**跟随镜头中插入鸟瞰**：只需在 `segments:` 下按顺序列出多个片段，相邻片段会使用 `transition` 帧进行插值。

> 注意：当 `--config` 加载 YAML 预设时，传入**任何**用于设定片段形态的 CLI 参数（`--camera_mode`、`--back_offset`、`--up_offset`、`--look_offset`、`--smooth_window`、`--follow_scale_frames`、`--birdeye_start`、`--birdeye_duration`、`--reveal_height_mult`）都会丢弃 YAML 中的 `segments`，改为根据这些参数重建相机路径。若要完全由 YAML 驱动，请勿在命令行中传入这些参数。

### 输出文件

对于给定的输出名称（例如 `<scene>` 或 `<video_name>`）：

| 文件 | 说明 |
|------|-------------|
| `<name>_pointcloud.mp4` | 渲染后的点云穿行视频 |
| `<name>_pointcloud_rgb.mp4` | 编码为视频的原始 RGB 帧 |
| `<name>_pointcloud_config.yaml` | 本次运行的完整配置快照 |
| `batch_results.json` | 各场景成功状态/耗时汇总 |

## 📜 许可证

本项目根据 Apache License 2.0 发布。详情请参阅 [LICENSE](LICENSE.txt) 文件。

## 📖 引用

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## ✨ 致谢

感谢 Shangzhan Zhang、Jianyuan Wang、Yudong Jin、Christian Rupprecht 和 Xun Cao 提供的宝贵讨论与支持。

本工作基于以下优秀的开源项目构建：

- [VGGT](https://github.com/facebookresearch/vggt)
- [DINOv2](https://github.com/facebookresearch/dinov2)
- [Flashinfer](https://github.com/flashinfer-ai/flashinfer)

---
