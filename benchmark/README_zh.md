# 三维重建 Benchmark

用于评估深度估计、相机位姿和点云重建方法的三维重建 Benchmark 框架。

本框架提供标准化的三阶段流水线：**prepare**（原始数据转换为 BSS 格式） -> **run**（执行方法） -> **evaluate**（计算指标），以及可选的 **report** 生成。所有阶段通过 `.complete.json` 标记支持场景级中断与自动恢复。

---

## 目录

- [快速开始](#快速开始)
- [已支持的数据集](#已支持的数据集)
- [基准测试结果](#基准测试结果)
- [评估指标](#评估指标)
- [Viewer 系统](#viewer-系统)
- [配置系统](#配置系统)
- [BSS 存储系统](#bss-存储系统)
- [环境配置](#环境配置)
- [中断与恢复](#中断与恢复)
- [添加新方法](#添加新方法)
- [添加新数据集](#添加新数据集)

---

## 快速开始

### 1. 安装 conda 环境

涉及两个 conda env：

| Env | 用途 |
|-----|------|
| `bench` | 框架侧。承载 `prepare.py`、`evaluate.py`、`report.py`。`run.py` 也从这里启动，它会通过 `conda run` 把每个方法任务派发到方法侧 env。 |
| `lingbot_map` | 方法侧。装 PyTorch 和上游 [lingbot-map](https://github.com/robbyant/lingbot-map) 包。`run_worker.py` 在此 env 内执行模型。 |

```bash
# 框架 env（必装）。
bash envs/install_bench.sh

# 方法 env（要跑 lingbot_map 方法时必装）。
bash envs/install_lingbot_map.sh
```

如果你已经按上游 lingbot-map 装过了，`lingbot_map` env 已存在，脚本会检测并把 benchmark 侧依赖（open3d、evo、OpenEXR 等）追加进去，这样 `run_worker.py` 在方法 env 里也能读写 BSS 数据。非交互模式：`--append`（追加到已有 env）、`--force`（从零重建）。

### 2. 配置路径

仓库内的 YAML 配置文件统一使用 `/path/to/...` 占位符，运行前先替换为实际路径：

- `configs/methods/lingbot_map.yaml` — `_checkpoint` 设为 lingbot-map 权重文件路径。
- `configs/datasets/<name>.yaml` — `raw_data_root` 设为数据集本地根目录。
- `configs/<base>.yaml` — `workspace` 设为流水线输出目录。

### 3. 运行流水线

```bash
# 示例：Oxford Spires base config。其他开箱即用的数据集——
# eth3d / kitti / neural_rgbd / oxford（+ oxford_long） / seven_scenes / tat / tum / vbr / droid_w（或 all）——
# 流程一致，把 config 文件名换掉即可。
python prepare.py  --config configs/oxford.yaml
python run.py      --config configs/oxford.yaml
python evaluate.py --config configs/oxford.yaml

# 可选：生成报告
python report.py --workspace /path/to/workspace
```

### 常用参数

| 参数 | 说明 |
|------|------|
| `--force` / `-f` | 强制重跑，忽略已完成标记 |
| `--debug` | 仅处理每个数据集的第一个场景 |

### 单场景执行

`prepare.py`、`run.py`、`evaluate.py` 不接受 `--scene` 参数。如需处理单个场景，使用 `--debug`（仅第一个场景），或直接调用 `run_worker.py`：

```bash
conda run -n lingbot_map python run_worker.py \
    --config configs/oxford.yaml \
    --method lingbot_map \
    --dataset oxford \
    --scene {scene_name}
```

---

## 已支持的数据集

数据集适配器位于 `datasets/`，由 base config 通过 `datasets:` 字段引用。当前内置的适配器有：`eth3d`、`kitti`、`neural_rgbd`、`oxford_spires`、`seven_scenes`、`tnt`、`tum`、`vbr`、`droid_w`，另含一个 `general` 适配器，用于任意图像目录或视频文件（可选 COLMAP 集成自动估出内外参）。

开箱即用的 base config：

| Base config | 数据集适配器 | 启用的指标 |
|-------------|------------|-----------|
| `configs/eth3d.yaml` | `eth3d`（DA3 split） | traj + AUC + points |
| `configs/seven_scenes.yaml` | `seven_scenes`（stride 5） | traj + AUC + points |
| `configs/oxford.yaml` | `oxford_spires`（stride 12） | traj + AUC |
| `configs/oxford_long.yaml` | `oxford_spires`（stride 1，长序列） | traj |
| `configs/kitti.yaml` | `kitti`（504×280） | traj |
| `configs/vbr.yaml` | `vbr`（cover-fit 504×280） | traj |
| `configs/droid_w.yaml` | `droid_w`（宽度 518） | traj |
| `configs/tum.yaml` | `tum`（Freiburg，宽度 518） | traj |
| `configs/tat.yaml` | `tnt`（Tanks and Temples） | traj + AUC |
| `configs/neural_rgbd.yaml` | `neural_rgbd` | points |
| `configs/all.yaml` | 以上全部 | traj |

具体的数据集参数（raw data root、采样 stride、depth clip 等）在 `configs/datasets/<name>.yaml` 中配置。

### 数据准备

各数据集原始数据的获取与准备方式如下（各数据集期望的 `raw_data_root` 结构见对应的 `configs/datasets/<name>.yaml`）：

- **Oxford Spires** —— 数据准备请参考 `preprocess/oxford.py`。
- **ETH3D**、**7-Scenes**、**Neural RGB-D** —— 数据准备请参考 [Pi3](https://github.com/yyfz/Pi3)。
- **DROID-W** —— 从 [MoyangLi00/DROID-W](https://github.com/MoyangLi00/DROID-W) 下载。
- **VBR** —— 按 [Junyi42/LoGeR](https://github.com/Junyi42/LoGeR) 的预处理流程得到对齐后的数据。
- **TUM RGB-D** —— 从 [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) 下载序列。
- **KITTI** —— 从 [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) 下载里程计序列。
- TAT - 下载 Barn, Caterpillar, Church, Ignatius, Meeting room and Truck 从 [TAT](https://www.tanksandtemples.org/download/), 包括 ground truth 和 image set.

### VBR 与 DROID-W

两个仅评估轨迹的数据集，作为开箱即用示例内置。均按标准三段式命令运行：

```bash
# VBR（Vision Benchmark in Rome）——RGB + C2W TUM 轨迹 + 3x3 内参。
python prepare.py  --config configs/vbr.yaml
python run.py      --config configs/vbr.yaml
python evaluate.py --config configs/vbr.yaml

# DROID-W——RGB + C2W TUM 轨迹（按时间戳关联 GT）。
python prepare.py  --config configs/droid_w.yaml
python run.py      --config configs/droid_w.yaml
python evaluate.py --config configs/droid_w.yaml
```

运行前，编辑数据集配置使其指向本地数据根目录：

- `configs/datasets/vbr.yaml` —— `raw_data_root` 下应有 `{scene}_processed_aligned/` 目录（含 `rgb/`、`intrinsics.txt`）以及同级的 `processed_gt/{scene}_gt.txt`。`_target_size: [W, H]`（均为 14 的倍数）会对每帧做 cover-fit 缩放 + 中心裁剪，并同步更新内参。
- `configs/datasets/droid_w.yaml` —— `raw_data_root` 下应有按场景划分的目录（如 `downtown1/`），每个含 `images_anonymized/`（以 Unix 时间戳命名的 JPEG）和 `traj_gt.txt` / `traj_gt_fastlivo.txt`。`_load_img_size` 设定目标宽度（高度等比缩放并向下取整到 14 的倍数）；GT 位姿按最近时间戳与帧关联。

---

## 基准测试结果

以下结果由本流水线使用发布的 **`lingbot-map.pt`** 权重（streaming 模式）在随仓库提供的数据集配置上评测得到。每个数值为该数据集在全部评测场景上的聚合结果。

> 箭头表示更优方向：ATE / RPE / accuracy / completeness / chamfer 为**越低越好**（↓）；AUC / precision / recall / F1 为**越高越好**（↑）。RPE-rot 单位为度。

### 轨迹（ATE / RPE）

| 数据集 | 场景数 | ATE ↓ | RPE-trans ↓ | RPE-rot (°) ↓ |
|---|---:|---:|---:|---:|
| ETH3D | 11 | 0.439 | 0.493 | 3.339 |
| 7-Scenes | 18 | 0.079 | 0.020 | 0.579 |
| TUM RGB-D | 9 | 0.045 | 0.013 | 0.513 |
| Neural RGB-D | 9 | 0.056 | 0.019 | 0.257 |
| Oxford Spires | 10 | 5.374 | 0.930 | 3.694 |
| KITTI (504×280) | 11 | 24.046 | 2.861 | 0.696 |
| VBR | 7 | 31.204 | 2.717 | 4.564 |
| DROID-W | 7 | 0.909 | 0.184 | 6.115 |
| Tanks and Temples | 6 | 0.210 | 0.087 | 0.572 |

### 相机位姿 AUC

成对相对位姿在不同角度阈值（度）下的 AUC。**macro** 对各场景 AUC 等权平均；**micro** 汇总所有场景的成对误差后统一计算。

| 数据集 | 聚合方式 | AUC@3 ↑ | AUC@5 ↑ | AUC@15 ↑ | AUC@30 ↑ |
|---|---|---:|---:|---:|---:|
| ETH3D | macro | 37.22 | 50.83 | 72.99 | 81.10 |
| ETH3D | micro | 40.34 | 56.15 | 79.82 | 87.97 |
| 7-Scenes | macro | 12.35 | 23.23 | 60.01 | 78.09 |
| 7-Scenes | micro | 13.20 | 24.61 | 61.45 | 79.06 |

### 点云

点云由预测深度反投影得到（该权重以 `enable_point=False` 运行），因此这些数值反映的是深度 / 几何质量。

| 数据集 | Accuracy ↓ | Completeness ↓ | Chamfer ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ETH3D | 0.168 | 0.089 | 0.128 | 82.33 | 92.51 | 86.80 |
| 7-Scenes | 0.036 | 0.044 | 0.040 | 79.03 | 86.17 | 82.38 |
| Neural RGB-D | 0.074 | 0.030 | 0.052 | 51.77 | 89.68 | 65.10 |

### 轨迹可视化

每个数据集各取一个代表性场景。每张图把 Sim(3) 对齐后的预测轨迹（**蓝色实线**，`est`）叠加在真值轨迹（**灰色虚线**，`ref`）上，分别给出 3D 视图与三个坐标平面投影（XY / XZ / YZ）。

| ![Tanks and Temples](assets/traj/tat.png) | ![Oxford Spires](assets/traj/oxford.png) | ![KITTI](assets/traj/kitti.png) |
|:---:|:---:|:---:|
| Tanks and Temples — *Barn* | Oxford Spires — *observatory-quarter-01* | KITTI (504×280) — *seq 08* |
| ![VBR](assets/traj/vbr.png) | ![DROID-W](assets/traj/droid_w.png) | ![TUM RGB-D](assets/traj/tum.png) |
| VBR — *campus_train1* | DROID-W — *downtown3* | TUM RGB-D — *fr1/desk* |

---

## 评估指标

### 轨迹评估

若 `traj.txt` 存在则自动计算。

| 指标 | 说明 |
|------|------|
| ATE | Sim(3) 对齐后的绝对轨迹误差 RMSE |
| RPE Trans | 相对位姿误差（平移） |
| RPE Rot | 相对位姿误差（旋转，度） |

### AUC 评估

若 `traj.txt` 存在则自动计算。

| 指标 | 说明 |
|------|------|
| AUC@{3,5,15,30} | 不同角度阈值下的曲线下面积 |
| Racc@{3,5,15,30} | 旋转精度（低于阈值的 pair 百分比） |
| Tacc@{3,5,15,30} | 平移精度（低于阈值的 pair 百分比） |

聚合模式（通过 `evaluation.auc.aggregation` 配置）：

| 模式 | 说明 |
|------|------|
| `micro` | 池化所有 pair 后统一计算，帧数多的场景权重更大 |
| `macro` | 按场景分别计算后取平均，各场景等权 |
| `both` | 同时输出 `auc_micro.json` 和 `auc_macro.json` |

### 深度评估

可选，需要 GT 深度。

| 指标 | 说明 |
|------|------|
| abs_rel | 绝对相对误差 |
| sq_rel | 平方相对误差 |
| rmse | 均方根误差 |
| log_rmse | 对数均方根误差 |
| delta_1_25 | 阈值精度（1.25） |
| delta_1_25_2 | 阈值精度（1.25^2） |
| delta_1_25_3 | 阈值精度（1.25^3） |

### 点云评估

可选，由数据集实现。

| 指标 | 说明 |
|------|------|
| chamfer | Chamfer 距离（accuracy 与 completeness 均值） |
| accuracy | 预测点到 GT 的平均距离 |
| completeness | GT 到预测点的平均距离 |
| precision_T | 预测点中距 GT 小于阈值 T 的百分比 |
| recall_T | GT 中距预测点小于阈值 T 的百分比 |
| f1_T | precision 与 recall 的调和平均 |

---

## Viewer 系统

`viewer.py` 是基于 [viser](https://github.com/nerfstudio-project/viser) 的浏览器端交互式 3D 查看器，直接读取 BSS workspace，支持查看真值和各方法的输出。

### 使用方法

```bash
# 查看工作空间中的所有数据
python viewer.py /path/to/workspace

# 自定义端口和采样参数
python viewer.py /path/to/workspace -p 8080 -t 5 -s 4
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-p` / `--port` | 20540 | viser 服务端口 |
| `-t` / `--temporal-subsample` | 1 | 每 N 帧加载一帧 |
| `-s` / `--spatial-subsample` | 2 | 点云空间降采样倍数 |
| `--verbose` | 关闭 | 详细日志输出 |

### 功能

- **数据选择**：下拉菜单切换 数据集 / 场景 / 方法（含 gt），即时加载
- **逐帧点云**：深度图 + 轨迹反投影到世界坐标，支持置信度过滤
- **全局点云**：显示 `points.ply`（若存在）
- **相机视锥与轨迹**：可切换显示/隐藏，视锥大小可调
- **回放控制**：时间轴滑块、播放/暂停、FPS 调节、循环播放、首帧/上一帧/下一帧/末帧导航
- **历史帧**：独立滑块控制显示多少个历史相机视锥和历史点云帧
- **天空去除**：可选天空分割以过滤天空像素（首次运行后缓存结果）
- **点外观**：对数缩放点大小、运行时额外降采样
- **自动对齐**：若 `traj_transform.txt` 存在（evaluate 阶段生成的 Sim(3) Umeyama 对齐矩阵），viewer 会自动将预测轨迹和点云变换到 GT 坐标系下显示。GUI 中会标注对齐状态（GT / 已对齐 / 未对齐）
- **相机剪贴板**：复制当前相机视角（位置、朝向、上方向、FoV），在另一个浏览器客户端中粘贴恢复。适用于从完全相同的视角对比不同方法的重建结果
- **场景缓存**：预处理点云缓存到磁盘，可从 GUI 清除缓存
- **RGB 缩略图**：侧栏显示当前帧 RGB 图像

---

## 配置系统

框架采用三层 YAML 配置，位于 `configs/` 目录下。

### 第一层：基础配置

文件路径：`configs/*.yaml`

定义 workspace 路径、数据集与方法选择列表，以及全局 evaluation 默认值。

```yaml
workspace: /path/to/workspace

datasets:
  - oxford

methods:
  - lingbot_map

evaluation:
  traj:
    enable: true
    vis: true
  auc:
    enable: true
    vis: true
    aggregation: both
  depth:
    enable: false
  points:
    enable: false
```

### 第二层：数据集配置

文件路径：`configs/datasets/*.yaml`

平级文件，`dataset:` 字段映射到 `datasets/{module}.py`，`_` 前缀的键作为 `__init__` 参数传入。

```yaml
dataset: oxford_spires
raw_data_root: /path/to/oxford_spires
sampling:
  strategy: sequence
  stride: 12
evaluation:
  depth:
    gt_clip:
      min: 0.0
      max: 200.0
```

### 第三层：方法配置

文件路径：`configs/methods/*.yaml`

平级文件，`model:` 字段映射到 `methods/{module}.py`，`env:` 指定 conda 环境名，`_` 前缀的键作为 `__init__` 参数传入。

```yaml
model: lingbot_map
env: lingbot_map
_checkpoint: /path/to/lingbot-map.pt
_device: cuda
_mode: streaming
_use_amp: true
_image_size: 518
_patch_size: 14
_area_budget: 255000
_align: 14
```

### 配置合并顺序

Evaluation 配置按以下优先级合并：基础默认值 -> 数据集覆盖 -> 方法覆盖。

---

## BSS 存储系统

BSS（Benchmark Storage Structure）是本框架的统一数据存储格式。所有数据集经 prepare 阶段转换后，以及所有方法的输出，均以 BSS 格式存储在 workspace 中。

### 目录布局

```
workspace/
└── {dataset_name}/
    └── {scene_safe}/                   # 场景名中 '/' 替换为 '_'
        ├── gt/                         # 真值
        │   ├── .complete.json          # 完成标记
        │   ├── sampling.json           # 采样配置
        │   ├── resize.json             # 尺寸变换
        │   ├── rgb/                    # {timestamp}.png -- HxWx3 uint8 RGB
        │   ├── depth/                  # {timestamp}.exr -- float32 米
        │   ├── mask/                   # {timestamp}.png -- 感兴趣区域掩码
        │   ├── traj.txt                # 相机轨迹：时间戳 + 3x4 C2W（行优先）
        │   ├── intrinsics.txt          # 7 列：时间戳 fx fy cx cy width height
        │   └── points.ply              # 可选：GT 点云（Nx3 或 Nx6）
        │
        └── {method_name}/              # 方法输出
            ├── .complete.json
            ├── resize.json
            ├── rgb/
            ├── depth/                  # 预测深度
            ├── points/                 # 逐帧世界坐标点云（HxWx3 EXR）
            ├── confidence/             # 逐帧置信度图（HxW EXR）
            ├── traj.txt
            ├── intrinsics.txt
            ├── points.ply              # 可选：全局点云
            └── eval/                   # Layer 1：方法级评估结果
                ├── traj.json
                ├── auc.json
                ├── depth.json
                ├── points.json
                ├── traj_transform.txt  # Sim(3) 对齐矩阵
                ├── traj/               # 可视化目录
                ├── auc/
                ├── depth/
                └── points/
```

### 聚合目录

```
workspace/{dataset}/
├── {scene}/
│   └── eval/                           # Layer 2：场景级跨方法对比
│       ├── traj.json
│       ├── auc.json
│       ├── depth.json
│       └── points.json
│
└── eval/                               # Layer 3：数据集级聚合
    ├── auc_micro.json
    ├── auc_macro.json
    ├── traj.json
    ├── depth.json
    └── points.json
```

**Layer 1** 是原始评估数据，**Layer 2/3** 是由 Layer 1 聚合得到的派生视图，每次 evaluate 运行时重新生成。

### 数据格式规范

| 数据 | 格式 |
|------|------|
| RGB 图像 | HxWx3 uint8，RGB 通道顺序，sRGB |
| 深度图 | HxW float32，单位：米；无效深度 = 0 |
| 时间戳 | str，规范格式 `f"{float(ts):016.6f}"` |
| 相机姿态 | 4x4 C2W 矩阵 |
| 轨迹文件 | 每行 13 值：时间戳 r00 r01 r02 tx r10 ... r22 tz |
| 内参文件 | 每行 7 值（含注释头）：时间戳 fx fy cx cy width height |
| 点云 | `.ply`，Nx3 或 Nx6（xyzrgb），RGB 在 [0,1] |
| 深度/置信度存储 | `.exr`（OpenEXR） |

---

## 环境配置

### 系统前提

- CUDA 12.1（nvcc）/ Driver 支持 CUDA 13.0
- Conda（推荐 miniforge/mamba）

### 安装命令

```bash
# 框架 env（仅含 numpy/opencv/open3d/evo 等，无 PyTorch）。
# prepare.py / evaluate.py / report.py / run.py 都在此 env 内运行。
bash envs/install_bench.sh

# lingbot_map 方法 env。脚本会检测已有的 `lingbot_map` env（由上游
# lingbot-map 安装流程创建），并追加 bench 依赖。env 不存在时会从零构建。
bash envs/install_lingbot_map.sh                # 交互式
bash envs/install_lingbot_map.sh --append       # 非交互：追加 bench deps
bash envs/install_lingbot_map.sh --force        # 非交互：从零重建

# 自动发现 envs/ 下所有 install_*.sh 并依次执行（字母序）。
bash envs/install_all.sh
```

所有安装脚本均幂等，重复执行不会出错。仓库默认只附带 `install_bench.sh` 和 `install_lingbot_map.sh`；用户接入新方法时，把 `envs/install_<name>.sh` 放到同目录，`install_all.sh` 会自动识别并执行。约定 conda env 名直接用方法名本身（如 `lingbot_map` 而非 `lingbot_map_env`），但方法配置中的 `env` 字段可以覆盖此约定。

### `bench` env

必装。承载 `prepare.py`、`evaluate.py`、`report.py` 和 `run.py`（dispatcher）。主要依赖：numpy、opencv、open3d、evo、matplotlib、pyyaml、tqdm，外加可视化用的 imageio、trimesh、plyfile、OpenEXR。

---

## 中断与恢复

所有阶段（prepare、run、evaluate）通过 `.complete.json` 标记文件支持自动恢复，粒度为场景级：

1. 流水线开始处理场景
2. 中断执行（如 Ctrl+C）
3. 重新运行相同命令，已完成的场景自动跳过

使用 `--force` / `-f` 参数可忽略完成标记，强制重新处理。

---

## 添加新方法

> **说明**：本仓库仅内置维护 `lingbot_map` 一个方法作为示例。论文实验中使用的其他对比方法（如 VGGT、Fast3R、DROID-SLAM、MegaSaM、StreamVGGT、TTT3R 等）均有各自的上游 repo，本仓库**不再维护**它们的 wrapper。如需复现对比实验，请按下列步骤自行接入。`methods/lingbot_map.py` 和 `configs/methods/lingbot_map.yaml` 可作为接入参考。

### 步骤 1：Clone 方法仓库

约定以 `_repo` 后缀放在 `methods/` 下：

```bash
git clone https://github.com/example/method.git methods/method_repo
```

### 步骤 2：配置 conda 环境

使用 conda 做环境隔离，默认命名 `{method}_env`，也可以自行命名后在方法配置中通过 `env` 字段指定。

### 步骤 3：创建方法模块

创建 `methods/{name}.py`，实现 `process_scene` 方法：

```python
from benchmark.method.base import BaseMethod
from benchmark.core.loader import BSSLoader


class MyMethodMethod(BaseMethod):
    def __init__(self, checkpoint, device='cuda',
                 area_budget=255000, align=14, logger=None):
        super().__init__(area_budget=area_budget, align=align, logger=logger)
        # 加载模型...

    def process_scene(self, gt_artifact):
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        rgb_list = loader.load_rgb_list()
        timestamps = loader.get_timestamps()

        # 执行推理...

        return {
            'frame': {
                'rgb': rgb_list,         # 必填
                'depth': depth_list,     # 可选
                'pose': pose_list,       # 可选，4x4 C2W
                'intrinsics': intr_list, # 可选，[fx, fy, cx, cy]
                'confidence': conf_list, # 可选，HxW
                'points': pts_list,      # 可选，HxWx3 世界坐标
            },
            'global': {}
        }
```

### 步骤 4：编写 YAML 配置

在 `configs/methods/` 下创建配置文件，指定 `model`、`env` 和 `_` 前缀参数：

```yaml
model: my_method
env: my_method_env
_checkpoint: methods/method_repo/checkpoints/model.pth
_device: cuda
_area_budget: 255000
_align: 14
```

### 步骤 5：编写安装脚本（可选）

在 `envs/` 下编写 `install_{method}.sh`，脚本应幂等（重复执行不出错）。命名约定：conda env 名直接用方法名本身。

### 类名约定

snake_case 模块名自动映射为 PascalCase 类名：

| 模块文件名 | 类名 |
|-----------|------|
| `lingbot_map.py` | `LingbotMapMethod` |
| `seven_scenes.py` | `SevenScenesDataset` |
| `my_method.py` | `MyMethodMethod` |

无需手动注册，框架通过反射自动发现。

### 图像缩放

方法在 YAML 配置中声明 `area_budget` 与 `align`，`BSSLoader` 会自动等比缩放图像使 `W * H <= area_budget`，并将两个维度对齐到 `align` 的整数倍，同步调整相机内参。省略 `area_budget`（或设为 `None`）即按原分辨率加载。

| 模式 | 行为 |
|------|------|
| `none` | 不缩放（省略 `area_budget` 时的默认行为） |
| `area_budget` | 等比缩放使 `W * H <= area_budget`，尺寸对齐到 `align` |

若方法需要更复杂的预处理（letterbox、方形裁剪等），请在 wrapper 的 `process_scene()` 内自行处理图像，并返回对应调整后的内参。

### 方法子进程调度

如果方法配置中包含 `env` 字段，`run.py` 会通过以下方式启动子进程，而非在主进程中运行：

```bash
conda run -n {env} python run_worker.py --config ... --method ... --dataset ...
```

此机制隔离了各方法的 Python/CUDA 依赖，避免冲突。

---

## 添加新数据集

### 创建数据集模块

创建 `datasets/{name}.py`，实现以下接口：

```python
from benchmark.dataset.base import BaseDataset


class MyDatasetDataset(BaseDataset):
    def __init__(self, raw_data_root, logger=None, **kwargs):
        super().__init__(raw_data_root, logger)

    def get_scenes(self):
        """返回场景 ID 列表"""
        # 例如: ['scene_01', 'scene_02', ...]

    def get_frame_list(self, scene):
        """返回帧 ID 列表（整数）"""
        # 例如: [0, 1, 2, ..., 99]

    def load_frame_data(self, scene, frame_id):
        """加载单帧数据

        必填键：
            'timestamp' (float): 帧时间戳
            'rgb' (np.ndarray): HxWx3 uint8 RGB 图像

        可选键：
            'depth' (np.ndarray): HxW float32 深度图，单位米
            'pose' (np.ndarray): 4x4 C2W 变换矩阵
            'intrinsics' (np.ndarray): [fx, fy, cx, cy]
            'mask' (np.ndarray): HxW bool 掩码
        """
        return {
            'timestamp': ...,
            'rgb': ...,
            'depth': ...,      # 可选
            'pose': ...,       # 可选
            'intrinsics': ..., # 可选
        }

    def load_global_data(self, scene):
        """加载场景全局数据（可选）

        可选键：
            'points' (np.ndarray): Nx3 或 Nx6 点云
        """
        return {}
```

### 自定义保存器

如果数据集包含非标准数据类型，可实现自定义保存方法：

```python
def __save_{key}_file__(self, key_dir, timestamp, data):
    """自定义保存逻辑，key_dir 为输出子目录路径"""
    pass
```

### 自定义点云评估

数据集可实现静态方法来覆盖默认的点云评估逻辑：

```python
@staticmethod
def evaluate_pointcloud(gt_loader, pred_loader, logger, options=None):
    """自定义点云评估"""
    pass
```
