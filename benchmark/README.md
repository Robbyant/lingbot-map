# 3D Reconstruction Benchmark

A benchmark framework for evaluating depth estimation, camera pose, and point cloud reconstruction methods. The pipeline has three phases -- **prepare** (raw data to BSS format), **run** (execute methods), **evaluate** (compute metrics) -- plus optional **report** generation.

**We use lingbot-map.pt by default in the evaluation benchmark.**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Datasets](#supported-datasets)
3. [Benchmark Results](#benchmark-results)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Viewer](#viewer)
6. [Configuration System](#configuration-system)
7. [BSS Storage System](#bss-storage-system)
8. [Environment Setup](#environment-setup)
9. [Interrupt and Resume](#interrupt-and-resume)
10. [Adding a New Method](#adding-a-new-method)
11. [Adding a New Dataset](#adding-a-new-dataset)

---

## Quick Start

### 1. Install the conda envs

Two conda envs are involved:

| Env | Purpose |
|-----|---------|
| `bench` | Framework side. Hosts `prepare.py`, `evaluate.py`, `report.py`. `run.py` is also launched from here — it dispatches each method job to the corresponding method env via `conda run`. |
| `lingbot_map` | Method side. Holds PyTorch and the upstream [lingbot-map](https://github.com/robbyant/lingbot-map) package. `run_worker.py` runs inside this env to execute the model. |


```bash
# Method env (mandatory if you want to run the lingbot_map method).
bash envs/install_lingbot_map.sh

# Benchmark env (optional)
bash envs/install_bench.sh
```

If you already followed the upstream lingbot-map install, a `lingbot_map` env already exists. The script detects it and appends benchmark-side deps (open3d, evo, OpenEXR, ...) into it so that `run_worker.py` can read/write BSS data from inside the method env. Non-interactive flags: `--append` (append to existing env), `--force` (rebuild from scratch).


or you can try to install the following package based on following commands to prepare the env for evaluation:
```bash
pip install evo
pip install OpenEXR
pip install six
```

### 2. Configure paths

The shipped YAML files use `/path/to/...` placeholders. Before running anything, replace them with real paths:

- `configs/methods/lingbot_map.yaml` — set `_checkpoint` to the lingbot-map weights file.
- `configs/datasets/<name>.yaml` — set `raw_data_root` to the dataset's local root.
- `configs/<base>.yaml` — set `workspace` to where pipeline outputs should be written.

### 3. Run the pipeline

```bash
# Example: Oxford Spires base config. Other shipped datasets —
# eth3d / kitti / neural_rgbd / oxford (+ oxford_long) / seven_scenes / tat / tum / vbr / droid_w (or all) —
# follow the same three-command pattern.
python prepare.py  --config configs/oxford.yaml
python run.py      --config configs/oxford.yaml
python evaluate.py --config configs/oxford.yaml

# Optional: generate report
python report.py --workspace /path/to/workspace
```

### Useful flags

| Flag | Effect |
|------|--------|
| `--force` / `-f` | Re-run even if already complete |
| `--debug` | Process only the first scene per dataset |

`prepare.py`, `run.py`, and `evaluate.py` do **not** accept `--scene`. To run a single scene, use `--debug` (first scene only) or call `run_worker.py` directly:

```bash
conda run -n lingbot_map python run_worker.py \
    --config configs/oxford.yaml \
    --method lingbot_map \
    --dataset oxford \
    --scene <scene_name>
```

---

## Supported Datasets

Dataset adapters live in `datasets/` and are referenced from base configs via the `datasets:` field. Adapters currently shipped: `eth3d`, `kitti`, `neural_rgbd`, `oxford_spires`, `seven_scenes`, `tnt`, `tum`, `vbr`, `droid_w`, plus a `general` adapter that wraps an ad-hoc image folder or video file (optional COLMAP integration for intrinsics/extrinsics).

Ready-to-use base configs under `configs/`:

| Base config | Dataset adapter | Enabled metrics |
|-------------|----------------|-----------------|
| `configs/eth3d.yaml` | `eth3d` (DA3 split) | traj + AUC + points |
| `configs/seven_scenes.yaml` | `seven_scenes` (stride 5) | traj + AUC + points |
| `configs/oxford.yaml` | `oxford_spires` (stride 12) | traj + AUC |
| `configs/oxford_long.yaml` | `oxford_spires` (stride 1, long sequences) | traj |
| `configs/kitti.yaml` | `kitti` (504×280) | traj |
| `configs/vbr.yaml` | `vbr` (cover-fit 504×280) | traj |
| `configs/droid_w.yaml` | `droid_w` (width 518) | traj |
| `configs/tum.yaml` | `tum` (Freiburg, width 518) | traj |
| `configs/tat.yaml` | `tnt` (Tanks and Temples) | traj + AUC |
| `configs/neural_rgbd.yaml` | `neural_rgbd` | points |
| `configs/all.yaml` | all of the above | traj |

Per-dataset settings (raw data root, sampling stride, depth clip, ...) live in `configs/datasets/<name>.yaml`.

### Data preparation

Where to obtain and how to prepare each dataset's raw data (see the matching `configs/datasets/<name>.yaml` for the expected `raw_data_root` layout):

- **Oxford Spires** — prepare the data with `preprocess/oxford.py`.
- **ETH3D**, **7-Scenes**, **Neural RGB-D** — follow the data preparation in [Pi3](https://github.com/yyfz/Pi3).
- **DROID-W** — download from [MoyangLi00/DROID-W](https://github.com/MoyangLi00/DROID-W).
- **VBR** — follow the preprocessing in [Junyi42/LoGeR](https://github.com/Junyi42/LoGeR) to obtain the aligned data.
- **TUM RGB-D** — download sequences from the [TUM RGB-D benchmark](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
- **KITTI** — download the odometry sequences from the [KITTI odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
- **TAT** - download the Barn, Caterpillar, Church, Ignatius, Meeting room and Truck from [TAT](https://www.tanksandtemples.org/download/), including ground truth and image set.

### VBR and DROID-W

Two trajectory-only datasets shipped as drop-in examples. Both run via the standard three-command pattern:

```bash
# VBR (Vision Benchmark in Rome) — RGB + C2W TUM trajectory + 3x3 intrinsics.
python prepare.py  --config configs/vbr.yaml
python run.py      --config configs/vbr.yaml
python evaluate.py --config configs/vbr.yaml

# DROID-W — RGB + C2W TUM trajectory (timestamp-associated GT).
python prepare.py  --config configs/droid_w.yaml
python run.py      --config configs/droid_w.yaml
python evaluate.py --config configs/droid_w.yaml

# TUM RGB-D — RGB + C2W trajectory (timestamp-associated GT).
python prepare.py  --config configs/tum.yaml
python run.py      --config configs/tum.yaml
python evaluate.py --config configs/tum.yaml
```

Before running, edit the dataset configs to point at your local data root:

- `configs/datasets/vbr.yaml` — `raw_data_root` expects `{scene}_processed_aligned/` dirs (with `rgb/`, `intrinsics.txt`) plus a sibling `processed_gt/{scene}_gt.txt`. `_target_size: [W, H]` (multiples of 14) cover-fit resizes and center-crops each frame, updating intrinsics accordingly.
- `configs/datasets/droid_w.yaml` — `raw_data_root` expects per-scene dirs (e.g. `downtown1/`) each holding `images_anonymized/` (JPEGs named by Unix timestamp) and a `traj_gt.txt` / `traj_gt_fastlivo.txt`. `_load_img_size` sets the target width (height scaled and floored to a multiple of 14); GT poses are matched to frames by nearest timestamp.
- `configs/datasets/tum.yaml` — `raw_data_root` expects the unpacked `rgbd_dataset_freiburg*/` sequence dirs (each with `rgb/` PNGs named by timestamp and a `groundtruth.txt`). `_load_img_size` sets the target width (height floored to a multiple of 14); intrinsics use the official TUM Freiburg factory calibration, and each RGB frame is matched to the nearest GT pose within 0.02 s.

---

## Benchmark Results

Results below are produced by this pipeline with the released **`lingbot-map.pt`** checkpoint (streaming mode), evaluated on the shipped dataset configs. Each number is the dataset-level aggregate over all evaluated scenes.

> Arrows mark the better direction: ATE / RPE / accuracy / completeness / chamfer are **lower-is-better** (↓); AUC / precision / recall / F1 are **higher-is-better** (↑). RPE-rot is in degrees.

### Trajectory (ATE / RPE)

| Dataset | #Scenes | ATE ↓ | RPE-trans ↓ | RPE-rot (°) ↓ |
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

### Camera Pose AUC

Pairwise relative-pose AUC at angular thresholds (degrees). **macro** averages per-scene AUC equally; **micro** pools all pairwise errors across scenes.

| Dataset | Aggregation | AUC@3 ↑ | AUC@5 ↑ | AUC@15 ↑ | AUC@30 ↑ |
|---|---|---:|---:|---:|---:|
| ETH3D | macro | 37.22 | 50.83 | 72.99 | 81.10 |
| ETH3D | micro | 40.34 | 56.15 | 79.82 | 87.97 |
| 7-Scenes | macro | 12.35 | 23.23 | 60.01 | 78.09 |
| 7-Scenes | micro | 13.20 | 24.61 | 61.45 | 79.06 |

### Point Cloud

Point clouds are obtained by back-projecting predicted depth (the checkpoint runs with `enable_point=False`), so these numbers reflect depth / geometry quality.

| Dataset | Accuracy ↓ | Completeness ↓ | Chamfer ↓ | Precision ↑ | Recall ↑ | F1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| ETH3D | 0.168 | 0.089 | 0.128 | 82.33 | 92.51 | 86.80 |
| 7-Scenes | 0.036 | 0.044 | 0.040 | 79.03 | 86.17 | 82.38 |
| Neural RGB-D | 0.074 | 0.030 | 0.052 | 51.77 | 89.68 | 65.10 |

### Trajectory Visualizations

One representative scene per dataset. Each panel overlays the Sim(3)-aligned predicted trajectory (**solid blue**, `est`) on the ground truth (**dashed gray**, `ref`), viewed in 3D plus the three coordinate-plane projections (XY / XZ / YZ).

| ![Tanks and Temples](assets/traj/tat.png) | ![Oxford Spires](assets/traj/oxford.png) | ![KITTI](assets/traj/kitti.png) |
|:---:|:---:|:---:|
| Tanks and Temples — *Barn* | Oxford Spires — *observatory-quarter-01* | KITTI (504×280) — *seq 08* |
| ![VBR](assets/traj/vbr.png) | ![DROID-W](assets/traj/droid_w.png) | ![TUM RGB-D](assets/traj/tum.png) |
| VBR — *campus_train1* | DROID-W — *downtown3* | TUM RGB-D — *fr1/desk* |

---

## Evaluation Metrics

### Trajectory (automatic if `traj.txt` exists)

| Metric | Description |
|--------|-------------|
| ATE | Sim(3)-aligned RMSE of absolute trajectory error |
| RPE Trans | RMSE of frame-to-frame relative translation error |
| RPE Rot | RMSE of frame-to-frame relative rotation error |

### AUC (automatic if `traj.txt` exists)

| Metric | Description |
|--------|-------------|
| AUC@{3,5,15,30} | Area under curve at angular thresholds (degrees) |
| Racc@{3,5,15,30} | Rotation accuracy: fraction of pairs below threshold |
| Tacc@{3,5,15,30} | Translation accuracy: fraction of pairs below threshold |

Aggregation modes (configured via `evaluation.auc.aggregation`):

- **micro**: Pool all pairwise errors across scenes, compute AUC once. Larger scenes dominate due to O(N^2) pairs.
- **macro**: Compute AUC per scene, then take the arithmetic mean. Each scene weighted equally.
- **both**: Output both `auc_micro.json` and `auc_macro.json` at the dataset level.

### Depth (optional, requires GT depth)

| Metric | Description |
|--------|-------------|
| abs_rel | Absolute relative error |
| sq_rel | Squared relative error |
| rmse | Root mean squared error |
| log_rmse | Log-scale RMSE |
| delta_1_25 | Fraction of pixels with max(pred/gt, gt/pred) < 1.25 |
| delta_1_25_2 | Same threshold at 1.25^2 |
| delta_1_25_3 | Same threshold at 1.25^3 |

### Point cloud (optional, dataset-specific)

| Metric | Description |
|--------|-------------|
| chamfer | Average of accuracy and completeness |
| accuracy | Mean distance from predicted points to GT |
| completeness | Mean distance from GT points to predicted |
| precision_T | Fraction of predicted points within threshold T of GT |
| recall_T | Fraction of GT points within threshold T of predicted |
| f1_T | Harmonic mean of precision_T and recall_T |

---

## Viewer

`viewer.py` is a browser-based interactive 3D viewer built on [viser](https://github.com/nerfstudio-project/viser). It reads directly from the BSS workspace and supports both ground truth and method outputs.

### Usage

```bash
# View all data in workspace
python viewer.py /path/to/workspace

# Custom port and subsampling
python viewer.py /path/to/workspace -p 8080 -t 5 -s 4
```

| Flag | Default | Description |
|------|---------|-------------|
| `-p` / `--port` | 20540 | Viser server port |
| `-t` / `--temporal-subsample` | 1 | Load every N-th frame |
| `-s` / `--spatial-subsample` | 2 | Downsample point clouds by factor N |
| `--verbose` | off | Verbose logging |

### Features

- **Data selection**: dropdown menus for dataset / scene / method (including gt); switches on the fly
- **Per-frame point clouds**: depth + trajectory back-projected into world coordinates, with confidence-based filtering
- **Global point clouds**: displays `points.ply` when available
- **Camera frustums and trajectory**: toggle visibility, adjustable frustum size
- **Playback**: timeline slider, play / pause, FPS control, loop mode, first / prev / next / end navigation
- **History frames**: separate sliders for how many past camera frustums and point cloud frames to show
- **Sky removal**: optional sky segmentation to filter out sky pixels (cached after first run)
- **Point appearance**: logarithmic point-size scaling, additional runtime downsampling
- **Automatic alignment**: if `traj_transform.txt` exists (the Sim(3) matrix produced by the evaluate phase), the viewer applies it to align predicted trajectories and point clouds into the GT coordinate frame. Alignment status is shown in the GUI (GT / Aligned / Not aligned)
- **Camera clipboard**: copy the current camera viewpoint (position, look-at, up, FoV) and paste it in another browser client. This is useful for comparing different methods from exactly the same viewing angle
- **Scene caching**: pre-processed point clouds are cached to disk; cache can be cleared from the GUI
- **RGB thumbnail**: current frame's RGB image displayed in the sidebar

---

## Configuration System

Configuration is split across three layers of YAML files.

### Layer 1: Base config (`configs/<name>.yaml`)

Selects workspace path, datasets, methods, and global evaluation defaults.

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

### Layer 2: Dataset config (`configs/datasets/<name>.yaml`)

Flat file. The `dataset:` field maps to `datasets/<module>.py`. Keys prefixed with `_` are passed as kwargs to the dataset constructor.

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

### Layer 3: Method config (`configs/methods/<name>.yaml`)

Flat file. The `model:` field maps to `methods/<module>.py`. The `env:` field specifies the conda environment for subprocess dispatch. Keys prefixed with `_` are passed as kwargs to the method constructor.

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

### Config merge order

Evaluation config merges in this order (later values override earlier ones):

1. Base defaults
2. Dataset overrides
3. Method overrides

---

## BSS Storage System

BSS (Benchmark Storage Structure) is the canonical on-disk format. All pipeline phases read and write this layout.

### Directory layout

```
workspace/
└── {dataset_name}/
    └── {scene_safe}/                   # '/' in scene names replaced with '_'
        ├── gt/                         # Ground truth
        │   ├── .complete.json          # Completion marker
        │   ├── sampling.json           # Sampling config
        │   ├── resize.json             # Resize transform
        │   ├── rgb/                    # {timestamp}.png - HxWx3 uint8 RGB
        │   ├── depth/                  # {timestamp}.exr - float32 meters
        │   ├── mask/                   # {timestamp}.png - area-of-interest mask
        │   ├── traj.txt                # Benchmark Matrix format: timestamp + 3x4 C2W (row-major)
        │   ├── intrinsics.txt          # 7-col: timestamp fx fy cx cy width height
        │   └── points.ply              # Optional: GT point cloud (Nx3 or Nx6)
        │
        └── {method_name}/              # Method output
            ├── .complete.json
            ├── resize.json
            ├── rgb/
            ├── depth/                  # Predicted depth
            ├── points/                 # Per-frame world-coord point clouds (HxWx3 EXR)
            ├── confidence/             # Per-frame confidence maps (HxW EXR)
            ├── traj.txt
            ├── intrinsics.txt
            ├── points.ply              # Optional: global point cloud
            └── eval/                   # Layer 1 evaluation
                ├── traj.json
                ├── auc.json
                ├── depth.json
                ├── points.json
                ├── traj_transform.txt  # Sim(3) alignment matrix
                ├── traj/               # Visualization directories
                ├── auc/
                ├── depth/
                └── points/
```

### Aggregation layers

```
workspace/{dataset}/
├── {scene}/
│   └── eval/                           # Layer 2: scene-level cross-method comparison
│       ├── traj.json
│       ├── auc.json
│       ├── depth.json
│       └── points.json
│
└── eval/                               # Layer 3: dataset-level aggregation
    ├── auc_micro.json
    ├── auc_macro.json
    ├── traj.json
    ├── depth.json
    └── points.json
```

Layer 1 (per-scene, per-method) is the primary data source. Layers 2 and 3 are derived views recomputed from Layer 1 on each evaluation run.

### Data format conventions

| Data | Format |
|------|--------|
| RGB | HxWx3 uint8, RGB channel order, sRGB |
| Depth | HxW float32, meters; invalid pixels = 0 |
| Timestamps | String, canonical format `f"{float(ts):016.6f}"` |
| Camera pose | 4x4 camera-to-world (C2W) matrix |
| Trajectory file | 13 values per line: `timestamp r00 r01 r02 tx r10 ... r22 tz` |
| Intrinsics file | 7 values per line with header: `timestamp fx fy cx cy width height` |
| Point clouds | `.ply`, Nx3 or Nx6 (xyzrgb), RGB values in [0, 1] |
| Depth / confidence storage | `.exr` (OpenEXR) |

---

## Environment Setup

### Prerequisites

- CUDA 12.1 (nvcc) / Driver supporting CUDA 13.0
- Conda (miniforge / mamba recommended)

### Installation

```bash
# Framework env (numpy/opencv/open3d/evo/...; no PyTorch).
# Required to run prepare.py / evaluate.py / report.py / run.py.
bash envs/install_bench.sh

# Method env for lingbot_map. Detects an existing `lingbot_map` env
# (set up via the upstream lingbot-map repo) and appends bench deps to it.
# Falls back to creating the env from scratch when it does not exist.
bash envs/install_lingbot_map.sh                # interactive
bash envs/install_lingbot_map.sh --append       # non-interactive append
bash envs/install_lingbot_map.sh --force        # rebuild env from scratch

# Run every install_*.sh under envs/ (auto-discovered, alphabetical order).
bash envs/install_all.sh
```

All install scripts are idempotent. The repo only ships `install_bench.sh` and `install_lingbot_map.sh`; when you integrate additional methods, drop `envs/install_<name>.sh` next to them and `install_all.sh` will pick them up automatically. The convention is to name the conda env after the method itself (`lingbot_map`, not `lingbot_map_env`), but the `env` field in the method config can override this.

### `bench` env

Required: hosts `prepare.py`, `evaluate.py`, `report.py`, and `run.py` (the dispatcher). Main dependencies: numpy, opencv, open3d, evo, matplotlib, pyyaml, tqdm, plus a few extras for visualization (imageio, trimesh, plyfile, OpenEXR).

---

## Interrupt and Resume

All pipeline phases support automatic resumption. Progress is tracked at scene-level granularity via `.complete.json` marker files. If a run is interrupted (e.g., Ctrl+C or crash), re-running the same command will skip already-completed scenes and continue from where it left off.

---

## Adding a New Method

> **Note**: This repository bundles only `lingbot_map` as a maintained example. Other methods used in our experiments (e.g. VGGT, Fast3R, DROID-SLAM, MegaSaM, StreamVGGT, TTT3R, ...) each have their own upstream repos and are **not** maintained here. To reproduce comparisons against them, follow the steps below to integrate them yourself. `methods/lingbot_map.py` and `configs/methods/lingbot_map.yaml` serve as a reference wrapper.

### Step 1: Clone the method repository

Place the repository under `methods/` using the `_repo` suffix convention:

```bash
git clone https://github.com/example/method.git methods/method_repo
```

### Step 2: Set up the conda environment

Create a conda environment for the method. The convention is to name it after the method itself (e.g. `lingbot_map`). The `env` field in the method config can be customized to any conda env name.

### Step 3: Create the method module

Create `methods/<name>.py`. The class name must follow the snake_case-to-PascalCase convention: the module name `my_method` maps to the class `MyMethodMethod`.

```python
from benchmark.method.base import BaseMethod
from benchmark.core.loader import BSSLoader


class MyMethodMethod(BaseMethod):
    def __init__(self, checkpoint, device='cuda',
                 area_budget=255000, align=14, logger=None):
        super().__init__(area_budget=area_budget, align=align, logger=logger)
        # Load model weights, initialize state, etc.

    def process_scene(self, gt_artifact):
        loader = BSSLoader(gt_artifact, resize_context=self.resize_context)
        rgb_list = loader.load_rgb_list()
        timestamps = loader.get_timestamps()

        # Run inference...

        return {
            'frame': {
                'rgb': rgb_list,          # REQUIRED
                'depth': depth_list,      # Optional: predicted depth maps
                'pose': pose_list,        # Optional: 4x4 C2W matrices
                'intrinsics': intr_list,  # Optional: [fx, fy, cx, cy] per frame
                'confidence': conf_list,  # Optional: HxW confidence maps
                'points': pts_list,       # Optional: HxWx3 world-coord point maps
            },
            'global': {}
        }
```

### Step 4: Create the YAML config

Create `configs/methods/<name>.yaml` with `model`, `env`, and any `_`-prefixed kwargs.

### Step 5 (optional): Add an installation script

Place an idempotent install script at `envs/install_<name>.sh`.

### Class naming convention

Module file names use snake_case. The class loader converts them to PascalCase and appends the suffix:

| Module file | Class name |
|-------------|------------|
| `methods/lingbot_map.py` | `LingbotMapMethod` |
| `datasets/seven_scenes.py` | `SevenScenesDataset` |
| `methods/my_new_method.py` | `MyNewMethodMethod` |

### Image resize

Methods declare an `area_budget` (and an `align` divisor) in their YAML config. `BSSLoader` scales each image down so `W * H <= area_budget`, with both dimensions snapped to multiples of `align`. Camera intrinsics are adjusted accordingly. Omit `area_budget` (or set it to `None`) to load images at native resolution.

| Mode | Behavior |
|------|----------|
| `none` | No resize (the default when `area_budget` is omitted) |
| `area_budget` | Uniform downscale so `W * H <= area_budget`; dimensions aligned to `align` |

If a method needs more complex preprocessing (letterbox, square crop, etc.), do it inside the method wrapper's `process_scene()` and return the corresponding adjusted intrinsics.

### Method subprocess dispatch

If a method config includes an `env` field, `run.py` does not run the method in-process. Instead, it spawns:

```bash
conda run -n {env} python run_worker.py --config ... --method ... --dataset ...
```

This isolates each method's Python and CUDA dependencies.

---

## Adding a New Dataset

Create `datasets/<name>.py`. The class name follows the same snake_case-to-PascalCase convention with a `Dataset` suffix.

```python
from benchmark.dataset.base import BaseDataset


class MyDatasetDataset(BaseDataset):
    def __init__(self, raw_data_root, logger=None, **kwargs):
        super().__init__(raw_data_root, logger)

    def get_scenes(self):
        """Return a list of scene IDs (strings)."""
        ...

    def get_frame_list(self, scene):
        """Return a list of frame IDs (integers) for the given scene."""
        ...

    def load_frame_data(self, scene, frame_id):
        """Load data for a single frame.

        Required keys:
            'timestamp' (float): Frame timestamp.
            'rgb' (np.ndarray): HxWx3 uint8 RGB image.

        Optional keys:
            'depth' (np.ndarray): HxW float32 depth in meters.
            'pose' (np.ndarray): 4x4 C2W transformation matrix.
            'intrinsics' (np.ndarray): [fx, fy, cx, cy].
            'mask' (np.ndarray): HxW boolean mask.
        """
        ...

    def load_global_data(self, scene):
        """Optional: return global scene data (e.g., point cloud).

        Optional keys:
            'points' (np.ndarray): Nx3 or Nx6 (xyzrgb) point cloud.
        """
        return {}
```

### Custom saver

To define a custom save method for a non-standard data key, implement `__save_{key}_file__` on the dataset class:

```python
def __save_semantic_file__(self, key_dir, timestamp, data):
    # key_dir is the directory for this data type (e.g., output/semantic/)
    # timestamp is the canonical timestamp string
    # data is whatever load_frame_data returned under the 'semantic' key
    ...
```

### Point cloud evaluation

Datasets can provide a custom point cloud evaluation method:

```python
@staticmethod
def evaluate_pointcloud(gt_loader, pred_loader, logger, options=None):
    ...
```
