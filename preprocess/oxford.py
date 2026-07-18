#!/usr/bin/env python3
"""
process_final.py — Oxford Spires cam0 processing pipeline.

Per-frame outputs:
  images/{frame_idx:06d}.png   — undistorted PNG (fisheye balance=0.0)
  depth/{frame_idx:06d}.npy    — float32 (H, W) visibility-filtered depth in metres
                                 [omitted with --images_only]
  intrinsics.txt               — fx fy cx cy width height  (rectified pinhole)
  poses_c2w.txt                — 16 floats per line (4×4 C2W row-major)

Global ground-truth output:
  ground_truth.ply             — TLS points visible from ≥ 1 frame after
                                 visibility filtering (binary PLY, float32)
                                 [omitted with --images_only]

The ground-truth cloud is the union of per-frame visibility-filtered visible
points.  This is strictly tighter than raw frustum clipping: occluded points
are removed by the visibility kernel before accumulation, so the result
represents what the camera could actually see during the traversal.

Point cloud source: TLS ground-truth maps (mm accuracy, TLS world frame).
Pose source: processed/trajectory/gt-tum.txt (~20 Hz, TLS world frame).
Matching: nearest-neighbour, max_gap = 0.1 s.

Coordinate transform chain:
  TLS world ←T_WB(t_img)← gt-tum.txt
  Base frame ←inv(T_base_lidar)←
  LiDAR frame ←inv(T_cam0_lidar)←
  cam0 frame

  C2W = T_WB @ T_base_cam0
  T_cam0_base = T_cam0_lidar @ inv(T_base_lidar)
  T_base_cam0 = inv(T_cam0_base)

Usage:
  python process_final.py --sequence 2024-03-12-keble-college-02
  python process_final.py --sequence 2024-03-12-keble-college-02 --images_only
  python process_final.py  # all sequences
  python process_final.py --dataset_dir /path/to/dataset --output_dir /path/to/out
"""

import argparse
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
import yaml
from scipy.spatial.transform import Rotation
from tqdm import tqdm


# ── Optional CUDA visibility extension ────────────────────────────────────────
_vis_ext           = None
_HAS_VIS_EXT       = False
_VIS_EXT_ATTEMPTED = False


def ensure_visibility_extension() -> bool:
    """Load the optional CUDA visibility extension on first use."""
    global _vis_ext, _HAS_VIS_EXT, _VIS_EXT_ATTEMPTED

    if _VIS_EXT_ATTEMPTED:
        return _HAS_VIS_EXT

    _VIS_EXT_ATTEMPTED = True
    try:
        from torch.utils.cpp_extension import load as _ext_load
        _EXT_DIR = Path(__file__).parent / "points_visibility"
        _vis_ext = _ext_load(
            name="points_visibility",
            sources=[
                str(_EXT_DIR / "visibility.cpp"),
                str(_EXT_DIR / "visibility_kernel.cu"),
                str(_EXT_DIR / "frustum_cull.cu"),
            ],
            build_directory=str(_EXT_DIR),
            verbose=True,
        )
        _HAS_VIS_EXT = torch.cuda.is_available()
    except Exception as _e:
        print(f"[visibility] CUDA extension unavailable ({_e}); NumPy fallback active.")
        _vis_ext = None
        _HAS_VIS_EXT = False

    return _HAS_VIS_EXT


# ── Default paths ──────────────────────────────────────────────────────────────
DATASET_DIR  = Path("/data3/gaojian/oxford_spires_dataset")
OUTPUT_DIR   = Path("/data0/gaojian/oxford_spires")
MAX_TIME_GAP = 0.1    # seconds
MAX_DEPTH    = 200.0   # metres — frustum far plane

# We only use these scenes for evaluation.
PROCESS_SCENE = [
    "2024-03-12-keble-college-02",
    "2024-03-12-keble-college-03",
    "2024-03-12-keble-college-04",
    "2024-03-12-keble-college-05",
    "2024-03-13-observatory-quarter-01",
    "2024-03-13-observatory-quarter-02",
    "2024-03-18-christ-church-02",
    "2024-03-18-christ-church-03",
    "2024-03-18-christ-church-05",
    "2024-05-20-bodleian-library-02",
]

# TLC points are not aligned with the images. --image_only is required.
# PROCESS_SCENE = [
#     "2024-03-18-christ-church-01",
#     "2024-03-14-blenheim-palace-01",
#     "2024-03-14-blenheim-palace-02",
#     "2024-03-14-blenheim-palace-05",
# ]

# Other scens has no gt traj.

# ── Scene keyword extraction ───────────────────────────────────────────────────

def extract_scene_keyword(seq_name: str) -> str:
    """
    Strip leading date (YYYY-MM-DD-) and trailing sequence number (-N or -NN).
    '2024-03-12-keble-college-02' → 'keble-college'
    """
    name = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", seq_name)
    name = re.sub(r"-\d{1,2}$", "", name)
    return name


def get_tls_pcd_path(gt_map_dir: Path, scene_kw: str) -> Path | None:
    """Return the TLS merged-cloud PCD path for *scene_kw*, or None if absent."""
    scene_dir = gt_map_dir / scene_kw
    if not scene_dir.is_dir():
        return None
    candidates = [
        scene_dir / "merged-cloud-1cm.pcd",
        scene_dir / "merged-cloud-5cm.pcd",   # observatory-quarter uses 5 cm
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ── Calibration ───────────────────────────────────────────────────────────────

def load_cam0_intrinsics(cam0_yaml: Path):
    """Return K (3×3), dist (4,), W, H from cam0.yaml."""
    with open(cam0_yaml) as f:
        d = yaml.safe_load(f)
    md = d["camera_matrix"]["data"]
    K  = np.array([[md[0], 0.0, md[2]],
                   [0.0, md[4], md[5]],
                   [0.0,  0.0,  1.0 ]], dtype=np.float64)
    dist = np.array(d["distortion_coefficients"]["data"], dtype=np.float64)
    W, H = int(d["image_width"]), int(d["image_height"])
    return K, dist, W, H


def load_T_cam0_lidar(cam_lidar_yaml: Path) -> np.ndarray:
    """4×4 T_cam0_lidar: lidar frame → cam0 frame."""
    with open(cam_lidar_yaml) as f:
        d = yaml.safe_load(f)
    return np.array(d["cam0"]["T_cam_lidar"], dtype=np.float64).reshape(4, 4)


def build_T_base_lidar() -> np.ndarray:
    """
    Fixed hardware calibration (sensor.yaml):
      t=[0,0,0.124 m],  q_xyzw=[0,0,1,0]  →  180° Z rotation + 124 mm Z lift.
    Lidar frame → base frame.
    """
    T = np.eye(4)
    T[0, 0] = -1.0
    T[1, 1] = -1.0
    T[2, 3] =  0.124
    return T


# ── Fisheye rectification ─────────────────────────────────────────────────────

class ImageRectifier:
    """Fisheye undistortion (OPENCV_FISHEYE / equidistant, balance=0.0)."""

    def __init__(self, K: np.ndarray, dist: np.ndarray, W: int, H: int):
        self.new_K = K.copy()
        self.new_K[0, 2] = W / 2.0
        self.new_K[1, 2] = H / 2.0
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, dist, np.eye(3), self.new_K, (W, H), cv2.CV_32FC1
        )
        self.W, self.H = W, H

    def rectify(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.map1, self.map2,
                         cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def valid_mask(self) -> np.ndarray:
        """Return (H, W) bool mask — True where rectification maps to a valid
        source pixel.  Pixels outside the fisheye projection (black border after
        undistortion) are False; depth values should be zeroed there."""
        ones = np.ones((self.H, self.W), dtype=np.uint8) * 255
        rect = cv2.remap(ones, self.map1, self.map2,
                         cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return rect > 0

    @property
    def fx(self): return float(self.new_K[0, 0])
    @property
    def fy(self): return float(self.new_K[1, 1])
    @property
    def cx(self): return float(self.new_K[0, 2])
    @property
    def cy(self): return float(self.new_K[1, 2])


# ── Trajectory ────────────────────────────────────────────────────────────────

def load_tum(tum_file: Path):
    """Parse a TUM-format trajectory file.

    Returns:
        timestamps : (N,) float64
        lines      : list[str]  — original text lines (no trailing newline)
    """
    timestamps, lines = [], []
    with open(tum_file) as f:
        for line in f:
            s = line.rstrip("\n")
            if not s or s.lstrip().startswith("#"):
                continue
            timestamps.append(float(s.split()[0]))
            lines.append(s)
    return np.array(timestamps, dtype=np.float64), lines


def tum_line_to_T_WB(line: str) -> np.ndarray:
    """'ts tx ty tz qx qy qz qw' → 4×4 T_world_base."""
    p  = line.split()
    T  = np.eye(4)
    T[:3, :3] = Rotation.from_quat([float(p[4]), float(p[5]),
                                     float(p[6]), float(p[7])]).as_matrix()
    T[:3, 3]  = [float(p[1]), float(p[2]), float(p[3])]
    return T


def find_nearest(target: float, timestamps: np.ndarray, max_gap: float):
    """Nearest-neighbour search.

    Returns (index, diff) if diff <= max_gap, else (None, diff).
    """
    diffs = np.abs(timestamps - target)
    idx   = int(np.argmin(diffs))
    diff  = float(diffs[idx])
    return (idx, diff) if diff <= max_gap else (None, diff)


# ── TLS point cloud ───────────────────────────────────────────────────────────

def load_tls_cloud(pcd_path: Path) -> tuple:
    """Load full-resolution TLS PCD.

    Returns:
        pts    : (N, 3) float32 — XYZ coordinates
        colors : (N, 3) float32 in [0, 1] — RGB, or None if the cloud has no colors
    """
    print(f"  Loading TLS PCD: {pcd_path} ...", flush=True)
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    n   = len(pcd.points)
    print(f"  TLS cloud: {n:,} points (full resolution)", flush=True)
    pts    = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    return pts, colors


# ── Per-frame projection ──────────────────────────────────────────────────────

def frustum_cull_and_project(
    pts_world: np.ndarray,
    C2W: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    W: int, H: int,
    near_plane: float = 0.1,
    far_plane: float = 100.0,
    pts_gpu=None,
) -> tuple:
    """
    Frustum cull the TLS cloud and build a depth map for one frame.

    GPU path  (pts_gpu is not None and _HAS_VIS_EXT):
        Calls the compiled frustum_cull CUDA kernel.

    NumPy fallback:
        Float32 matrix multiply, boolean FOV mask, argsort-based depth painting.

    Returns:
        depth_map  : (H, W) float32 — Z-depth in metres, 0 = no data
        winner_map : (H, W) int32   — original index into pts_world of the
                     nearest in-frustum point; -1 where no point projects.
    """
    empty = (np.zeros((H, W), dtype=np.float32),
             np.full((H, W), -1, dtype=np.int32))

    # ── GPU path ─────────────────────────────────────────────────────────────
    if _HAS_VIS_EXT and pts_gpu is not None:
        T_cw = np.linalg.inv(C2W)
        R_t  = torch.from_numpy(T_cw[:3, :3].astype(np.float32)).cuda()
        t_t  = torch.from_numpy(T_cw[:3, 3] .astype(np.float32)).cuda()
        depth_t, winner_t = _vis_ext.frustum_cull(
            pts_gpu, R_t, t_t, fx, fy, cx, cy, W, H, near_plane, far_plane)
        return depth_t.cpu().numpy(), winner_t.cpu().numpy()

    # ── NumPy fallback ────────────────────────────────────────────────────────
    T_cw  = np.linalg.inv(C2W)
    R_cw  = T_cw[:3, :3].astype(np.float32)
    t_cw  = T_cw[:3, 3] .astype(np.float32)
    p_cam = (R_cw @ pts_world.T).T + t_cw      # (N, 3) float32

    z     = p_cam[:, 2]
    front = z > near_plane
    z_s   = np.where(front, z, 1.0)
    u_i   = (p_cam[:, 0] / z_s * fx + cx).astype(np.int32)
    v_i   = (p_cam[:, 1] / z_s * fy + cy).astype(np.int32)

    fov = front & (z < far_plane) & (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)
    if not fov.any():
        return empty

    u_fov    = u_i[fov]
    v_fov    = v_i[fov]
    z_fov    = z[fov].astype(np.float32)
    orig_idx = np.where(fov)[0].astype(np.int32)

    order      = np.argsort(-z_fov)
    depth_map  = np.zeros((H, W), dtype=np.float32)
    winner_map = np.full((H, W), -1, dtype=np.int32)
    depth_map [v_fov[order], u_fov[order]] = z_fov[order]
    winner_map[v_fov[order], u_fov[order]] = orig_idx[order]

    return depth_map, winner_map


# ── Visibility filter ─────────────────────────────────────────────────────────

def _apply_vis_filter_cuda(
    depth_map: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    threshold: float, radius: int,
) -> np.ndarray:
    """Thin wrapper around the compiled visibility2 CUDA kernel."""
    H, W = depth_map.shape
    depth_t  = torch.from_numpy(depth_map).float().cuda().contiguous()
    intr_t   = torch.tensor([fx, fy, cx, cy],
                             dtype=torch.float32).cuda().contiguous()
    output_t = depth_t.clone().contiguous()
    _vis_ext.visibility2(depth_t, intr_t, output_t, W, H,
                         float(threshold), int(radius))
    return output_t.cpu().numpy()


def _apply_vis_filter2_numpy(
    depth_map: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    threshold: float, radius: int,
) -> np.ndarray:
    """
    NumPy port of visibility_kernel2.

    For each non-zero pixel v at (row, col) with depth d:
      - Back-project to 3D:  v = ((col-cx)*d/fx, (row-cy)*d/fy, d)
      - Unit vector to camera:  v2 = -v / |v|
      - In each of 4 image quadrants find the neighbour n maximising
            dot(normalise(n_3d - v), v2)
      - If max_Q0 + max_Q1 + max_Q2 + max_Q3 >= threshold → occluded → 0.
    """
    H, W = depth_map.shape
    row_g = np.arange(H, dtype=np.float32)[:, None]
    col_g = np.arange(W, dtype=np.float32)[None, :]

    d     = depth_map
    valid = d > 0

    vx = (col_g - cx) * d / fx
    vy = (row_g - cy) * d / fy
    vz = d

    v_sq   = vx*vx + vy*vy + vz*vz
    v_n    = np.where(v_sq > 0, np.sqrt(v_sq), 1.0)
    v2x, v2y, v2z = -vx / v_n, -vy / v_n, -vz / v_n

    mq = [np.full((H, W), -1.0, dtype=np.float32) for _ in range(4)]

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            qs = []
            if di <= 0 and dj <= 0: qs.append(0)
            if di >= 0 and dj <= 0: qs.append(1)
            if di <= 0 and dj >= 0: qs.append(2)
            if di >= 0 and dj >= 0: qs.append(3)

            r_ss = max(0,  di);  r_se = H + min(0,  di)
            c_ss = max(0,  dj);  c_se = W + min(0,  dj)
            r_ds = max(0, -di);  r_de = H + min(0, -di)
            c_ds = max(0, -dj);  c_de = W + min(0, -dj)
            if r_se <= r_ss or c_se <= c_ss:
                continue
            nbr = np.zeros((H, W), dtype=np.float32)
            nbr[r_ds:r_de, c_ds:c_de] = d[r_ss:r_se, c_ss:c_se]

            nb_ok = nbr > 0
            nx = (col_g + dj - cx) * nbr / fx
            ny = (row_g + di - cy) * nbr / fy
            nz = nbr

            crx, cry, crz = nx - vx, ny - vy, nz - vz
            c_sq  = crx*crx + cry*cry + crz*crz
            c_ok  = nb_ok & (c_sq > 0)
            c_n   = np.where(c_ok, np.sqrt(c_sq), 1.0)
            dot   = (crx / c_n) * v2x + (cry / c_n) * v2y + (crz / c_n) * v2z
            dot_v = np.where(valid & c_ok, dot, -1.0)

            for q in qs:
                np.maximum(mq[q], dot_v, out=mq[q])

    total    = mq[0] + mq[1] + mq[2] + mq[3]
    occluded = valid & (total >= threshold)
    out = depth_map.copy()
    out[occluded] = 0.0
    return out


def apply_visibility_filter(
    depth_map: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    threshold: float = 7.0,
    radius: int = 11,
) -> np.ndarray:
    """
    Visibility filter (CMRNet / iralab visibility_kernel2).

    Removes depth pixels that are occluded by a foreground surface.
    Dispatches to the compiled CUDA extension when available; otherwise
    falls back to the vectorised NumPy implementation.

    Args:
        depth_map  : (H, W) float32, metres, 0 = no data
        fx,fy,cx,cy: rectified pinhole intrinsics
        threshold  : occlusion decision threshold (default 7.0)
        radius     : neighbourhood half-size in pixels (default 11)
    """
    if _HAS_VIS_EXT:
        return _apply_vis_filter_cuda(depth_map, fx, fy, cx, cy, threshold, radius)
    return _apply_vis_filter2_numpy(depth_map, fx, fy, cx, cy, threshold, radius)


# ── Depth pseudo-colour visualisation ────────────────────────────────────────

def depth_to_colormap(depth_map: np.ndarray, max_depth: float = 60.0) -> np.ndarray:
    """Convert a float32 depth map to a BGR pseudo-colour image (COLORMAP_TURBO).

    Invalid pixels (depth == 0) are rendered as black.

    Args:
        depth_map : (H, W) float32, metres, 0 = no data
        max_depth : depth value mapped to the far end of the colormap

    Returns:
        (H, W, 3) uint8 BGR image
    """
    valid = depth_map > 0
    norm  = np.zeros_like(depth_map, dtype=np.uint8)
    norm[valid] = np.clip(
        depth_map[valid] / max_depth * 255.0, 0.0, 255.0
    ).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)   # (H, W, 3)
    color[~valid] = 0                                      # black for no-data
    return color


# ── Debug depth overlay ───────────────────────────────────────────────────────

def make_depth_overlay(rect_img: np.ndarray, depth_map: np.ndarray,
                       max_depth: float = 50.0) -> np.ndarray:
    """Blend depth colormap with the rectified image.

    Valid depth pixels show the TURBO colormap; invalid pixels show the
    original image.
    """
    color  = depth_to_colormap(depth_map, max_depth)
    valid  = depth_map > 0
    result = rect_img.copy()
    result[valid] = color[valid]
    return result


# ── Per-scene processing ──────────────────────────────────────────────────────

def process_scene(
    seq_dir: Path, calib_dir: Path, gt_map_dir: Path, output_dir: Path,
    max_time_gap: float = MAX_TIME_GAP,
    max_frames: int = 0,
    debug: bool = False,
    debug_frames: int = 50,
    debug_stride: int = 10,
    images_only: bool = False,
) -> str | None:
    """
    Process one sequence.

    For every matched frame:
      1. Decode & rectify the cam0 image → save images/{:06d}.png
      2. Save the matched C2W pose for the rectified frame
      3. Unless `images_only`:
         - Frustum-cull TLS cloud → raw depth map
         - Visibility filter → filtered depth map → save depth/{:06d}.npy
         - Back-project surviving pixels → mark corresponding TLS points visible

    After all frames:
      4. Unless `images_only`: save ground_truth.ply — union of all
         per-frame visible TLS points

    Returns None on success, or an error/skip reason string.
    """
    seq_name  = seq_dir.name
    scene_kw  = extract_scene_keyword(seq_name)
    seq_short = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", seq_name)

    gt_tum = seq_dir / "processed" / "trajectory" / "gt-tum.txt"
    if not gt_tum.exists():
        return f"no gt-tum.txt at {gt_tum}"

    images_zip = seq_dir / "raw" / "images.zip"
    if not images_zip.exists():
        return f"no images.zip at {images_zip}"

    tls_path = None
    if not images_only:
        tls_path = get_tls_pcd_path(gt_map_dir, scene_kw)
        if tls_path is None:
            return f"no TLS PCD for scene '{scene_kw}'"

    # ── Calibration
    K, dist, W, H = load_cam0_intrinsics(calib_dir / "cam0.yaml")
    T_cam0_lidar  = load_T_cam0_lidar(calib_dir / "cam-lidar-imu.yaml")
    T_base_lidar  = build_T_base_lidar()
    T_cam0_base   = T_cam0_lidar @ np.linalg.inv(T_base_lidar)
    T_base_cam0   = np.linalg.inv(T_cam0_base)

    rectifier = ImageRectifier(K, dist, W, H)
    fx, fy, cx, cy = rectifier.fx, rectifier.fy, rectifier.cx, rectifier.cy
    print(f"  Rectified intrinsics: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")

    # ── Trajectory
    traj_ts, traj_lines = load_tum(gt_tum)
    print(f"  Trajectory: {len(traj_ts)} poses  "
          f"[{traj_ts[0]:.3f} → {traj_ts[-1]:.3f}]")

    pts_world = None
    pts_colors = None
    pts_gpu = None
    visible_mask = None
    N = 0

    if images_only:
        print("  Image-only mode: skipping TLS cloud, depth maps, and ground-truth export.")
    else:
        ensure_visibility_extension()
        pts_world, pts_colors = load_tls_cloud(tls_path)
        N = len(pts_world)
        visible_mask = np.zeros(N, dtype=bool)   # accumulates over all frames

        # Upload to GPU once (frustum_cull CUDA kernel)
        if _HAS_VIS_EXT:
            print("  Uploading TLS cloud to GPU ...", end=" ", flush=True)
            pts_gpu = torch.from_numpy(pts_world).cuda()
            print("done")

    # ── Output directories
    dst     = output_dir / seq_short
    out_img = dst / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    if not images_only:
        out_dep     = dst / "depth"
        out_dep_vis = dst / "depth_vis"
        for d in (out_dep, out_dep_vis):
            d.mkdir(parents=True, exist_ok=True)
        if debug:
            out_dbg = dst / "debug"
            out_dbg.mkdir(parents=True, exist_ok=True)

    # ── Per-frame loop
    pose_records = []
    frame_idx    = 0
    skip_pose    = 0
    skip_other   = 0

    with zipfile.ZipFile(images_zip, "r") as zf:
        cam0_names = sorted(
            n for n in zf.namelist()
            if n.startswith("cam0/") and n.lower().endswith(".jpg")
        )
        print(f"  cam0 images in zip: {len(cam0_names)}")

        if max_frames > 0:
            cam0_names = cam0_names[:max_frames]
            print(f"  max_frames={max_frames}: using first {len(cam0_names)} frames")

        if debug:
            cam0_names = cam0_names[::debug_stride][:debug_frames]
            print(f"  [debug] selected {len(cam0_names)} frames "
                  f"(stride={debug_stride}, max={debug_frames})")

        for name in tqdm(cam0_names, desc=seq_name, unit="frame"):
            stem  = Path(name).stem
            t_img = float(stem)

            t_idx, _ = find_nearest(t_img, traj_ts, max_time_gap)
            if t_idx is None:
                skip_pose += 1
                continue

            # Decode & rectify
            raw_bytes = zf.read(name)
            arr       = np.frombuffer(raw_bytes, dtype=np.uint8)
            img       = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                skip_other += 1
                continue
            rect = rectifier.rectify(img)

            # C2W pose
            T_WB = tum_line_to_T_WB(traj_lines[t_idx])
            C2W  = T_WB @ T_base_cam0

            if not images_only:
                # Frustum cull → raw depth map
                depth_map, winner_map = frustum_cull_and_project(
                    pts_world, C2W, fx, fy, cx, cy, W, H,
                    near_plane=0.1, far_plane=MAX_DEPTH,
                    pts_gpu=pts_gpu,
                )

                # Visibility filter → remove occluded points
                vis_radius = 20
                filtered_depth = apply_visibility_filter(
                    depth_map, fx, fy, cx, cy, threshold=2.5, radius=vis_radius
                )
                # Mask image border — the filter neighbourhood is incomplete within
                # `vis_radius` pixels of each edge, so occluded points there are never
                # detected and leave a spurious depth strip.
                filtered_depth[:vis_radius, :]  = 0
                filtered_depth[-vis_radius:, :] = 0
                filtered_depth[:, :vis_radius]  = 0
                filtered_depth[:, -vis_radius:] = 0

                # Accumulate visible TLS point indices into global mask
                valid_px = filtered_depth > 0
                vis_idx  = winner_map[valid_px]
                vis_idx  = vis_idx[vis_idx >= 0]
                visible_mask[vis_idx] = True

            # Save per-frame outputs
            cv2.imwrite(str(out_img / f"{frame_idx:06d}.png"), rect)
            if not images_only:
                np.save(str(out_dep / f"{frame_idx:06d}.npy"), filtered_depth)
                cv2.imwrite(str(out_dep_vis / f"{frame_idx:06d}.jpg"),
                            depth_to_colormap(filtered_depth, max_depth=MAX_DEPTH),
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

                if debug:
                    overlay = make_depth_overlay(rect, filtered_depth,
                                                 max_depth=MAX_DEPTH)
                    cv2.imwrite(str(out_dbg / f"{frame_idx:06d}.jpg"), overlay,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

            pose_records.append((frame_idx, stem, C2W))
            frame_idx += 1

    print(f"  Saved {len(pose_records)} frames  |  "
          f"skipped: {skip_pose} (no pose)  {skip_other} (decode error)")

    if not pose_records:
        return "no matched frames"

    if not images_only:
        # ── Ground-truth point cloud: union of all per-frame visibility-filtered pts
        n_vis, n_tot = int(visible_mask.sum()), N
        print(f"  Ground-truth cloud: {n_vis:,} / {n_tot:,} points "
              f"({100.0 * n_vis / max(n_tot, 1):.1f}%) visible after vis-filter")

        vis_pts    = pts_world[visible_mask]
        vis_colors = pts_colors[visible_mask] if pts_colors is not None else None

        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(vis_pts.astype(np.float64))
        if vis_colors is not None:
            pcd_out.colors = o3d.utility.Vector3dVector(vis_colors.astype(np.float64))
        gt_ply = dst / "ground_truth.ply"
        o3d.io.write_point_cloud(str(gt_ply), pcd_out,
                                 write_ascii=False, compressed=False)
        print(f"  Saved ground_truth.ply")

    # ── poses_c2w.txt: 16 floats per line (4×4 C2W row-major)
    with open(dst / "poses_c2w.txt", "w") as f:
        for _, _, C2W in pose_records:
            vals = " ".join(f"{v:.10f}" for v in C2W.flatten())
            f.write(f"{vals}\n")

    # ── intrinsics.txt: fx fy cx cy width height
    with open(dst / "intrinsics.txt", "w") as f:
        f.write(f"{fx:.10f} {fy:.10f} {cx:.10f} {cy:.10f} {W} {H}\n")

    return None


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Oxford Spires cam0 pipeline: rectify images, optionally compute "
            "visibility-filtered depth maps, and build a per-sequence ground-truth "
            "point cloud."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python process_final.py --sequence 2024-03-12-keble-college-02
  python process_final.py --sequence 2024-03-12-keble-college-02 --images_only
  python process_final.py
  python process_final.py --dataset_dir /data3/gaojian/oxford_spires_dataset \
                           --output_dir /data3/gaojian/output
""",
    )
    parser.add_argument(
        "--dataset_dir", default=str(DATASET_DIR),
        help="Root dataset directory (contains sequences/, calibration/, ground_truth_map/)",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help="Output root directory",
    )
    parser.add_argument(
        "--sequence", default=None,
        help="Process a single sequence (e.g. 2024-03-12-keble-college-02). "
             "Default: all sequences.",
    )
    parser.add_argument(
        "--max_time_gap", type=float, default=MAX_TIME_GAP,
        help=f"Max image↔pose timestamp gap in seconds (default: {MAX_TIME_GAP})",
    )
    parser.add_argument(
        "--max_frames", type=int, default=3840,
        help="Only process the first N frames per sequence (0 = all frames).",
    )
    parser.add_argument(
        "--images_only", action="store_true",
        help="Skip TLS point-cloud/depth processing and only save rectified images, poses_c2w.txt, and intrinsics.txt.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode: process a small subset of frames and save depth-overlay JPEGs.",
    )
    parser.add_argument(
        "--debug_frames", type=int, default=50,
        help="[debug] Number of frames to process (default: 50).",
    )
    parser.add_argument(
        "--debug_stride", type=int, default=10,
        help="[debug] Stride for frame selection (default: 10).",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir)
    calib_dir   = dataset_dir / "calibration"
    seqs_dir    = dataset_dir / "sequences"
    gt_map_dir  = dataset_dir / "ground_truth_map"
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("cam0.yaml", "cam-lidar-imu.yaml"):
        p = calib_dir / fname
        if not p.exists():
            raise FileNotFoundError(f"Missing calibration file: {p}")

    if args.sequence:
        seq_dirs = [seqs_dir / args.sequence]
    else:
        seq_dirs = sorted(d for d in seqs_dir.iterdir() if d.is_dir())

    skipped = []
    for seq_dir in seq_dirs:
        if not seq_dir.is_dir():
            print(f"[SKIP] {seq_dir}: not a directory")
            continue
        if seq_dir.name not in PROCESS_SCENE:
            print(f"[SKIP] {seq_dir.name}: not in PROCESS_SCENE list")
            continue

        print(f"\n{'='*60}")
        print(f"Sequence: {seq_dir.name}")
        print(f"{'='*60}")

        err = process_scene(
            seq_dir, calib_dir, gt_map_dir, output_dir,
            max_time_gap=args.max_time_gap,
            max_frames=args.max_frames,
            debug=args.debug,
            debug_frames=args.debug_frames,
            debug_stride=args.debug_stride,
            images_only=args.images_only,
        )
        if err:
            print(f"  [SKIP] {err}")
            skipped.append(f"{seq_dir.name}: {err}")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for s in skipped:
            print(f"  {s}")
    else:
        print("All sequences processed successfully.")
    print()


if __name__ == "__main__":
    main()
