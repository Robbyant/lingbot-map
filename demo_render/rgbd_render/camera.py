"""Camera representations, keyframe paths, and preset generators.

Core types:
- Camera: single camera state (eye, center, up, fov)
- CameraPath: keyframe-based path with interpolation, JSON-serializable

Preset generators (return CameraPath):
- make_follow_path(): smooth follow-camera from poses
- make_birdeye_path(): overhead view from point cloud extent
- make_static_path(): fixed camera from a pose
- make_pivot_path(): fixed eye, lookat follows scan direction
- build_camera_path(): segment-based multi-mode path
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

class Camera:
    """Single camera state in world coordinates."""

    __slots__ = ('eye', 'center', 'up', 'fov_deg')

    def __init__(self, eye: np.ndarray, center: np.ndarray,
                 up: np.ndarray, fov_deg: float = 60.0):
        self.eye = np.asarray(eye, dtype=np.float32)
        self.center = np.asarray(center, dtype=np.float32)
        self.up = np.asarray(up, dtype=np.float32)
        self.fov_deg = fov_deg

    def w2c(self) -> np.ndarray:
        """W2C (4x4) in OpenCV convention. Rows of R: [right, down, forward]."""
        return lookat(self.eye, self.center, self.up)

    @property
    def R_cw(self) -> np.ndarray:
        return self.w2c()[:3, :3]

    @property
    def t_cw(self) -> np.ndarray:
        return self.w2c()[:3, 3]

    def fov_intrinsics(self, render_w: int, render_h: int) -> Tuple[float, float, float, float]:
        """Derive pinhole intrinsics (fx, fy, cx, cy) from vertical FOV."""
        fov_rad = np.radians(self.fov_deg)
        fy = render_h / (2.0 * np.tan(fov_rad / 2.0))
        fx = fy
        return fx, fy, render_w / 2.0, render_h / 2.0

    def to_dict(self) -> dict:
        return {
            'eye': self.eye.tolist(),
            'center': self.center.tolist(),
            'up': self.up.tolist(),
            'fov': self.fov_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'Camera':
        return cls(d['eye'], d['center'], d['up'], d.get('fov', 60.0))

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s in self.__slots__:
            setattr(self, s, state[s])


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """W2C (4x4) in OpenCV convention. Camera +Z = forward (toward scene)."""
    fwd = target - eye
    fwd_n = np.linalg.norm(fwd)
    fwd = fwd / fwd_n if fwd_n > 1e-8 else np.array([0, 0, 1], dtype=np.float32)

    right = np.cross(up, fwd)
    right_n = np.linalg.norm(right)
    right = right / right_n if right_n > 1e-8 else np.array([1, 0, 0], dtype=np.float32)

    down = np.cross(right, fwd)

    R = np.stack([right, down, fwd], axis=0).astype(np.float32)
    t = (-R @ eye).astype(np.float32)
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C


def compute_global_up(c2w_poses: np.ndarray) -> np.ndarray:
    """Scene world-up from median of camera -Y axes."""
    up = np.median(-c2w_poses[:, :3, 1], axis=0)
    norm = np.linalg.norm(up)
    if norm < 1e-6:
        return np.array([0, 1, 0], dtype=np.float32)
    return (up / norm).astype(np.float32)


def compute_scene_scale(c2w_poses: np.ndarray) -> float:
    """Diagonal of the camera trajectory bounding box."""
    cam_pos = c2w_poses[:, :3, 3]
    return max(float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0))), 1e-3)


def compute_local_scale(c2w_poses: np.ndarray, n_frames: int = 100) -> float:
    """Diagonal of the first-N-frames trajectory bounding box."""
    n = min(n_frames, len(c2w_poses))
    cam_pos = c2w_poses[:n, :3, 3]
    return max(float(np.linalg.norm(cam_pos.max(0) - cam_pos.min(0))), 1e-3)


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------

def _smoothstep(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def _lerp_camera(a: Camera, b: Camera, t: float) -> Camera:
    """Linearly interpolate between two cameras."""
    eye = ((1 - t) * a.eye + t * b.eye).astype(np.float32)
    center = ((1 - t) * a.center + t * b.center).astype(np.float32)
    up_raw = (1 - t) * a.up + t * b.up
    up_norm = np.linalg.norm(up_raw)
    up = (up_raw / max(up_norm, 1e-6)).astype(np.float32)
    fov = (1 - t) * a.fov_deg + t * b.fov_deg
    return Camera(eye, center, up, fov)


# ---------------------------------------------------------------------------
# CameraPath: keyframe + interpolation + JSON
# ---------------------------------------------------------------------------

class CameraPath:
    """Keyframe-based camera path with interpolation.

    Presets (follow, birdeye) are just functions that generate CameraPath.

    JSON format:
        {
            "total_frames": 500,
            "interpolation": "smoothstep",
            "keyframes": [
                {"frame": 0, "eye": [...], "center": [...], "up": [...], "fov": 60},
                ...
            ]
        }
    """

    def __init__(self, keyframes: List[dict], total_frames: int,
                 interpolation: str = 'smoothstep'):
        """
        Args:
            keyframes: List of dicts with keys: frame, eye, center, up, fov.
                       Must be sorted by frame and cover frame 0 .. total_frames-1.
            total_frames: Total number of frames in the sequence.
            interpolation: 'linear' | 'smoothstep'
        """
        self.total_frames = total_frames
        self.interpolation = interpolation
        # Normalize keyframes: ensure numpy arrays
        self.keyframes = []
        for kf in sorted(keyframes, key=lambda k: k['frame']):
            self.keyframes.append({
                'frame': int(kf['frame']),
                'eye': np.asarray(kf['eye'], dtype=np.float32),
                'center': np.asarray(kf['center'], dtype=np.float32),
                'up': np.asarray(kf['up'], dtype=np.float32),
                'fov': float(kf.get('fov', 60.0)),
            })

    def get_camera(self, frame_idx: int) -> Camera:
        """Interpolate keyframes to get camera at any frame."""
        if not self.keyframes:
            return Camera([0, 0, 0], [0, 0, 1], [0, 1, 0])

        # Clamp
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))

        # Find surrounding keyframes
        right_idx = 0
        for i, kf in enumerate(self.keyframes):
            if kf['frame'] > frame_idx:
                right_idx = i
                break
        else:
            # frame_idx >= last keyframe
            kf = self.keyframes[-1]
            return Camera(kf['eye'], kf['center'], kf['up'], kf['fov'])

        if right_idx == 0:
            # frame_idx <= first keyframe
            kf = self.keyframes[0]
            return Camera(kf['eye'], kf['center'], kf['up'], kf['fov'])

        left_kf = self.keyframes[right_idx - 1]
        right_kf = self.keyframes[right_idx]

        # Compute interpolation parameter
        span = right_kf['frame'] - left_kf['frame']
        if span == 0:
            t = 0.0
        else:
            t = (frame_idx - left_kf['frame']) / span

        if self.interpolation == 'smoothstep':
            t = _smoothstep(t)

        cam_a = Camera(left_kf['eye'], left_kf['center'], left_kf['up'], left_kf['fov'])
        cam_b = Camera(right_kf['eye'], right_kf['center'], right_kf['up'], right_kf['fov'])
        return _lerp_camera(cam_a, cam_b, t)

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> Camera:
        return self.get_camera(idx)

    def save(self, path: str):
        """Save to JSON."""
        data = {
            'total_frames': self.total_frames,
            'interpolation': self.interpolation,
            'keyframes': [
                {
                    'frame': kf['frame'],
                    'eye': kf['eye'].tolist(),
                    'center': kf['center'].tolist(),
                    'up': kf['up'].tolist(),
                    'fov': kf['fov'],
                }
                for kf in self.keyframes
            ],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'CameraPath':
        """Load from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            keyframes=data['keyframes'],
            total_frames=data['total_frames'],
            interpolation=data.get('interpolation', 'smoothstep'),
        )


# ---------------------------------------------------------------------------
# Preset: follow camera
# ---------------------------------------------------------------------------

def _follow_camera_at(frame_idx: int, c2w_poses: np.ndarray, scene_scale: float,
                      smooth_window: int, back_offset: float, up_offset: float,
                      look_offset: float, follow_scale: Optional[float],
                      global_up: np.ndarray,
                      last_forward_h: Optional[np.ndarray] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute follow-camera eye/center/up for one frame.

    Returns (eye, center, up, forward_h) — forward_h is carried to next frame
    as fallback when the scanner points straight up/down.
    """
    S = len(c2w_poses)
    half = smooth_window // 2
    start = max(0, frame_idx - half)
    end = min(S, frame_idx + half + 1)

    cam_pos = c2w_poses[start:end, :3, 3].mean(axis=0)
    forward = c2w_poses[start:end, :3, 2].mean(axis=0)
    forward = forward / max(np.linalg.norm(forward), 1e-6)

    # --- eye: project forward to horizontal plane ---
    forward_h = forward - np.dot(forward, global_up) * global_up
    fh_norm = np.linalg.norm(forward_h)

    if fh_norm < 0.1:
        # Scanner nearly vertical — fallback to velocity direction
        vel_half = max(smooth_window // 2, 5)
        p0 = c2w_poses[max(0, frame_idx - vel_half), :3, 3]
        p1 = c2w_poses[min(S - 1, frame_idx + vel_half), :3, 3]
        vel = p1 - p0
        forward_h = vel - np.dot(vel, global_up) * global_up
        fh_norm = np.linalg.norm(forward_h)

    if fh_norm < 1e-4:
        # Still degenerate — use last valid forward_h
        if last_forward_h is not None:
            forward_h = last_forward_h
        else:
            forward_h = np.array([1, 0, 0], dtype=np.float32)
        fh_norm = np.linalg.norm(forward_h)

    forward_h = (forward_h / fh_norm).astype(np.float32)

    scale = follow_scale if follow_scale is not None else scene_scale
    eye = cam_pos - forward_h * scale * back_offset + global_up * scale * up_offset

    # --- center: keep original forward, preserves look-up/down ---
    center = cam_pos + forward * scale * look_offset

    return (eye.astype(np.float32), center.astype(np.float32),
            global_up.copy(), forward_h)


def make_follow_path(c2w_poses: np.ndarray, scene_scale: float,
                     smooth_window: int = 40,
                     back_offset: float = 0.4,
                     up_offset: float = 0.1,
                     look_offset: float = 0.5,
                     follow_scale: Optional[float] = None,
                     fov_deg: float = 60.0,
                     keyframe_interval: int = 1) -> CameraPath:
    """Generate follow-camera path as keyframes.

    Args:
        keyframe_interval: Generate a keyframe every N frames.
            1 = every frame (lossless), larger = sparser (interpolated).
    """
    S = len(c2w_poses)
    global_up = compute_global_up(c2w_poses)
    keyframes = []
    last_fh = None
    for fi in range(0, S, keyframe_interval):
        eye, center, up, last_fh = _follow_camera_at(
            fi, c2w_poses, scene_scale, smooth_window,
            back_offset, up_offset, look_offset, follow_scale,
            global_up, last_fh)
        keyframes.append({
            'frame': fi, 'eye': eye, 'center': center, 'up': up, 'fov': fov_deg,
        })
    # Ensure last frame is included
    if keyframes[-1]['frame'] != S - 1:
        eye, center, up, last_fh = _follow_camera_at(
            S - 1, c2w_poses, scene_scale, smooth_window,
            back_offset, up_offset, look_offset, follow_scale,
            global_up, last_fh)
        keyframes.append({
            'frame': S - 1, 'eye': eye, 'center': center, 'up': up, 'fov': fov_deg,
        })

    return CameraPath(keyframes, total_frames=S,
                      interpolation='linear' if keyframe_interval == 1 else 'smoothstep')


# ---------------------------------------------------------------------------
# Preset: bird's-eye camera
# ---------------------------------------------------------------------------

def _compute_birdeye_camera(
    c2w_poses: np.ndarray,
    sorted_xyz: np.ndarray,
    sorted_frames: Optional[np.ndarray],
    scene_scale: float,
    render_w: int, render_h: int,
    fov_deg: float,
    reveal_height_mult: float = 2.0,
    max_frame: Optional[int] = None,
) -> Camera:
    """Compute a single bird's-eye camera from trajectory + point cloud."""
    scene_center = c2w_poses[:, :3, 3].mean(axis=0).astype(np.float32)
    up_global = compute_global_up(c2w_poses)

    traj_pts = c2w_poses[:, :3, 3]
    traj_vec = traj_pts[-1] - traj_pts[0]
    traj_vec_h = traj_vec - np.dot(traj_vec, up_global) * up_global
    traj_vec_h_norm = np.linalg.norm(traj_vec_h)
    traj_dir = (traj_vec_h / traj_vec_h_norm).astype(np.float32) if traj_vec_h_norm > 1e-6 \
        else np.array([1, 0, 0], dtype=np.float32)

    is_portrait = render_h > render_w

    if is_portrait:
        end_up = traj_dir
        cam_right = np.cross(-up_global, traj_dir)
        cam_right_n = np.linalg.norm(cam_right)
        cam_right = (cam_right / cam_right_n).astype(np.float32) if cam_right_n > 1e-6 \
            else np.array([1, 0, 0], dtype=np.float32)
    else:
        end_up = np.cross(up_global, traj_dir)
        end_up_n = np.linalg.norm(end_up)
        end_up = (end_up / end_up_n).astype(np.float32) if end_up_n > 1e-6 \
            else np.array([1, 0, 0], dtype=np.float32)

    # Select points to use for height computation
    use_xyz = sorted_xyz
    if max_frame is not None and sorted_frames is not None and len(sorted_xyz) > 0:
        mask = sorted_frames <= max_frame
        if mask.sum() > 100:
            use_xyz = sorted_xyz[mask]

    height = scene_scale * reveal_height_mult
    margin = 1.15

    if len(use_xyz) > 0:
        cam_forward = -up_global
        cam_up = end_up
        cam_right = np.cross(cam_forward, cam_up)
        cam_right = (cam_right / max(np.linalg.norm(cam_right), 1e-6)).astype(np.float32)

        pts_centered = use_xyz - scene_center
        proj_right = np.dot(pts_centered, cam_right)
        proj_up = np.dot(pts_centered, cam_up)

        r_min, r_max = float(np.percentile(proj_right, 1)), float(np.percentile(proj_right, 99))
        u_min, u_max = float(np.percentile(proj_up, 1)), float(np.percentile(proj_up, 99))

        scene_center = scene_center + ((r_min + r_max) / 2) * cam_right + ((u_min + u_max) / 2) * cam_up

        half_w = (r_max - r_min) / 2
        half_h = (u_max - u_min) / 2
        fov_rad = np.radians(fov_deg)
        aspect = render_w / render_h
        h_vert = half_h / max(np.tan(fov_rad / 2), 1e-6)
        h_horiz = half_w / max(aspect * np.tan(fov_rad / 2), 1e-6)
        height = max(h_vert, h_horiz) * margin

    tilt = 0.1 if max_frame is None else 0.15
    birdeye_eye = (scene_center + up_global * height - end_up * scene_scale * tilt).astype(np.float32)

    return Camera(birdeye_eye, scene_center.copy(), end_up, fov_deg)


def make_birdeye_path(c2w_poses: np.ndarray,
                      sorted_xyz: np.ndarray,
                      sorted_frames: Optional[np.ndarray],
                      scene_scale: float,
                      total_frames: int,
                      render_w: int, render_h: int,
                      fov_deg: float = 60.0,
                      reveal_height_mult: float = 2.0) -> CameraPath:
    """Generate a static bird's-eye CameraPath (single keyframe)."""
    cam = _compute_birdeye_camera(
        c2w_poses, sorted_xyz, sorted_frames, scene_scale,
        render_w, render_h, fov_deg, reveal_height_mult)

    keyframes = [{'frame': 0, **cam.to_dict()},
                 {'frame': total_frames - 1, **cam.to_dict()}]
    return CameraPath(keyframes, total_frames, interpolation='linear')



# ---------------------------------------------------------------------------
# Preset: static camera
# ---------------------------------------------------------------------------

def _static_camera(c2w_poses: np.ndarray, start_frame: int,
                   eye: Optional[List[float]] = None,
                   lookat_pt: Optional[List[float]] = None,
                   fov_deg: float = 60.0) -> Camera:
    """Compute a static camera. Defaults from start frame pose if not given."""
    c2w = c2w_poses[start_frame]
    cam_pos = c2w[:3, 3].astype(np.float32)
    cam_fwd = c2w[:3, 2].astype(np.float32)
    cam_fwd = cam_fwd / max(np.linalg.norm(cam_fwd), 1e-6)
    cam_up = (-c2w[:3, 1]).astype(np.float32)
    cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-6)

    eye_pt = np.array(eye, dtype=np.float32) if eye else cam_pos
    center_pt = np.array(lookat_pt, dtype=np.float32) if lookat_pt else cam_pos + cam_fwd
    return Camera(eye_pt, center_pt, cam_up, fov_deg)


def make_static_path(c2w_poses: np.ndarray, start_frame: int, end_frame: int,
                     eye: Optional[List[float]] = None,
                     lookat_pt: Optional[List[float]] = None,
                     fov_deg: float = 60.0) -> CameraPath:
    """Static camera: fixed eye and lookat for the entire segment."""
    cam = _static_camera(c2w_poses, start_frame, eye, lookat_pt, fov_deg)
    total = end_frame - start_frame
    keyframes = [
        {'frame': 0, **cam.to_dict()},
        {'frame': total - 1, **cam.to_dict()},
    ]
    return CameraPath(keyframes, total, interpolation='linear')


# ---------------------------------------------------------------------------
# Preset: pivot camera
# ---------------------------------------------------------------------------

def make_pivot_path(c2w_poses: np.ndarray, start_frame: int, end_frame: int,
                    eye: Optional[List[float]] = None,
                    fov_deg: float = 60.0,
                    smooth_window: int = 20) -> CameraPath:
    """Pivot camera: fixed eye, lookat follows scan trajectory direction."""
    c2w = c2w_poses[start_frame]
    cam_pos = c2w[:3, 3].astype(np.float32)
    cam_up = (-c2w[:3, 1]).astype(np.float32)
    cam_up = cam_up / max(np.linalg.norm(cam_up), 1e-6)

    eye_pt = np.array(eye, dtype=np.float32) if eye else cam_pos
    total = end_frame - start_frame

    keyframes = []
    for fi in range(total):
        abs_fi = start_frame + fi
        half = smooth_window // 2
        s = max(0, abs_fi - half)
        e = min(len(c2w_poses), abs_fi + half + 1)
        # Lookat: smoothed scan position ahead of current
        look_pos = c2w_poses[s:e, :3, 3].mean(axis=0)
        fwd = c2w_poses[s:e, :3, 2].mean(axis=0)
        fwd = fwd / max(np.linalg.norm(fwd), 1e-6)
        center = (look_pos + fwd * 0.5).astype(np.float32)
        keyframes.append({
            'frame': fi, 'eye': eye_pt, 'center': center,
            'up': cam_up, 'fov': fov_deg,
        })

    return CameraPath(keyframes, total, interpolation='linear')


# ---------------------------------------------------------------------------
# Segment-based camera path builder
# ---------------------------------------------------------------------------

def build_camera_path(cam_config, scene) -> CameraPath:
    """Build a CameraPath from CameraConfig segments.

    Args:
        cam_config: CameraConfig with fov, transition, segments
        scene: Scene with c2w_poses, scene_scale, sorted_xyz, etc.

    If segments is empty, creates a default full-range follow path.
    """
    from .config import CameraSegment

    segments = cam_config.segments
    total_frames = scene.num_frames
    fov = cam_config.fov
    transition = cam_config.transition

    # Default: single follow segment
    if not segments:
        segments = [CameraSegment(mode='follow', frames=[0, -1])]

    # Resolve -1 → last frame
    resolved = []
    for seg in segments:
        s = seg.frames[0] if len(seg.frames) > 0 else 0
        e = seg.frames[1] if len(seg.frames) > 1 else -1
        if s < 0:
            s = max(0, total_frames + s)
        if e < 0:
            e = total_frames
        resolved.append((s, e, seg))

    # Build per-segment camera paths
    seg_paths = []
    for s, e, seg in resolved:
        if e <= s:
            continue

        if seg.mode == 'follow':
            follow_scale = compute_local_scale(
                scene.c2w_poses, seg.scale_frames) \
                if 0 < seg.scale_frames < total_frames else None
            path = make_follow_path(
                scene.c2w_poses, scene.scene_scale,
                smooth_window=seg.smooth_window,
                back_offset=seg.back_offset,
                up_offset=seg.up_offset,
                look_offset=seg.look_offset,
                follow_scale=follow_scale,
                fov_deg=fov)
            seg_paths.append((s, e, path))

        elif seg.mode == 'birdeye':
            path = make_birdeye_path(
                scene.c2w_poses, scene.sorted_xyz,
                scene.sorted_frames, scene.scene_scale,
                total_frames,
                scene.intrinsics[0][0, 2].item() * 2 if scene.intrinsics is not None else 1920,
                scene.intrinsics[0][1, 2].item() * 2 if scene.intrinsics is not None else 1080,
                fov, seg.reveal_height_mult)
            seg_paths.append((s, e, path))

        elif seg.mode == 'static':
            path = make_static_path(
                scene.c2w_poses, s, e, seg.eye, seg.lookat, fov)
            seg_paths.append((s, e, path))

        elif seg.mode == 'pivot':
            path = make_pivot_path(
                scene.c2w_poses, s, e, seg.eye, fov,
                seg.smooth_window)
            seg_paths.append((s, e, path))

    if not seg_paths:
        # Fallback
        return make_follow_path(
            scene.c2w_poses, scene.scene_scale, fov_deg=fov)

    # Single segment: extract frames directly from its path
    if len(seg_paths) == 1:
        s, e, path = seg_paths[0]
        keyframes = []
        for fi in range(total_frames):
            cam = path.get_camera(fi)
            keyframes.append({'frame': fi, **cam.to_dict()})
        return CameraPath(keyframes, total_frames, interpolation='linear')

    # Multiple segments: combine with transitions
    # For each frame, find which segment(s) it belongs to and blend
    keyframes = []
    half_t = transition // 2

    for fi in range(total_frames):
        # Find active segment
        active = None
        for idx, (s, e, path) in enumerate(seg_paths):
            if s <= fi < e:
                active = (idx, s, e, path)
                break

        if active is None:
            # Frame not covered by any segment — use nearest
            best_dist = float('inf')
            for idx, (s, e, path) in enumerate(seg_paths):
                dist = min(abs(fi - s), abs(fi - e))
                if dist < best_dist:
                    best_dist = dist
                    active = (idx, s, e, path)

        idx, s, e, path = active

        # Check if in transition zone with previous or next segment
        cam = path.get_camera(fi)

        # Transition with previous segment
        if idx > 0:
            ps, pe, ppath = seg_paths[idx - 1][:3]
            if fi < s + half_t and fi >= pe - half_t:
                # In transition zone
                t_start = pe - half_t
                t_end = s + half_t
                t_len = t_end - t_start
                if t_len > 0:
                    blend = _smoothstep((fi - t_start) / t_len)
                    prev_cam = ppath.get_camera(fi)
                    cam = _lerp_camera(prev_cam, cam, blend)

        # Transition with next segment
        if idx < len(seg_paths) - 1:
            ns, ne, npath = seg_paths[idx + 1][:3]
            if fi >= e - half_t and fi < ns + half_t:
                t_start = e - half_t
                t_end = ns + half_t
                t_len = t_end - t_start
                if t_len > 0:
                    blend = _smoothstep((fi - t_start) / t_len)
                    next_cam = npath.get_camera(fi)
                    cam = _lerp_camera(cam, next_cam, blend)

        keyframes.append({'frame': fi, **cam.to_dict()})

    return CameraPath(keyframes, total_frames, interpolation='linear')
