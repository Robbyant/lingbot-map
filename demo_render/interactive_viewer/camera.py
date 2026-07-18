"""Orbit camera controller and keyframe animator for interactive viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _lookat_w2c(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """W2C (4x4) in OpenCV convention. Camera +Z = forward."""
    fwd = _normalize(center - eye)
    right = _normalize(np.cross(up, fwd))
    down = np.cross(right, fwd)
    R = np.stack([right, down, fwd], axis=0).astype(np.float32)
    t = (-R @ eye).astype(np.float32)
    W2C = np.eye(4, dtype=np.float32)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C


class OrbitCamera:
    """Interactive orbit camera with WASD FPS-style movement."""

    def __init__(
        self,
        eye: np.ndarray,
        look_at: np.ndarray,
        up: np.ndarray = None,
        fov_deg: float = 60.0,
    ):
        self.eye = np.asarray(eye, dtype=np.float32).copy()
        self.look_at = np.asarray(look_at, dtype=np.float32).copy()
        self.up = np.asarray(
            up if up is not None else [0, -1, 0], dtype=np.float32
        ).copy()
        self.fov_deg = fov_deg

    # ----- Derived vectors -----

    @property
    def forward(self) -> np.ndarray:
        return _normalize(self.look_at - self.eye)

    @property
    def right(self) -> np.ndarray:
        return _normalize(np.cross(self.up, self.forward))

    @property
    def camera_up(self) -> np.ndarray:
        """Actual up vector (orthogonal to forward and right)."""
        return _normalize(np.cross(self.right, self.forward))

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.look_at - self.eye))

    # ----- Mouse interactions -----

    def orbit(self, dx: float, dy: float, sensitivity: float = 0.005):
        """Rotate around look_at point. dx = horizontal, dy = vertical."""
        offset = self.eye - self.look_at

        # Horizontal rotation (around world up)
        angle_h = -dx * sensitivity
        cos_h, sin_h = np.cos(angle_h), np.sin(angle_h)
        up = _normalize(self.up)
        # Rodrigues rotation around up axis
        offset = (
            offset * cos_h
            + np.cross(up, offset) * sin_h
            + up * np.dot(up, offset) * (1 - cos_h)
        )

        # Vertical rotation (around right axis)
        angle_v = -dy * sensitivity
        right = _normalize(np.cross(up, _normalize(offset)))
        cos_v, sin_v = np.cos(angle_v), np.sin(angle_v)
        offset = (
            offset * cos_v
            + np.cross(right, offset) * sin_v
            + right * np.dot(right, offset) * (1 - cos_v)
        )

        self.eye = (self.look_at + offset).astype(np.float32)

    def pan(self, dx: float, dy: float, sensitivity: float = 0.002):
        """Pan camera in screen plane."""
        dist = self.distance
        right = self.right
        cam_up = self.camera_up
        delta = (-right * dx + cam_up * dy) * sensitivity * dist
        self.eye += delta
        self.look_at += delta

    def zoom(self, delta: float, factor: float = 0.1):
        """Zoom by moving eye along view direction."""
        dist = self.distance
        move = self.forward * delta * factor * dist
        new_eye = self.eye + move
        # Prevent flipping through look_at
        if np.dot(self.look_at - new_eye, self.forward) > 0.01 * dist:
            self.eye = new_eye.astype(np.float32)

    # ----- WASD movement -----

    def move_forward(self, speed: float):
        delta = self.forward * speed
        self.eye += delta
        self.look_at += delta

    def move_right(self, speed: float):
        delta = self.right * speed
        self.eye += delta
        self.look_at += delta

    def move_up(self, speed: float):
        delta = self.camera_up * speed
        self.eye += delta
        self.look_at += delta

    # ----- Export -----

    def get_w2c(self) -> np.ndarray:
        return _lookat_w2c(self.eye, self.look_at, self.up)

    def fov_intrinsics(self, w: int, h: int) -> Tuple[float, float, float, float]:
        fov_rad = np.radians(self.fov_deg)
        fy = h / (2.0 * np.tan(fov_rad / 2.0))
        return fy, fy, w / 2.0, h / 2.0

    def to_dict(self) -> dict:
        return {
            "eye": self.eye.tolist(),
            "look_at": self.look_at.tolist(),
            "up": self.up.tolist(),
            "fov_deg": self.fov_deg,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OrbitCamera":
        return cls(d["eye"], d["look_at"], d.get("up"), d.get("fov_deg", 60.0))

    def copy(self) -> "OrbitCamera":
        return OrbitCamera(
            self.eye.copy(), self.look_at.copy(), self.up.copy(), self.fov_deg
        )


# ---------------------------------------------------------------------------
# Keyframe animator
# ---------------------------------------------------------------------------


@dataclass
class Keyframe:
    eye: np.ndarray
    look_at: np.ndarray
    up: np.ndarray
    fov_deg: float
    frame_idx: int

    def to_dict(self) -> dict:
        return {
            "eye": self.eye.tolist(),
            "look_at": self.look_at.tolist(),
            "up": self.up.tolist(),
            "fov_deg": self.fov_deg,
            "frame_idx": self.frame_idx,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Keyframe":
        return cls(
            eye=np.array(d["eye"], dtype=np.float32),
            look_at=np.array(d["look_at"], dtype=np.float32),
            up=np.array(d["up"], dtype=np.float32),
            fov_deg=d.get("fov_deg", 60.0),
            frame_idx=d.get("frame_idx", 0),
        )


class KeyframeAnimator:
    """Manages keyframes and produces interpolated camera paths."""

    def __init__(self):
        self.keyframes: List[Keyframe] = []

    def add(self, camera: OrbitCamera, frame_idx: int = 0):
        kf = Keyframe(
            eye=camera.eye.copy(),
            look_at=camera.look_at.copy(),
            up=camera.up.copy(),
            fov_deg=camera.fov_deg,
            frame_idx=frame_idx,
        )
        self.keyframes.append(kf)

    def remove_last(self):
        if self.keyframes:
            self.keyframes.pop()

    def clear(self):
        self.keyframes.clear()

    def interpolate(self, steps_per_segment: int = 60) -> List[Tuple[OrbitCamera, int]]:
        """Interpolate keyframes with cubic spline. Returns list of (camera, frame_idx)."""
        if len(self.keyframes) < 2:
            if self.keyframes:
                cam = OrbitCamera(
                    self.keyframes[0].eye,
                    self.keyframes[0].look_at,
                    self.keyframes[0].up,
                    self.keyframes[0].fov_deg,
                )
                return [(cam, self.keyframes[0].frame_idx)]
            return []

        from scipy.interpolate import CubicSpline

        n = len(self.keyframes)
        t_knots = np.linspace(0, 1, n)
        t_interp = np.linspace(0, 1, (n - 1) * steps_per_segment + 1)

        # Stack all fields
        eyes = np.array([kf.eye for kf in self.keyframes])
        look_ats = np.array([kf.look_at for kf in self.keyframes])
        ups = np.array([kf.up for kf in self.keyframes])
        fovs = np.array([kf.fov_deg for kf in self.keyframes])
        frames = np.array([kf.frame_idx for kf in self.keyframes], dtype=np.float64)

        cs_eye = CubicSpline(t_knots, eyes)
        cs_look = CubicSpline(t_knots, look_ats)
        cs_up = CubicSpline(t_knots, ups)
        cs_fov = CubicSpline(t_knots, fovs)
        cs_frame = CubicSpline(t_knots, frames)

        result = []
        for t in t_interp:
            e = cs_eye(t).astype(np.float32)
            la = cs_look(t).astype(np.float32)
            u = _normalize(cs_up(t).astype(np.float32))
            f = float(cs_fov(t))
            fi = int(np.clip(np.round(cs_frame(t)), 0, max(frames)))
            cam = OrbitCamera(e, la, u, f)
            result.append((cam, fi))
        return result

    def save(self, path: str):
        data = [kf.to_dict() for kf in self.keyframes]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        self.keyframes = [Keyframe.from_dict(d) for d in data]

    def to_list(self) -> List[dict]:
        return [kf.to_dict() for kf in self.keyframes]
