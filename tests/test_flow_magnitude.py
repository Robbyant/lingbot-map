import math

import pytest
import torch

from lingbot_map.models.gct_stream_window import (
    _compute_flow_magnitude as compute_flow_magnitude,
)
from lingbot_map.models.gct_stream_window_v2 import (
    _compute_flow_magnitude as compute_flow_magnitude_v2,
)
from lingbot_map.utils.rotation import mat_to_quat


def _rotation_xyz(x: float, y: float, z: float) -> torch.Tensor:
    cx, sx = math.cos(x), math.sin(x)
    cy, sy = math.cos(y), math.sin(y)
    cz, sz = math.cos(z), math.sin(z)
    return torch.tensor(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ]
    )


def _pose_encoding(
    rotation_w2c: torch.Tensor,
    translation_w2c: torch.Tensor,
    fov_h: torch.Tensor,
    fov_w: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [translation_w2c, mat_to_quat(rotation_w2c), fov_h[None], fov_w[None]]
    )[None, None]


def _reference_flow_magnitude(
    cur_rotation_w2c: torch.Tensor,
    cur_translation_w2c: torch.Tensor,
    kf_rotation_w2c: torch.Tensor,
    kf_translation_w2c: torch.Tensor,
    depth: torch.Tensor,
    cur_focal_xy: tuple[float, float],
    kf_focal_xy: tuple[float, float],
    stride: int,
    *,
    legacy_inverse_convention: bool = False,
) -> float:
    height, width = depth.shape[2:4]
    cur_fx, cur_fy = cur_focal_xy
    kf_fx, kf_fy = kf_focal_xy
    v, u = torch.meshgrid(
        torch.arange(0, height, stride, dtype=depth.dtype),
        torch.arange(0, width, stride, dtype=depth.dtype),
        indexing="ij",
    )
    z = depth[0, 0, ::stride, ::stride, 0]
    cur_camera = torch.stack(
        [
            (u - width / 2) / cur_fx * z,
            (v - height / 2) / cur_fy * z,
            z,
        ],
        dim=-1,
    ).reshape(-1, 3)

    if legacy_inverse_convention:
        # Previous implementation interpreted both w2c matrices backwards:
        # x_kf = E_kf^-1 E_cur x_cur.
        world = (cur_rotation_w2c @ cur_camera.transpose(0, 1)).transpose(
            0, 1
        ) + cur_translation_w2c
        keyframe_camera = (
            kf_rotation_w2c.transpose(0, 1)
            @ (world - kf_translation_w2c).transpose(0, 1)
        ).transpose(0, 1)
    else:
        # Pose encodings contain w2c matrices, so x_kf = E_kf E_cur^-1 x_cur.
        world = (
            cur_rotation_w2c.transpose(0, 1)
            @ (cur_camera - cur_translation_w2c).transpose(0, 1)
        ).transpose(0, 1)
        keyframe_camera = (kf_rotation_w2c @ world.transpose(0, 1)).transpose(
            0, 1
        ) + kf_translation_w2c

    keyframe_pixels = torch.stack(
        [
            kf_fx * keyframe_camera[:, 0] / keyframe_camera[:, 2] + width / 2,
            kf_fy * keyframe_camera[:, 1] / keyframe_camera[:, 2] + height / 2,
        ],
        dim=-1,
    )
    source_pixels = torch.stack([u, v], dim=-1).reshape(-1, 2)
    valid = keyframe_camera[:, 2] > 1e-6
    return (keyframe_pixels[valid] - source_pixels[valid]).norm(dim=-1).mean().item()


@pytest.mark.parametrize("flow_fn", [compute_flow_magnitude, compute_flow_magnitude_v2])
def test_flow_magnitude_respects_world_to_camera_pose_contract(flow_fn):
    height, width = 24, 40
    cur_focal_xy = (55.0, 43.0)
    kf_focal_xy = (61.0, 49.0)
    stride = 4
    cur_fov_h = 2 * torch.atan(torch.tensor((height / 2) / cur_focal_xy[1]))
    cur_fov_w = 2 * torch.atan(torch.tensor((width / 2) / cur_focal_xy[0]))
    kf_fov_h = 2 * torch.atan(torch.tensor((height / 2) / kf_focal_xy[1]))
    kf_fov_w = 2 * torch.atan(torch.tensor((width / 2) / kf_focal_xy[0]))

    cur_rotation_w2c = _rotation_xyz(0.3, 0.3, 0.0)
    kf_rotation_w2c = _rotation_xyz(0.4, 0.3, -0.1)
    cur_translation_w2c = torch.tensor([0.4, 0.0, 0.0])
    kf_translation_w2c = torch.tensor([0.3, 0.1, 0.1])
    cur_pose = _pose_encoding(
        cur_rotation_w2c, cur_translation_w2c, cur_fov_h, cur_fov_w
    )
    kf_pose = _pose_encoding(kf_rotation_w2c, kf_translation_w2c, kf_fov_h, kf_fov_w)
    depth = torch.linspace(2, 12, height * width).reshape(1, 1, height, width, 1)

    reference_args = (
        cur_rotation_w2c,
        cur_translation_w2c,
        kf_rotation_w2c,
        kf_translation_w2c,
        depth,
        cur_focal_xy,
        kf_focal_xy,
        stride,
    )
    expected = _reference_flow_magnitude(*reference_args)
    legacy = _reference_flow_magnitude(*reference_args, legacy_inverse_convention=True)
    actual = flow_fn(cur_pose, kf_pose, depth, (height, width), stride=stride)

    assert actual == pytest.approx(expected, rel=1e-5, abs=1e-5)
    assert actual < 5.0 < legacy
