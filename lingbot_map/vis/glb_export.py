# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GLB 3D export utilities for GCT predictions.
"""

import os
import copy
from typing import Optional, Tuple

import numpy as np
import cv2
import matplotlib
from scipy.spatial.transform import Rotation

from lingbot_map.utils.geometry import unproject_depth_map_to_point_map
from lingbot_map.vis.sky_segmentation import (
    _SKYSEG_INPUT_SIZE,
    _mask_to_uint8,
    _result_map_to_non_sky_conf,
    apply_sky_segmentation,
    run_skyseg as _canonical_run_skyseg,
    segment_sky as _canonical_segment_sky,
)

try:
    import trimesh
except ImportError:
    trimesh = None
    print("trimesh not found. GLB export will not work.")


def predictions_to_glb(
    predictions: dict,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    prediction_mode: str = "Predicted Pointmap",
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> "trimesh.Scene":
    """
    Converts GCT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions: Dictionary containing model predictions with keys:
            - world_points: Optional 3D point coordinates (S, H, W, 3)
            - world_points_conf: Optional point confidence scores (S, H, W)
            - depth: Depth maps used when world_points is unavailable
              (S, H, W) or (S, H, W, 1)
            - depth_conf: Optional depth confidence scores (S, H, W)
            - intrinsic: Camera intrinsic matrices (S, 3, 3)
            - images: Input images (S, H, W, 3) or (S, 3, H, W)
            - extrinsic: World-to-camera matrices (S, 3, 4)
        conf_thres: Percentage of low-confidence points to filter out
        filter_by_frames: Frame filter specification ("all" or frame index)
        mask_black_bg: Mask out black background pixels
        mask_white_bg: Mask out white background pixels
        show_cam: Include camera visualization
        mask_sky: Apply sky segmentation mask
        target_dir: Output directory for intermediate files
        prediction_mode: "Predicted Pointmap" or "Predicted Depthmap"
        skyseg_model_path: Path to the sky segmentation ONNX model
        sky_mask_dir: Optional directory for cached sky masks
        sky_mask_visualization_dir: Optional directory for mask visualizations

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
        ImportError: If trimesh is not available
    """
    if trimesh is None:
        raise ImportError("trimesh is required for GLB export. Install with: pip install trimesh")

    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    print("Building GLB scene")

    # Parse frame filter
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    # Select prediction source
    if "Pointmap" in prediction_mode:
        print("Using Pointmap Branch")
        if predictions.get("world_points") is not None:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            print("Warning: world_points not found, falling back to depth-based points")
            pred_world_points = _get_depth_based_world_points(predictions)
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = _get_depth_based_world_points(predictions)
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    # Apply sky segmentation if enabled
    if mask_sky:
        pred_world_points_conf = _apply_sky_mask(
            pred_world_points_conf,
            target_dir,
            images,
            skyseg_model_path=skyseg_model_path,
            sky_mask_dir=sky_mask_dir,
            sky_mask_visualization_dir=sky_mask_visualization_dir,
        )

    # Apply frame filter
    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    # Prepare vertices and colors
    vertices_3d = pred_world_points.reshape(-1, 3)

    # Handle different image formats
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Apply confidence filtering
    conf = pred_world_points_conf.reshape(-1)
    conf_threshold = np.percentile(conf, conf_thres) if conf_thres > 0 else 0.0
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    # Apply background masking
    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        white_bg_mask = ~(
            (colors_rgb[:, 0] > 240) &
            (colors_rgb[:, 1] > 240) &
            (colors_rgb[:, 2] > 240)
        )
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # Handle empty point cloud
    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = max(float(np.linalg.norm(upper_percentile - lower_percentile)), 1e-3)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Build scene
    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    # Prepare camera matrices
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    # Add cameras
    if show_cam:
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def _get_depth_based_world_points(predictions: dict) -> np.ndarray:
    """Return depth-unprojected world points, computing them when necessary."""
    precomputed = predictions.get("world_points_from_depth")
    if precomputed is not None:
        return precomputed

    required_keys = ("depth", "extrinsic", "intrinsic")
    missing_keys = [
        key for key in required_keys if predictions.get(key) is None
    ]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            "Depth-based GLB export requires depth, extrinsic, and intrinsic "
            f"predictions; missing: {missing}"
        )

    print("Unprojecting depth maps for GLB export")
    return unproject_depth_map_to_point_map(
        predictions["depth"],
        predictions["extrinsic"],
        predictions["intrinsic"],
    )


def _apply_sky_mask(
    conf: np.ndarray,
    target_dir: Optional[str],
    images: np.ndarray,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> np.ndarray:
    """Apply sky segmentation mask to confidence scores."""
    image_folder = None
    if target_dir is not None:
        candidate = os.path.join(target_dir, "images")
        if os.path.isdir(candidate):
            image_folder = candidate
        if sky_mask_dir is None:
            sky_mask_dir = os.path.join(target_dir, "sky_masks")

    return apply_sky_segmentation(
        conf,
        image_folder=image_folder,
        images=images,
        skyseg_model_path=skyseg_model_path,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
    )


def integrate_camera_into_scene(
    scene: "trimesh.Scene",
    transform: np.ndarray,
    face_colors: Tuple[int, int, int],
    scene_scale: float,
    frustum_thickness: float = 1.0,
):
    """
    Integrates a camera mesh into the 3D scene.

    Args:
        scene: The 3D scene to add the camera model
        transform: Transformation matrix for camera positioning
        face_colors: RGB color tuple for the camera
        scene_scale: Scale of the scene
        frustum_thickness: Multiplier for frustum edge thickness (>1 = thicker)
    """
    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Build thicker frustum by stacking rotated copies
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    shell_scales = [1.0, 0.95]
    shell_transforms = [np.eye(4), slight_rotation]
    # Add extra shells for thickness
    if frustum_thickness > 1.0:
        n_extra = max(1, int(frustum_thickness - 1))
        for k in range(1, n_extra + 1):
            # Progressively rotated and scaled copies
            angle = 2.0 + k * 2.0
            scale = 1.0 + k * 0.02
            rot = np.eye(4)
            rot[:3, :3] = Rotation.from_euler("z", angle, degrees=True).as_matrix()
            shell_scales.append(scale)
            shell_transforms.append(rot)
            rot_neg = np.eye(4)
            rot_neg[:3, :3] = Rotation.from_euler("z", -angle, degrees=True).as_matrix()
            shell_scales.append(scale)
            shell_transforms.append(rot_neg)

    vertices_parts = []
    for s, t_mat in zip(shell_scales, shell_transforms):
        vertices_parts.append(
            transform_points(t_mat, s * camera_cone_shape.vertices)
        )
    vertices_combined = np.concatenate(vertices_parts)
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces_multi(camera_cone_shape, len(shell_scales))
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(
    scene_3d: "trimesh.Scene",
    extrinsics_matrices: np.ndarray
) -> "trimesh.Scene":
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d: The 3D scene to be aligned
        extrinsics_matrices: Camera extrinsic matrices

    Returns:
        Aligned 3D scene
    """
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    initial_transformation = (
        np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    )
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """Returns the OpenGL conversion matrix (flips Y and Z axes)."""
    matrix = np.identity(4)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(
    transformation: np.ndarray,
    points: np.ndarray,
    dim: Optional[int] = None
) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation: Transformation matrix
        points: Points to be transformed
        dim: Dimension for reshaping the result

    Returns:
        Transformed points
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    transformation = transformation.swapaxes(-1, -2)
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    return points[..., :dim].reshape(*initial_shape, dim)


def compute_camera_faces(cone_shape: "trimesh.Trimesh") -> np.ndarray:
    """Computes the faces for the camera mesh."""
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend([
            (v1, v2, v2_offset),
            (v1, v1_offset, v3),
            (v3_offset, v2, v3),
            (v1, v2, v2_offset_2),
            (v1, v1_offset_2, v3),
            (v3_offset_2, v2, v3),
        ])

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def compute_camera_faces_multi(cone_shape: "trimesh.Trimesh", num_shells: int) -> np.ndarray:
    """Computes faces for a camera mesh with multiple shells (for thicker frustums).

    Connects each consecutive pair of vertex shells to form the frustum edges.
    """
    faces_list = []
    nv = len(cone_shape.vertices)

    for s in range(num_shells - 1):
        off_a = s * nv
        off_b = (s + 1) * nv
        for face in cone_shape.faces:
            if 0 in face:
                continue
            v1, v2, v3 = face
            faces_list.extend([
                (v1 + off_a, v2 + off_a, v2 + off_b),
                (v1 + off_a, v1 + off_b, v3 + off_a),
                (v3 + off_b, v2 + off_a, v3 + off_a),
            ])

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(
    image_path: str,
    onnx_session,
    mask_filename: Optional[str] = None,
) -> np.ndarray:
    """
    Segments sky from an image using an ONNX model.

    Delegates to ``lingbot_map.vis.sky_segmentation.segment_sky`` so directory
    creation, missing-image validation, and constant-map epsilon guards stay
    consistent across visualization and GLB export.
    """
    return _canonical_segment_sky(image_path, onnx_session, output_path=mask_filename)


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    """
    Runs sky segmentation inference using ONNX model.

    Delegates to ``lingbot_map.vis.sky_segmentation.run_skyseg`` so ``(W, H)``
    input_size ordering and uniform-output division guards match the viewer.
    """
    return _canonical_run_skyseg(onnx_session, input_size, image)


def download_file_from_url(url: str, filename: str):
    """Downloads a file from a URL, handling redirects."""
    import requests

    try:
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()

        if response.status_code == 302:
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
