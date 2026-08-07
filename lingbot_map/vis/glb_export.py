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
    _SKYSEG_SOFT_THRESHOLD,
    _mask_to_float,
    _mask_to_uint8,
    _result_map_to_non_sky_conf,
)

try:
    import trimesh
except ImportError:
    trimesh = None
    print("trimesh not found. GLB export will not work.")


def _world_points_from_depth(predictions: dict) -> np.ndarray:
    """Unproject predicted depth maps into world-space points.

    The released checkpoints are trained with the point head disabled
    (``GCTStream`` defaults to ``enable_point=False``), so ``predictions`` has
    no ``world_points`` key and the point cloud has to be built from depth plus
    camera parameters. This mirrors what the interactive viewers already do,
    so GLB exports and the viser preview show the same geometry.

    Args:
        predictions: Prediction dict containing ``depth`` (S, H, W, 1),
            ``extrinsic`` (S, 3, 4) and ``intrinsic`` (S, 3, 3).

    Returns:
        np.ndarray: World points of shape (S, H, W, 3).

    Raises:
        KeyError: If any of the required keys is missing.
    """
    required = ("depth", "extrinsic", "intrinsic")
    missing = [key for key in required if predictions.get(key) is None]
    if missing:
        raise KeyError(
            "cannot build a point cloud from depth: predictions is missing "
            f"{', '.join(missing)}. Available keys: {sorted(predictions)}"
        )

    return unproject_depth_map_to_point_map(
        predictions["depth"], predictions["extrinsic"], predictions["intrinsic"]
    )


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
) -> "trimesh.Scene":
    """
    Converts GCT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions: Dictionary containing model predictions with keys:
            - images: Input images (S, H, W, 3) or (S, 3, H, W)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
            - intrinsic: Camera intrinsic matrices (S, 3, 3)
            - world_points: 3D point coordinates (S, H, W, 3), optional
            - world_points_conf: Confidence scores (S, H, W), optional
            - depth: Depth maps (S, H, W, 1), used to build the point cloud
              whenever world_points is absent
            - depth_conf: Depth confidence (S, H, W), optional
        conf_thres: Percentage of low-confidence points to filter out
        filter_by_frames: Frame filter specification ("all" or frame index)
        mask_black_bg: Mask out black background pixels
        mask_white_bg: Mask out white background pixels
        show_cam: Include camera visualization
        mask_sky: Apply sky segmentation mask
        target_dir: Output directory for intermediate files
        prediction_mode: "Predicted Pointmap" or "Predicted Depthmap"

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
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]
            pred_world_points_conf = predictions.get(
                "world_points_conf", np.ones_like(pred_world_points[..., 0])
            )
        else:
            print("Warning: world_points not found, falling back to depth-based points")
            pred_world_points = _world_points_from_depth(predictions)
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = _world_points_from_depth(predictions)
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    # Apply sky segmentation if enabled
    if mask_sky and target_dir is not None:
        pred_world_points_conf = _apply_sky_mask(
            pred_world_points_conf, target_dir, images
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
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

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


def _apply_sky_mask(
    conf: np.ndarray,
    target_dir: str,
    images: np.ndarray
) -> np.ndarray:
    """Apply sky segmentation mask to confidence scores."""
    try:
        import onnxruntime
    except ImportError:
        print("Warning: onnxruntime not available, skipping sky masking")
        return conf

    target_dir_images = os.path.join(target_dir, "images")
    if not os.path.exists(target_dir_images):
        print(f"Warning: Images directory not found at {target_dir_images}")
        return conf

    image_list = sorted(os.listdir(target_dir_images))
    S, H, W = conf.shape if hasattr(conf, "shape") else (len(images), images.shape[1], images.shape[2])

    skyseg_model_path = "skyseg.onnx"
    if not os.path.exists(skyseg_model_path):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            skyseg_model_path
        )

    skyseg_session = onnxruntime.InferenceSession(skyseg_model_path)
    sky_mask_list = []

    for i, image_name in enumerate(image_list[:S]):
        image_filepath = os.path.join(target_dir_images, image_name)
        mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H), interpolation=cv2.INTER_LINEAR)

        sky_mask_list.append(_mask_to_float(sky_mask))

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > _SKYSEG_SOFT_THRESHOLD).astype(np.float32)
    return conf * sky_mask_binary


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
    mask_filename: str
) -> np.ndarray:
    """
    Segments sky from an image using an ONNX model.

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        Continuous non-sky confidence map in [0, 1]
    """
    image = cv2.imread(image_path)
    result_map = run_skyseg(onnx_session, _SKYSEG_INPUT_SIZE, image)
    result_map_original = cv2.resize(
        result_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    output_mask = _result_map_to_non_sky_conf(result_map_original)

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, _mask_to_uint8(output_mask))
    return output_mask


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray
) -> np.ndarray:
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        Segmentation mask
    """
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    return onnx_result.astype("uint8")


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
