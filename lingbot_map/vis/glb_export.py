# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
3D export utilities for GCT predictions.
"""

import os
import copy
import importlib
import importlib.util
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2
import matplotlib
from scipy.spatial.transform import Rotation

from lingbot_map.vis.sky_segmentation import (
    _SKYSEG_INPUT_SIZE,
    _SKYSEG_SOFT_THRESHOLD,
    _mask_to_float,
    _mask_to_uint8,
    _result_map_to_non_sky_conf,
)

trimesh: Any = None
if importlib.util.find_spec("trimesh") is not None:
    trimesh = importlib.import_module("trimesh")
else:
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
) -> "trimesh.Scene":
    """
    Converts GCT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions: Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3) or (S, 3, H, W)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
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
    vertices_3d, colors_rgb, extrinsics_matrices, scene_scale = _prepare_export_point_cloud(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )

    if np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    scene_3d = trimesh.Scene()
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)

    # Add cameras
    if show_cam and len(extrinsics_matrices) > 0:
        num_cameras = len(extrinsics_matrices)
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = (
                int(255 * rgba_color[0]),
                int(255 * rgba_color[1]),
                int(255 * rgba_color[2]),
            )
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene
    if len(extrinsics_matrices) > 0:
        scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    print("GLB Scene built")
    return scene_3d


def predictions_to_ply(
    predictions: dict,
    output_path: str,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    prediction_mode: str = "Predicted Pointmap",
) -> int:
    """
    Export filtered prediction points as a PLY point cloud.

    Args:
        predictions: Prediction dictionary in the same format as predictions_to_glb.
        output_path: Destination ``.ply`` path.
        conf_thres: Percentage of low-confidence points to filter out.
        filter_by_frames: Frame filter specification ("all" or frame index).
        mask_black_bg: Mask out black background pixels.
        mask_white_bg: Mask out white background pixels.
        mask_sky: Apply sky segmentation mask.
        target_dir: Output directory for intermediate files.
        prediction_mode: "Predicted Pointmap" or "Predicted Depthmap".

    Returns:
        Number of exported points.
    """
    vertices_3d, colors_rgb, extrinsics_matrices, _ = _prepare_export_point_cloud(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=filter_by_frames,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    if len(vertices_3d) > 0 and len(extrinsics_matrices) > 0:
        vertices_3d = apply_scene_alignment_to_vertices(vertices_3d, extrinsics_matrices)
    save_point_cloud_to_ply(vertices_3d, colors_rgb, output_path)
    return int(len(vertices_3d))


def save_point_cloud_to_ply(
    vertices: np.ndarray,
    colors_rgb: np.ndarray,
    output_path: str,
    normals: Optional[np.ndarray] = None,
    alpha: Optional[np.ndarray] = None,
    extra_vertex_properties: Optional[Dict[str, np.ndarray]] = None,
    comments: Optional[List[str]] = None,
    extra_elements: Optional[List[Tuple[str, Dict[str, np.ndarray]]]] = None,
) -> None:
    """Write a point cloud to a binary little-endian PLY file."""
    vertices = np.asarray(vertices, dtype=np.float32)
    colors_rgb = _coerce_colors_to_uint8(colors_rgb)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if colors_rgb.ndim != 2 or colors_rgb.shape[1] != 3:
        raise ValueError(f"colors_rgb must have shape (N, 3), got {colors_rgb.shape}")
    if len(vertices) != len(colors_rgb):
        raise ValueError(
            f"vertices/colors length mismatch: {len(vertices)} vs {len(colors_rgb)}"
        )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    vertex_properties: "OrderedDict[str, np.ndarray]" = OrderedDict()
    vertex_properties["x"] = vertices[:, 0]
    vertex_properties["y"] = vertices[:, 1]
    vertex_properties["z"] = vertices[:, 2]
    vertex_properties["red"] = colors_rgb[:, 0]
    vertex_properties["green"] = colors_rgb[:, 1]
    vertex_properties["blue"] = colors_rgb[:, 2]

    if normals is not None:
        normals = np.asarray(normals, dtype=np.float32)
        if normals.ndim != 2 or normals.shape != vertices.shape:
            raise ValueError(
                f"normals must have shape {vertices.shape}, got {normals.shape}"
            )
        vertex_properties["nx"] = normals[:, 0]
        vertex_properties["ny"] = normals[:, 1]
        vertex_properties["nz"] = normals[:, 2]

    if alpha is not None:
        alpha = np.asarray(alpha)
        if alpha.ndim != 1 or len(alpha) != len(vertices):
            raise ValueError(
                f"alpha must have shape ({len(vertices)},), got {alpha.shape}"
            )
        vertex_properties["alpha"] = np.clip(alpha, 0, 255).astype(np.uint8)

    if extra_vertex_properties:
        for name, values in extra_vertex_properties.items():
            if name in vertex_properties:
                raise ValueError(f"duplicate PLY vertex property: {name}")
            vertex_properties[name] = _coerce_ply_property_array(
                name, values, len(vertices)
            )

    elements: List[Tuple[str, Dict[str, np.ndarray]]] = [("vertex", vertex_properties)]
    if extra_elements:
        elements.extend(extra_elements)

    _write_ply_elements(output_path, elements, comments)


def _prepare_export_point_cloud(
    predictions: dict,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    prediction_mode: str = "Predicted Pointmap",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Prepare filtered point-cloud export data shared by GLB and PLY."""
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
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get(
                "depth_conf", np.ones_like(pred_world_points[..., 0])
            )
    else:
        print("Using Depthmap and Camera Branch")
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get(
            "depth_conf", np.ones_like(pred_world_points[..., 0])
        )

    images = predictions["images"]
    camera_matrices = predictions["extrinsic"]

    if mask_sky and target_dir is not None:
        pred_world_points_conf = _apply_sky_mask(
            pred_world_points_conf, target_dir, images
        )

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = np.asarray(pred_world_points).reshape(-1, 3).astype(np.float32, copy=False)

    if images.ndim == 4 and images.shape[1] == 3:
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = _coerce_colors_to_uint8(colors_rgb.reshape(-1, 3))

    conf = np.asarray(pred_world_points_conf).reshape(-1)
    conf_threshold = np.percentile(conf, conf_thres) if conf_thres > 0 else 0.0
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

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

    extrinsics_matrices = np.zeros((len(camera_matrices), 4, 4), dtype=np.float32)
    if len(camera_matrices) > 0:
        extrinsics_matrices[:, :3, :4] = camera_matrices
        extrinsics_matrices[:, 3, 3] = 1

    if np.asarray(vertices_3d).size == 0:
        scene_scale = 1.0
    else:
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        scene_scale = float(np.linalg.norm(upper_percentile - lower_percentile))
        scene_scale = max(scene_scale, 0.1)

    return vertices_3d, colors_rgb, extrinsics_matrices, scene_scale


def _coerce_colors_to_uint8(colors_rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0, 1] or integer RGB in [0, 255] to uint8."""
    colors_rgb = np.asarray(colors_rgb)
    if colors_rgb.dtype == np.uint8:
        return colors_rgb
    if np.issubdtype(colors_rgb.dtype, np.floating):
        return (np.clip(colors_rgb, 0.0, 1.0) * 255).astype(np.uint8)
    return np.clip(colors_rgb, 0, 255).astype(np.uint8)


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

        if sky_mask is None:
            print(f"Warning: failed to read sky mask for {image_name}, keeping all pixels")
            sky_mask = np.full((H, W), 255, dtype=np.uint8)
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H), interpolation=cv2.INTER_LINEAR)

        sky_mask_list.append(_mask_to_float(sky_mask))

    sky_mask_array = np.array(sky_mask_list)
    sky_mask_binary = (sky_mask_array > _SKYSEG_SOFT_THRESHOLD).astype(np.float32)
    return conf * sky_mask_binary


def integrate_camera_into_scene(
    scene: "trimesh.Scene",
    transform: np.ndarray,
    face_colors: Sequence[int],
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
    camera_visual = camera_mesh.visual
    camera_visual.face_colors[:, :3] = tuple(face_colors[:3])
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
    initial_transformation = get_scene_alignment_transform(extrinsics_matrices)
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def apply_scene_alignment_to_vertices(
    vertices: np.ndarray,
    extrinsics_matrices: np.ndarray,
) -> np.ndarray:
    """Apply the same scene alignment used by GLB export directly to vertices."""
    if len(extrinsics_matrices) == 0:
        return np.asarray(vertices, dtype=np.float32)

    transformation = get_scene_alignment_transform(extrinsics_matrices)
    return transform_points(transformation, np.asarray(vertices, dtype=np.float32))


def apply_scene_alignment_to_directions(
    vectors: np.ndarray,
    extrinsics_matrices: np.ndarray,
) -> np.ndarray:
    """Apply scene alignment rotation to direction vectors such as normals."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(extrinsics_matrices) == 0 or vectors.size == 0:
        return vectors

    rotation_only = np.eye(4, dtype=np.float32)
    rotation_only[:3, :3] = get_scene_alignment_transform(extrinsics_matrices)[:3, :3]
    rotated = transform_points(rotation_only, vectors)
    norm = np.linalg.norm(rotated, axis=-1, keepdims=True)
    valid = norm > 1e-8
    rotated = np.where(valid, rotated / np.where(valid, norm, 1.0), 0.0)
    return rotated.astype(np.float32, copy=False)


def get_scene_alignment_transform(extrinsics_matrices: np.ndarray) -> np.ndarray:
    """Return the world transform used to align GLB/PLY exports."""
    if len(extrinsics_matrices) == 0:
        return np.eye(4, dtype=np.float32)

    opengl_conversion_matrix = get_opengl_conversion_matrix()

    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    transformation = (
        np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    )
    return transformation.astype(np.float32, copy=False)


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


def _write_ply_elements(
    output_path: str,
    elements: List[Tuple[str, Dict[str, np.ndarray]]],
    comments: Optional[List[str]] = None,
) -> None:
    """Write one or more binary PLY elements to disk."""
    prepared_elements = [
        _prepare_ply_element(name, properties)
        for name, properties in elements
    ]

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
    ]
    for comment in comments or []:
        header_lines.append(f"comment {comment}")
    for name, structured, property_specs in prepared_elements:
        header_lines.append(f"element {name} {len(structured)}")
        for property_name, ply_type in property_specs:
            header_lines.append(f"property {ply_type} {property_name}")
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(output_path, "wb") as f:
        f.write(header.encode("ascii"))
        for _, structured, _ in prepared_elements:
            structured.tofile(f)


def _prepare_ply_element(
    element_name: str,
    properties: Dict[str, np.ndarray],
) -> Tuple[str, np.ndarray, List[Tuple[str, str]]]:
    """Normalize one PLY element into a structured array plus header specs."""
    if not properties:
        raise ValueError(f"PLY element {element_name!r} has no properties")

    normalized: "OrderedDict[str, np.ndarray]" = OrderedDict()
    element_length: Optional[int] = None
    for property_name, values in properties.items():
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError(
                f"PLY property {element_name}.{property_name} must be 1D, got {values.shape}"
            )
        if element_length is None:
            element_length = len(values)
        elif len(values) != element_length:
            raise ValueError(
                f"PLY element {element_name!r} property length mismatch for {property_name}: "
                f"expected {element_length}, got {len(values)}"
            )
        normalized[property_name] = _normalize_ply_property_dtype(values)

    assert element_length is not None
    dtype_fields = []
    property_specs = []
    for property_name, values in normalized.items():
        little_endian_dtype = values.dtype.newbyteorder("<")
        dtype_fields.append((property_name, little_endian_dtype))
        property_specs.append((property_name, _numpy_dtype_to_ply_type(little_endian_dtype)))

    structured = np.empty(element_length, dtype=np.dtype(dtype_fields))
    for property_name, values in normalized.items():
        structured[property_name] = values.astype(
            structured[property_name].dtype,
            copy=False,
        )

    return element_name, structured, property_specs


def _coerce_ply_property_array(
    property_name: str,
    values: np.ndarray,
    expected_length: int,
) -> np.ndarray:
    """Validate an extra PLY property array against the expected vertex count."""
    values = np.asarray(values)
    if values.ndim != 1 or len(values) != expected_length:
        raise ValueError(
            f"PLY property {property_name!r} must have shape ({expected_length},), "
            f"got {values.shape}"
        )
    return _normalize_ply_property_dtype(values)


def _normalize_ply_property_dtype(values: np.ndarray) -> np.ndarray:
    """Cast property arrays to PLY-compatible scalar dtypes."""
    values = np.asarray(values)
    if values.dtype == np.bool_:
        return values.astype(np.uint8)
    if values.dtype == np.float16:
        return values.astype(np.float32)
    if np.issubdtype(values.dtype, np.floating):
        return values.astype(np.float32 if values.dtype.itemsize <= 4 else np.float64)
    if np.issubdtype(values.dtype, np.signedinteger):
        return values.astype(np.int32 if values.dtype.itemsize > 4 else values.dtype)
    if np.issubdtype(values.dtype, np.unsignedinteger):
        if values.dtype.itemsize > 4:
            return values.astype(np.uint32)
        return values.astype(values.dtype)
    raise ValueError(f"Unsupported PLY property dtype: {values.dtype}")


def _numpy_dtype_to_ply_type(dtype: np.dtype) -> str:
    """Map a numpy scalar dtype to a binary PLY header type."""
    dtype = np.dtype(dtype).newbyteorder("=")
    dtype_map = {
        np.dtype(np.int8): "char",
        np.dtype(np.uint8): "uchar",
        np.dtype(np.int16): "short",
        np.dtype(np.uint16): "ushort",
        np.dtype(np.int32): "int",
        np.dtype(np.uint32): "uint",
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
    }
    try:
        return dtype_map[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY dtype: {dtype}") from exc


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
    if image is None:
        raise ValueError(f"Failed to read image for sky segmentation: {image_path}")
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
