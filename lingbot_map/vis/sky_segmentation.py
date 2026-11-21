# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sky segmentation utilities for filtering sky points from point clouds.
"""

import glob
import os
from typing import Optional, Tuple

import numpy as np
import cv2
from tqdm.auto import tqdm

try:
    import onnxruntime
except ImportError:
    onnxruntime = None
    print("onnxruntime not found. Sky segmentation may not work.")


_SKYSEG_INPUT_SIZE = (320, 320)
# Non-sky confidence threshold; > 0.875 keeps raw skyseg scores < 32/255, matching VGGT.
_SKYSEG_SOFT_THRESHOLD = 0.875
_SKYSEG_CACHE_VERSION = "imagenet_norm_softmap_inverted_v4"
_SKYSEG_MODEL_URL = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"


def _get_cache_version_path(sky_mask_dir: str) -> str:
    return os.path.join(sky_mask_dir, ".skyseg_cache_version")


def _prepare_sky_mask_cache(sky_mask_dir: Optional[str]) -> bool:
    """Ensure the sky mask cache directory exists; return True if cached masks are stale."""
    if sky_mask_dir is None:
        return False
    os.makedirs(sky_mask_dir, exist_ok=True)
    version_path = _get_cache_version_path(sky_mask_dir)
    refresh_cache = True
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            refresh_cache = f.read().strip() != _SKYSEG_CACHE_VERSION
    if refresh_cache:
        print(f"Sky mask cache at {sky_mask_dir} uses an older format; regenerating masks")
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(_SKYSEG_CACHE_VERSION)
    return refresh_cache


def _ensure_skyseg_model(skyseg_model_path: str) -> None:
    """Download the skyseg model when the requested path is not a regular file."""
    if os.path.isfile(skyseg_model_path):
        return

    print(f"Sky segmentation model not found at {skyseg_model_path}, downloading...")
    try:
        download_skyseg_model(skyseg_model_path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to download the sky segmentation model to {skyseg_model_path} "
            f"from {_SKYSEG_MODEL_URL}: {error}. Download it manually from "
            f"{_SKYSEG_MODEL_URL} and set skyseg_model_path to the downloaded model file."
        ) from error

    if not os.path.isfile(skyseg_model_path):
        raise RuntimeError(
            f"Sky segmentation model download to {skyseg_model_path} from "
            f"{_SKYSEG_MODEL_URL} completed, but no model file was created. "
            f"Download it manually from {_SKYSEG_MODEL_URL} and set "
            "skyseg_model_path to the downloaded model file."
        )


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    """
    Run ONNX sky segmentation on a BGR image and return an 8-bit score map.
    """
    resize_image = cv2.resize(image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[1], input_size[0]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    denom = max(max_value - min_value, 1e-8)
    onnx_result = (onnx_result - min_value) / denom
    onnx_result *= 255.0
    return onnx_result.astype(np.uint8)


def _mask_to_float(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        # run_skyseg outputs and cached PNGs are 8-bit maps in [0, 255].
        return mask.astype(np.float32) / 255.0
    mask = mask.astype(np.float32)
    if mask.size == 0:
        return mask
    return np.clip(mask, 0.0, 1.0)


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype == np.uint8:
        return mask
    mask = mask.astype(np.float32)
    if mask.size > 0 and mask.max() <= 1.0:
        mask = mask * 255.0
    return np.clip(mask, 0.0, 255.0).astype(np.uint8)


def _result_map_to_non_sky_conf(result_map: np.ndarray) -> np.ndarray:
    # The raw skyseg map is higher on sky and lower on non-sky.
    return 1.0 - _mask_to_float(result_map)


def segment_sky_from_array(
    image: np.ndarray,
    skyseg_session,
    target_h: int,
    target_w: int
) -> np.ndarray:
    """
    Segment sky from an image array using ONNX model.

    Args:
        image: Input image as numpy array (H, W, 3) or (3, H, W), values in [0, 1] or [0, 255]
        skyseg_session: ONNX runtime inference session
        target_h: Target output height
        target_w: Target output width

    Returns:
        Continuous non-sky confidence map in [0, 1].
    """
    image_rgb = _image_to_rgb_uint8(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image_bgr)
    result_map = cv2.resize(result_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return _result_map_to_non_sky_conf(result_map)


def _imread_unicode(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Read an image with Unicode / non-ASCII path safety across platforms."""
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, flags)
    except OSError:
        return None


def _imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Write an image with Unicode / non-ASCII path safety across platforms."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, encoded = cv2.imencode(ext, img)
    if not ok:
        return False
    encoded.tofile(path)
    return True


def _default_sky_mask_dir(image_folder: str) -> str:
    resolved = os.path.abspath(image_folder).rstrip("/\\")
    return f"{resolved}_sky_masks"


def segment_sky(
    image_path: str,
    skyseg_session,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Segment sky from an image using ONNX model.

    Args:
        image_path: Path to the input image
        skyseg_session: ONNX runtime inference session
        output_path: Optional path to save the mask

    Returns:
        Continuous non-sky confidence map in [0, 1].
    """
    image = _imread_unicode(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = _result_map_to_non_sky_conf(result_map)

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        _imwrite_unicode(output_path, _mask_to_uint8(mask))

    return mask


def _list_image_files(image_folder: str) -> list[str]:
    image_files = sorted(glob.glob(os.path.join(glob.escape(image_folder), "*")))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return [f for f in image_files if os.path.splitext(f.lower())[1] in image_extensions]


def _image_to_rgb_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = image.transpose(1, 2, 0)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected image with shape (H, W, 3) or (3, H, W), got {image.shape}")

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)

    return image


def _get_mask_filename(image_paths: Optional[list[str]], index: int) -> str:
    if image_paths is not None and index < len(image_paths):
        return os.path.basename(image_paths[index])
    return f"frame_{index:06d}.png"


def _save_sky_mask_visualization(
    image: np.ndarray,
    sky_mask: np.ndarray,
    output_path: str,
) -> None:
    image_rgb = _image_to_rgb_uint8(image)
    if sky_mask.shape[:2] != image_rgb.shape[:2]:
        sky_mask = cv2.resize(
            sky_mask,
            (image_rgb.shape[1], image_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    mask_uint8 = _mask_to_uint8(sky_mask)
    mask_rgb = np.repeat(mask_uint8[..., None], 3, axis=2)
    overlay = image_rgb.astype(np.float32).copy()
    sky_pixels = _mask_to_float(sky_mask) <= _SKYSEG_SOFT_THRESHOLD
    overlay[sky_pixels] = overlay[sky_pixels] * 0.35 + np.array([255, 64, 64], dtype=np.float32) * 0.65
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    panel = np.concatenate([image_rgb, mask_rgb, overlay], axis=1)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    _imwrite_unicode(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def load_or_create_sky_masks(
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    num_frames: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load cached sky masks or generate them with the ONNX model.

    Args:
        image_folder: Folder containing input images.
        image_paths: Optional explicit image file list, in the exact order to process.
        images: Optional image array with shape (S, 3, H, W) or (S, H, W, 3).
        skyseg_model_path: Path to the sky segmentation ONNX model.
        sky_mask_dir: Optional directory for cached raw masks.
        sky_mask_visualization_dir: Optional directory for side-by-side visualizations.
        target_shape: Optional output mask shape (H, W) after resizing.
        num_frames: Optional maximum number of frames to process.

    Returns:
        Sky masks with shape (S, H, W), or None if sky segmentation could not run.
    """
    if onnxruntime is None:
        print("Warning: onnxruntime not available, skipping sky segmentation")
        return None

    if image_folder is None and image_paths is None and images is None:
        print("Warning: Neither image_folder/image_paths nor images provided, skipping sky segmentation")
        return None

    _ensure_skyseg_model(skyseg_model_path)

    skyseg_session = onnxruntime.InferenceSession(skyseg_model_path)
    sky_masks = []

    if sky_mask_visualization_dir is not None:
        os.makedirs(sky_mask_visualization_dir, exist_ok=True)
        print(f"Saving sky mask visualizations to {sky_mask_visualization_dir}")

    if images is not None:
        if image_paths is None and image_folder is not None:
            image_paths = _list_image_files(image_folder)

        num_images = images.shape[0]
        if num_frames is not None:
            num_images = min(num_images, num_frames)
        if image_paths is not None:
            image_paths = image_paths[:num_images]

        if sky_mask_dir is None and image_folder is not None:
            sky_mask_dir = _default_sky_mask_dir(image_folder)
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        print("Generating sky masks from image array...")
        for i in tqdm(range(num_images)):
            image_rgb = _image_to_rgb_uint8(images[i])
            image_h, image_w = image_rgb.shape[:2]
            image_name = _get_mask_filename(image_paths, i)
            mask_filepath = os.path.join(sky_mask_dir, image_name) if sky_mask_dir is not None else None

            if mask_filepath is not None and not refresh_cache and os.path.exists(mask_filepath):
                sky_mask = _imread_unicode(mask_filepath, cv2.IMREAD_GRAYSCALE)
                if sky_mask is not None and sky_mask.shape[:2] == (image_h, image_w):
                    # Reuse cached mask
                    pass
                else:
                    sky_mask = segment_sky_from_array(image_rgb, skyseg_session, image_h, image_w)
                    _imwrite_unicode(mask_filepath, _mask_to_uint8(sky_mask))
            else:
                sky_mask = segment_sky_from_array(image_rgb, skyseg_session, image_h, image_w)
                if mask_filepath is not None:
                    _imwrite_unicode(mask_filepath, _mask_to_uint8(sky_mask))

            if sky_mask_visualization_dir is not None:
                _save_sky_mask_visualization(
                    image_rgb,
                    sky_mask,
                    os.path.join(sky_mask_visualization_dir, image_name),
                )

            if target_shape is not None and sky_mask.shape[:2] != target_shape:
                sky_mask = cv2.resize(
                    sky_mask,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            sky_masks.append(_mask_to_float(sky_mask))

    else:
        if image_paths is None and image_folder is not None:
            image_paths = _list_image_files(image_folder)

    if images is None and image_paths is not None:
        if len(image_paths) == 0:
            print("Warning: No image files provided, skipping sky segmentation")
            return None

        if num_frames is not None:
            image_paths = image_paths[:num_frames]

        if sky_mask_dir is None:
            if image_folder is None:
                image_folder = os.path.dirname(image_paths[0]) or "."
            sky_mask_dir = _default_sky_mask_dir(image_folder)
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        print("Generating sky masks from image files...")
        for image_path in tqdm(image_paths):
            image_name = os.path.basename(image_path)
            mask_filepath = os.path.join(sky_mask_dir, image_name)

            if not refresh_cache and os.path.exists(mask_filepath):
                sky_mask = _imread_unicode(mask_filepath, cv2.IMREAD_GRAYSCALE)
                if sky_mask is None:
                    print(f"Warning: Failed to read cached sky mask {mask_filepath}, regenerating it")
                    sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)
            else:
                sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

            if sky_mask is None:
                print(f"Warning: Failed to produce sky mask for {image_path}, skipping frame")
                continue

            if sky_mask_visualization_dir is not None:
                image_bgr = _imread_unicode(image_path, cv2.IMREAD_COLOR)
                if image_bgr is not None:
                    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                    _save_sky_mask_visualization(
                        image_rgb,
                        sky_mask,
                        os.path.join(sky_mask_visualization_dir, image_name),
                    )

            if target_shape is not None and sky_mask.shape[:2] != target_shape:
                sky_mask = cv2.resize(
                    sky_mask,
                    (target_shape[1], target_shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            sky_masks.append(_mask_to_float(sky_mask))

    if len(sky_masks) == 0:
        print("Warning: No sky masks generated, skipping sky segmentation")
        return None

    try:
        return np.stack(sky_masks, axis=0)
    except ValueError:
        return np.array(sky_masks, dtype=object)


def apply_sky_segmentation(
    conf: np.ndarray,
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf: Confidence scores with shape (S, H, W)
        image_folder: Path to the folder containing input images (optional if images provided)
        image_paths: Optional explicit image file list in processing order
        images: Image array with shape (S, 3, H, W) or (S, H, W, 3) (optional if image_folder provided)
        skyseg_model_path: Path to the sky segmentation ONNX model
        sky_mask_dir: Optional directory for cached raw masks
        sky_mask_visualization_dir: Optional directory for side-by-side mask visualization images

    Returns:
        Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape

    sky_mask_array = load_or_create_sky_masks(
        image_folder=image_folder,
        image_paths=image_paths,
        images=images,
        skyseg_model_path=skyseg_model_path,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
        target_shape=(H, W),
        num_frames=S,
    )
    if sky_mask_array is None:
        return conf

    if sky_mask_array.shape[0] < S:
        print(
            f"Warning: Only {sky_mask_array.shape[0]} sky masks generated for {S} frames; "
            "leaving the remaining frames unmasked"
        )
        padded = np.zeros((S, H, W), dtype=sky_mask_array.dtype)
        padded[: sky_mask_array.shape[0]] = sky_mask_array
        sky_mask_array = padded
    elif sky_mask_array.shape[0] > S:
        sky_mask_array = sky_mask_array[:S]

    sky_mask_binary = (sky_mask_array > _SKYSEG_SOFT_THRESHOLD).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


def download_skyseg_model(output_path: str = "skyseg.onnx") -> str:
    """
    Download sky segmentation model from HuggingFace.

    Args:
        output_path: Path to save the model

    Returns:
        Path to the downloaded model
    """
    import tempfile

    import requests

    url = _SKYSEG_MODEL_URL

    print(f"Downloading sky segmentation model from {url}...")
    response = requests.get(url, stream=True)
    temp_path = None
    try:
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        output_dir = os.path.dirname(output_path) or "."
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as model_file:
            temp_path = model_file.name
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    model_file.write(chunk)
                    pbar.update(len(chunk))

        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        try:
            response.close()
        except Exception:
            pass

    print(f"Model saved to {output_path}")
    return output_path
