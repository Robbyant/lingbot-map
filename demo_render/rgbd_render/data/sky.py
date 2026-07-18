"""Sky segmentation utilities.

Adapted from lingbot_map.vis.sky_segmentation to keep demo_render self-contained.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


_SKYSEG_INPUT_SIZE = (320, 320)
_SKYSEG_SOFT_THRESHOLD = 0.1
_SKYSEG_CACHE_VERSION = "imagenet_norm_softmap_inverted_v3"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _get_cache_version_path(sky_mask_dir: str) -> str:
    return os.path.join(sky_mask_dir, ".skyseg_cache_version")


def _prepare_sky_mask_cache(sky_mask_dir: Optional[str]) -> bool:
    if sky_mask_dir is None:
        return False

    os.makedirs(sky_mask_dir, exist_ok=True)
    version_path = _get_cache_version_path(sky_mask_dir)
    refresh_cache = True
    if os.path.exists(version_path):
        with open(version_path, "r", encoding="utf-8") as f:
            refresh_cache = f.read().strip() != _SKYSEG_CACHE_VERSION

    if refresh_cache:
        print(
            f"Sky mask cache at {sky_mask_dir} uses an older format; "
            "regenerating masks with ImageNet-normalized skyseg input"
        )
        with open(version_path, "w", encoding="utf-8") as f:
            f.write(_SKYSEG_CACHE_VERSION)

    return refresh_cache


def _mask_to_float(mask: np.ndarray) -> np.ndarray:
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


def _result_map_to_non_sky_conf(result_map: np.ndarray) -> np.ndarray:
    return 1.0 - _mask_to_float(result_map)


def _list_image_files(image_folder: str) -> list[str]:
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    return [f for f in image_files if os.path.splitext(f.lower())[1] in image_extensions]


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
    sky_pixels = (_mask_to_float(sky_mask) <= _SKYSEG_SOFT_THRESHOLD)[..., None]  # (H, W, 1)
    red_tint = np.array([255, 64, 64], dtype=np.float32)
    overlay = np.where(sky_pixels, overlay * 0.35 + red_tint * 0.65, overlay)
    overlay = np.clip(overlay, 0.0, 255.0).astype(np.uint8)

    panel = np.concatenate([image_rgb, mask_rgb, overlay], axis=1)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def _preprocess_skyseg(image_bgr: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess a single BGR image for skyseg ONNX model. Returns (3, H, W) float32."""
    resized = cv2.resize(image_bgr, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x / 255.0 - mean) / std
    return x.transpose(2, 0, 1)


def _normalize_skyseg_output(raw: np.ndarray) -> np.ndarray:
    """Normalize a single raw score map to uint8 [0, 255]."""
    min_value = np.min(raw)
    max_value = np.max(raw)
    denom = max(max_value - min_value, 1e-8)
    normed = (raw - min_value) / denom * 255.0
    return normed.astype(np.uint8)


def run_skyseg(
    onnx_session,
    input_size: Tuple[int, int],
    image: np.ndarray,
) -> np.ndarray:
    """Run ONNX sky segmentation on a single BGR image and return an 8-bit score map."""
    x = _preprocess_skyseg(image, input_size)
    x = x.reshape(1, 3, input_size[1], input_size[0]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    onnx_result = np.array(onnx_result).squeeze()
    return _normalize_skyseg_output(onnx_result)


def run_skyseg_batch(
    onnx_session,
    input_size: Tuple[int, int],
    images_bgr: List[np.ndarray],
) -> List[np.ndarray]:
    """Run ONNX sky segmentation on a batch of BGR images. Returns list of uint8 score maps."""
    if not images_bgr:
        return []
    batch = np.stack([_preprocess_skyseg(img, input_size) for img in images_bgr]).astype("float32")

    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: batch})

    raw = np.array(onnx_result)
    # ONNX wraps output in a list -> (1, B, [C], H, W). Remove the list dim.
    while raw.ndim > 3 and raw.shape[0] == 1:
        raw = raw.squeeze(0)
    # Now raw should be (B, H, W) or (B, C, H, W)
    if raw.ndim == 4:
        # Multi-channel output: take last channel (sky channel)
        raw = raw[:, -1]
    if raw.ndim == 2:
        raw = raw[np.newaxis]
    return [_normalize_skyseg_output(raw[i]) for i in range(raw.shape[0])]


def segment_sky_from_array(
    image: np.ndarray,
    skyseg_session,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Segment sky from an image array, returning continuous non-sky confidence in [0, 1]."""
    image_rgb = _image_to_rgb_uint8(image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image_bgr)
    result_map = cv2.resize(result_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return _result_map_to_non_sky_conf(result_map)


def segment_sky_from_array_batch(
    images: List[np.ndarray],
    skyseg_session,
    target_h: int,
    target_w: int,
) -> List[np.ndarray]:
    """Batch version of segment_sky_from_array."""
    bgr_images = []
    for img in images:
        rgb = _image_to_rgb_uint8(img)
        bgr_images.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    score_maps = run_skyseg_batch(skyseg_session, _SKYSEG_INPUT_SIZE, bgr_images)
    results = []
    for sm in score_maps:
        resized = cv2.resize(sm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        results.append(_result_map_to_non_sky_conf(resized))
    return results


def segment_sky(
    image_path: str,
    skyseg_session,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Segment sky from an image file, returning continuous non-sky confidence in [0, 1]."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    result_map = run_skyseg(skyseg_session, _SKYSEG_INPUT_SIZE, image)
    result_map = cv2.resize(result_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    mask = _result_map_to_non_sky_conf(result_map)

    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, _mask_to_uint8(mask))

    return mask


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_or_create_sky_masks(
    image_folder: Optional[str] = None,
    image_paths: Optional[list[str]] = None,
    images: Optional[np.ndarray] = None,
    skyseg_model_path: str = "skyseg.onnx",
    sky_mask_dir: Optional[str] = None,
    sky_mask_visualization_dir: Optional[str] = None,
    target_shape: Optional[Tuple[int, int]] = None,
    num_frames: Optional[int] = None,
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    """
    Load cached sky masks or generate them with the ONNX model.

    Args:
        batch_size: Number of images to process per ONNX batch inference call.

    Returns:
        Sky masks with shape (S, H, W) as float32 in [0, 1], or None.
    """
    if onnxruntime is None:
        print("Warning: onnxruntime not available, skipping sky segmentation")
        return None

    if image_folder is None and image_paths is None and images is None:
        print("Warning: Neither image_folder/image_paths nor images provided, skipping sky segmentation")
        return None

    if not os.path.exists(skyseg_model_path):
        print(f"Warning: Sky segmentation model not found at {skyseg_model_path}")
        return None

    skyseg_session = onnxruntime.InferenceSession(skyseg_model_path)
    sky_masks: List[np.ndarray] = []

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
            sky_mask_dir = image_folder.rstrip("/") + "_sky_masks"
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        print(f"Generating sky masks from image array (batch_size={batch_size})...")
        for bs in tqdm(range(0, num_images, batch_size),
                       desc="Sky segmentation", unit="batch"):
            be = min(bs + batch_size, num_images)

            # Separate cached vs. needs-inference
            batch_rgb = []
            batch_indices = []  # indices within [bs, be) that need inference
            batch_results: dict[int, np.ndarray] = {}

            for i in range(bs, be):
                image_rgb = _image_to_rgb_uint8(images[i])
                image_h, image_w = image_rgb.shape[:2]
                image_name = _get_mask_filename(image_paths, i)
                mask_filepath = (os.path.join(sky_mask_dir, image_name)
                                 if sky_mask_dir is not None else None)

                # Try loading from cache
                cached = False
                if mask_filepath is not None and not refresh_cache and os.path.exists(mask_filepath):
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                    if sky_mask is not None and sky_mask.shape[:2] == (image_h, image_w):
                        batch_results[i] = _result_map_to_non_sky_conf(sky_mask) if sky_mask.max() > 1 else _mask_to_float(sky_mask)
                        cached = True

                if not cached:
                    batch_rgb.append(image_rgb)
                    batch_indices.append(i)

            # Batch inference for uncached frames
            if batch_rgb:
                image_h, image_w = batch_rgb[0].shape[:2]
                inferred = segment_sky_from_array_batch(
                    batch_rgb, skyseg_session, image_h, image_w)
                for idx, sky_mask in zip(batch_indices, inferred):
                    batch_results[idx] = sky_mask
                    # Save to cache
                    image_name = _get_mask_filename(image_paths, idx)
                    mask_filepath = (os.path.join(sky_mask_dir, image_name)
                                     if sky_mask_dir is not None else None)
                    if mask_filepath is not None:
                        cv2.imwrite(mask_filepath, _mask_to_uint8(sky_mask))

            # Collect in order, apply viz and resize
            for i in range(bs, be):
                sky_mask = batch_results[i]

                if sky_mask_visualization_dir is not None:
                    image_rgb = _image_to_rgb_uint8(images[i])
                    image_name = _get_mask_filename(image_paths, i)
                    _save_sky_mask_visualization(
                        image_rgb, sky_mask,
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
                image_folder = os.path.dirname(image_paths[0])
            sky_mask_dir = image_folder.rstrip("/") + "_sky_masks"
        refresh_cache = _prepare_sky_mask_cache(sky_mask_dir)

        print(f"Generating sky masks from image files (batch_size={batch_size})...")
        for bs in tqdm(range(0, len(image_paths), batch_size),
                       desc="Sky segmentation", unit="batch"):
            be = min(bs + batch_size, len(image_paths))
            batch_paths = image_paths[bs:be]

            # Separate cached vs. needs-inference
            batch_images_bgr = []
            batch_indices = []
            batch_results: dict[int, np.ndarray] = {}

            for j, image_path in enumerate(batch_paths):
                idx = bs + j
                image_name = os.path.basename(image_path)
                mask_filepath = os.path.join(sky_mask_dir, image_name)

                cached = False
                if not refresh_cache and os.path.exists(mask_filepath):
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                    if sky_mask is not None:
                        batch_results[idx] = _result_map_to_non_sky_conf(sky_mask) if sky_mask.max() > 1 else _mask_to_float(sky_mask)
                        cached = True

                if not cached:
                    img_bgr = cv2.imread(image_path)
                    if img_bgr is None:
                        print(f"Warning: Failed to read {image_path}, skipping")
                        continue
                    batch_images_bgr.append((idx, image_path, img_bgr))
                    batch_indices.append(idx)

            # Batch inference for uncached
            if batch_images_bgr:
                bgr_list = [item[2] for item in batch_images_bgr]
                h, w = bgr_list[0].shape[:2]
                rgb_list = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgr_list]
                inferred = segment_sky_from_array_batch(
                    rgb_list, skyseg_session, h, w)
                for (idx, image_path, _), sky_mask in zip(batch_images_bgr, inferred):
                    batch_results[idx] = sky_mask
                    image_name = os.path.basename(image_path)
                    mask_filepath = os.path.join(sky_mask_dir, image_name)
                    cv2.imwrite(mask_filepath, _mask_to_uint8(sky_mask))

            # Collect in order
            for j, image_path in enumerate(batch_paths):
                idx = bs + j
                if idx not in batch_results:
                    continue
                sky_mask = batch_results[idx]

                if sky_mask_visualization_dir is not None:
                    image_bgr = cv2.imread(image_path)
                    if image_bgr is not None:
                        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                        _save_sky_mask_visualization(
                            image_rgb, sky_mask,
                            os.path.join(sky_mask_visualization_dir,
                                         os.path.basename(image_path)),
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
