"""ONNX-based sky segmentation.

Input:  a folder of RGB images.
Output: sky_mask/ sibling folder with per-image PNG masks.

Output sky_mask PNG convention:
    pixel value 255  ->  sky
    pixel value 0    ->  not sky

Usage:
    python seg_sky.py --rgb_dir /path/to/scene/dense/rgb
    python seg_sky.py --rgb_dir /path/to/rgb --batch_size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from tqdm import tqdm


# Default ONNX model path
DEFAULT_MODEL_PATH = Path("sky_seg/skyseg_batch.onnx")

# Supported RGB image extensions
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".JPG", ".PNG"]


# ---------------------------------------------------------------------------
# Model lookup
# ---------------------------------------------------------------------------

def get_model(model_path: Optional[str] = None) -> Path:
    """Return Path to ONNX model. Raises FileNotFoundError if not present."""
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Sky segmentation model not found: {path}\n"
            f"Please download the model and place it at: {path.resolve()}"
        )
    return path


# ---------------------------------------------------------------------------
# SkySegmenter
# ---------------------------------------------------------------------------

class SkySegmenter:
    """ONNX-based sky segmenter."""

    def __init__(self, onnx_path: str):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        available = ort.get_available_providers()
        if 'CUDAExecutionProvider' in available:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.providers = self.session.get_providers()
        self.model_h = inp.shape[2] if isinstance(inp.shape[2], int) else 384
        self.model_w = inp.shape[3] if isinstance(inp.shape[3], int) else 384

    def _preprocess(self, rgb_hw3: np.ndarray) -> np.ndarray:
        img = cv2.resize(rgb_hw3, (self.model_w, self.model_h)).astype(np.float32) / 255.0
        return ((img - self.mean) / self.std).transpose(2, 0, 1)

    def _extract_sky_probs(self, out: np.ndarray) -> np.ndarray:
        if out.ndim == 4 and out.shape[1] == 2: return out[:, 1]
        if out.ndim == 4 and out.shape[1] == 1: return out[:, 0]
        if out.ndim == 3: return out
        raise ValueError(f"Unexpected output shape: {out.shape}")

    def segment_batch(self, rgb_batch: List[np.ndarray]) -> List[np.ndarray]:
        """Segment a batch of RGB images. Returns list of bool arrays (True = sky)."""
        if not rgb_batch:
            return []
        sizes = [img.shape[:2] for img in rgb_batch]
        inp = np.stack([self._preprocess(img) for img in rgb_batch])
        probs = self._extract_sky_probs(self.session.run(None, {self.input_name: inp})[0])
        return [
            cv2.resize(
                (p > 0.5).astype(np.uint8), (W, H),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for p, (H, W) in zip(probs, sizes)
        ]

    def segment(self, rgb_hw3: np.ndarray) -> np.ndarray:
        """Segment a single RGB image. Returns bool array (True = sky)."""
        return self.segment_batch([rgb_hw3])[0]


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def collect_images(rgb_dir: Path) -> list:
    """Return sorted list of image files in rgb_dir."""
    files = []
    for ext in IMAGE_EXTS:
        files.extend(rgb_dir.glob(f"*{ext}"))
    return sorted(set(files))


def process(rgb_dir: Path, segmenter: SkySegmenter, batch_size: int):
    """Process rgb_dir: read images, predict sky masks, write PNGs to sibling sky_mask/."""
    sky_mask_dir = rgb_dir.parent / "sky_mask"
    sky_mask_dir.mkdir(exist_ok=True)

    image_files = collect_images(rgb_dir)
    if not image_files:
        print("No images found.")
        return

    # Skip already-saved masks
    todo = [p for p in image_files if not (sky_mask_dir / (p.stem + ".png")).exists()]
    if not todo:
        print(f"All {len(image_files)} masks already exist. Nothing to do.")
        return

    print(f"Total: {len(image_files)}, pending: {len(todo)}")

    for batch_start in tqdm(range(0, len(todo), batch_size), desc="Batches"):
        batch_paths = todo[batch_start:batch_start + batch_size]
        images_rgb = [np.array(Image.open(p).convert("RGB")) for p in batch_paths]
        sky_masks = segmenter.segment_batch(images_rgb)
        for p, mask in zip(batch_paths, sky_masks):
            cv2.imwrite(str(sky_mask_dir / (p.stem + ".png")), mask.astype(np.uint8) * 255)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sky segmentation (ONNX).")
    parser.add_argument("--rgb_dir", type=str, required=True, help="Path to folder of RGB images.")
    parser.add_argument("--model_path", type=str, default=None,
                        help=f"Path to ONNX model file (default: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Images per forward pass (default: 4). Reduce if OOM.")
    args = parser.parse_args()

    model_path = str(get_model(args.model_path))
    rgb_dir = Path(args.rgb_dir)
    if not rgb_dir.is_dir():
        print(f"[error] rgb_dir not found: {rgb_dir}")
        sys.exit(1)

    segmenter = SkySegmenter(model_path)
    print(f"Providers: {segmenter.providers}")
    process(rgb_dir, segmenter, args.batch_size)
    print("Done.")


if __name__ == "__main__":
    main()
