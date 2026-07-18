"""Sky segmentation utilities.

Convention: output mask uint8, 255=non-sky (valid), 0=sky.
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEFAULT_MODEL_PATH = Path.home() / ".cache" / "benchmark" / "skyseg.onnx"
INPUT_SIZE = 320


def get_model(model_path: Optional[str] = None) -> Path:
    """Return Path to ONNX model. Raises FileNotFoundError if not present.

    Args:
        model_path: Optional explicit path to ONNX model file.
                    If None, uses DEFAULT_MODEL_PATH.

    Returns:
        Path to the ONNX model file.
    """
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Sky segmentation model not found: {path}\n"
            f"Please download the model and place it at: {path.resolve()}"
        )
    return path


def create_session(model_path: Path):
    """Create onnxruntime.InferenceSession; raises ImportError if not installed.

    Args:
        model_path: Path to ONNX model file.

    Returns:
        onnxruntime.InferenceSession instance.

    Raises:
        ImportError: If onnxruntime is not installed.
    """
    try:
        import onnxruntime
    except ImportError:
        raise ImportError(
            "onnxruntime is required for sky segmentation. "
            "Install it with: pip install onnxruntime"
        )
    return onnxruntime.InferenceSession(str(model_path))


def segment_sky_rgb(rgb: np.ndarray, session) -> np.ndarray:
    """Run sky segmentation on HxWx3 uint8 RGB array.

    Args:
        rgb: HxWx3 uint8 RGB image array.
        session: onnxruntime.InferenceSession for the sky segmentation model.

    Returns:
        HxW uint8 mask at original resolution (255=non-sky/valid, 0=sky).
    """
    orig_h, orig_w = rgb.shape[:2]

    img = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0

    # https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing/blob/1f7811b32b64ddc957269defff84bc87a3f0b74f/onnx_interence.py#L15C5-L17C31
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std

    img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)  # (1, 3, H, W)

    result = session.run(
        [session.get_outputs()[0].name],
        {session.get_inputs()[0].name: img}
    )[0]

    # result shape: (1, 1, INPUT_SIZE, INPUT_SIZE)
    # Model outputs high value for sky regions (result > 0.5 = sky).
    # Convention: 255 = non-sky (valid), 0 = sky.
    mask = (result[0, 0] <= 0.5).astype(np.uint8) * 255

    return cv2.resize(mask, (orig_w, orig_h))
