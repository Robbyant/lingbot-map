"""Device / dtype / attention-backend helpers for portable CPU-first inference.

FlashInfer is an optional CUDA acceleration. SDPA is the default portable path.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import nullcontext
from typing import Literal, Optional, Union

import torch

logger = logging.getLogger(__name__)

# Conservative free-VRAM floor (bytes) before ``auto`` will select CUDA.
# Full fp16/bf16 load of lingbot-map needs multi-GB plus activations; 4 GB laptop
# GPUs (e.g. RTX 2050) cannot hold the ~4.4 GB checkpoint.
_AUTO_CUDA_MIN_FREE_BYTES = 8 * (1024 ** 3)

AttentionBackend = Literal["sdpa", "flashinfer"]

_FLASHINFER_PROBE: Optional[tuple[bool, str]] = None


def flashinfer_usable() -> tuple[bool, str]:
    """Soft-probe FlashInfer. Never raises.

    Returns:
        (usable, reason) where reason explains a negative result (or "ok").
    """
    global _FLASHINFER_PROBE
    if _FLASHINFER_PROBE is not None:
        return _FLASHINFER_PROBE

    try:
        import flashinfer  # noqa: F401
    except Exception as exc:  # ImportError and rare init failures
        _FLASHINFER_PROBE = (False, f"import failed: {type(exc).__name__}: {exc}")
        return _FLASHINFER_PROBE

    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            sm = major * 10 + minor
            # FlashInfer documents SM 7.5+ (Turing and later).
            if sm < 75:
                _FLASHINFER_PROBE = (
                    False,
                    f"GPU compute capability sm_{major}{minor} < sm_75",
                )
                return _FLASHINFER_PROBE
        except Exception as exc:
            _FLASHINFER_PROBE = (False, f"capability check failed: {exc}")
            return _FLASHINFER_PROBE

    _FLASHINFER_PROBE = (True, "ok")
    return _FLASHINFER_PROBE


def _cuda_free_memory_bytes(device_index: int = 0) -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free)
    except Exception:
        try:
            props = torch.cuda.get_device_properties(device_index)
            return int(props.total_memory)
        except Exception:
            return None


def resolve_device(
    requested: Optional[str] = "cpu",
    *,
    min_free_bytes: int = _AUTO_CUDA_MIN_FREE_BYTES,
) -> torch.device:
    """Resolve a user device request to a concrete ``torch.device``.

    - ``cpu`` (default): always CPU.
    - ``cuda``: CUDA if available, else warn and use CPU.
    - ``auto``: CUDA only when available and free VRAM >= ``min_free_bytes``, else CPU.
    - ``None`` / empty: treated as ``cpu``.
    """
    choice = (requested or "cpu").strip().lower()

    if choice in ("cpu",):
        return torch.device("cpu")

    if choice in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "CUDA was requested but is not available; falling back to CPU.",
            stacklevel=2,
        )
        return torch.device("cpu")

    if choice == "auto":
        if not torch.cuda.is_available():
            logger.info("resolve_device(auto): CUDA unavailable → cpu")
            return torch.device("cpu")
        free = _cuda_free_memory_bytes(0)
        if free is None or free < min_free_bytes:
            free_gb = (free or 0) / (1024 ** 3)
            need_gb = min_free_bytes / (1024 ** 3)
            warnings.warn(
                f"resolve_device(auto): free VRAM {free_gb:.1f} GiB < "
                f"{need_gb:.1f} GiB threshold → cpu "
                "(pass --device cuda to force GPU anyway).",
                stacklevel=2,
            )
            return torch.device("cpu")
        return torch.device("cuda")

    # Allow torch-style strings like "cuda:0"
    try:
        device = torch.device(choice)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(
            f"Unknown device {requested!r}; use cpu, cuda, or auto."
        ) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            f"Device {requested!r} requested but CUDA is unavailable; using CPU.",
            stacklevel=2,
        )
        return torch.device("cpu")
    return device


def resolve_dtype(device: Union[torch.device, str]) -> torch.dtype:
    """Inference dtype: fp32 on CPU; bf16/fp16 on CUDA by capability."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device.index or 0)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast_context(device: Union[torch.device, str], dtype: torch.dtype):
    """AMP context: enabled only on CUDA with a non-fp32 dtype."""
    device = torch.device(device) if isinstance(device, str) else device
    if device.type != "cuda" or dtype == torch.float32:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def select_attention_backend(
    device: Union[torch.device, str],
    *,
    force_sdpa: bool = False,
    force_flashinfer: bool = False,
) -> AttentionBackend:
    """Choose SDPA (portable) or FlashInfer (optional CUDA).

    Prefers SDPA whenever the device is CPU, FlashInfer is missing, or the
    caller asked for SDPA. ``force_flashinfer`` raises only when the probe fails
    after an explicit request.
    """
    device = torch.device(device) if isinstance(device, str) else device
    usable, reason = flashinfer_usable()

    if force_flashinfer:
        if device.type != "cuda":
            raise RuntimeError(
                "FlashInfer was forced but device is not CUDA "
                f"(got {device}). Use SDPA on CPU."
            )
        if not usable:
            raise RuntimeError(
                "FlashInfer was forced but is not usable "
                f"({reason}). Install a matching wheel or use --use_sdpa."
            )
        return "flashinfer"

    if force_sdpa or device.type != "cuda":
        if device.type != "cuda":
            logger.info("Attention backend: SDPA (CPU device)")
        elif force_sdpa:
            logger.info("Attention backend: SDPA (--use_sdpa / force)")
        return "sdpa"

    if not usable:
        warnings.warn(
            f"FlashInfer not usable ({reason}); using SDPA backend.",
            stacklevel=2,
        )
        return "sdpa"

    return "flashinfer"


def should_use_sdpa(
    device: Union[torch.device, str],
    *,
    prefer_sdpa: bool = False,
) -> bool:
    """Convenience wrapper: True when ``select_attention_backend`` chooses SDPA."""
    return select_attention_backend(device, force_sdpa=prefer_sdpa) == "sdpa"
