"""Validated single-call FlashInfer paged-KV append.

This module deliberately contains no custom arithmetic.  It re-expresses the
accepted patch-page and six-special-token writes as one graph-capturable
FlashInfer append operation while preserving the raw bf16 payload.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
from typing import Dict

import torch


FLASHINFER_DISTRIBUTION = "flashinfer-python"
FLASHINFER_VERSION = "0.6.11.post3"
FLASHINFER_SOURCE_HASHES = {
    "page.py": "05144a5fa890c501364a7c21550db8594df54e04b34af0555c2054343ee4dacb",
    "data/csrc/page.cu": "cba680599c65a58a7ea2ef9a34e7f7166781c45715015d4b4fbf686cbc0a0a60",
    "data/include/flashinfer/page.cuh": "6ad25f50ed08ea74495d8a8d172baf57266e53ce0d42b14dc9e979c1073c0468",
}
ALLOWED_TOKENS_PER_FRAME = frozenset((783, 978, 1005, 1042))
NUM_HEADS = 16
HEAD_DIM = 64
NUM_SPECIAL_TOKENS = 6


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_direct_append_runtime() -> Dict[str, object]:
    """Pin the installed FlashInfer append implementation before capture."""
    import flashinfer.page as flashinfer_page

    try:
        version = importlib.metadata.version(FLASHINFER_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"Combined KV append requires {FLASHINFER_DISTRIBUTION} distribution metadata"
        ) from exc

    root = Path(flashinfer_page.__file__).resolve().parent
    observed_hashes = {
        relative: _sha256_file(root / relative)
        for relative in FLASHINFER_SOURCE_HASHES
    }
    errors = []
    if version != FLASHINFER_VERSION:
        errors.append(f"{FLASHINFER_DISTRIBUTION}={version}, expected {FLASHINFER_VERSION}")
    for relative, expected in FLASHINFER_SOURCE_HASHES.items():
        observed = observed_hashes[relative]
        if observed != expected:
            errors.append(f"{relative} sha256={observed}, expected {expected}")
    if errors:
        raise RuntimeError("Combined KV append FlashInfer runtime rejected: " + "; ".join(errors))
    return {
        "distribution": FLASHINFER_DISTRIBUTION,
        "version": version,
        "source_root": str(root),
        "source_hashes": observed_hashes,
    }


def direct_append_module_fingerprint() -> Dict[str, object]:
    """Record the actual loaded JIT module in addition to pinned source inputs."""
    import flashinfer.page as flashinfer_page

    module = flashinfer_page.get_page_module()
    module_path_value = getattr(module, "__file__", None)
    module_path = Path(module_path_value).resolve() if module_path_value else None
    return {
        "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
        "module_path": str(module_path) if module_path is not None else None,
        "module_sha256": (
            _sha256_file(module_path)
            if module_path is not None and module_path.is_file()
            else None
        ),
    }


def _tensor_contract_error(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    stride: tuple[int, ...] | None = None,
) -> list[str]:
    errors = []
    if tuple(tensor.shape) != shape:
        errors.append(f"{name}.shape={tuple(tensor.shape)}, expected {shape}")
    if tensor.dtype != dtype:
        errors.append(f"{name}.dtype={tensor.dtype}, expected {dtype}")
    if tensor.device != device:
        errors.append(f"{name}.device={tensor.device}, expected {device}")
    if not tensor.is_contiguous():
        errors.append(f"{name}.stride={tuple(tensor.stride())}, expected contiguous")
    if stride is not None and tuple(tensor.stride()) != stride:
        errors.append(f"{name}.stride={tuple(tensor.stride())}, expected {stride}")
    return errors


def validate_direct_append_call(
    k: torch.Tensor,
    v: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    """Validate only static tensor metadata; never read CUDA values or sync."""
    errors = []
    if paged_kv_cache.ndim != 5:
        errors.append(
            f"paged_kv_cache.ndim={paged_kv_cache.ndim}, expected 5"
        )
        raise RuntimeError("Combined KV append append call rejected: " + "; ".join(errors))

    device = paged_kv_cache.device
    page_size = int(paged_kv_cache.shape[2])
    tokens_per_frame = page_size + NUM_SPECIAL_TOKENS
    expected_kv_shape = (tokens_per_frame, NUM_HEADS, HEAD_DIM)
    expected_kv_stride = (NUM_HEADS * HEAD_DIM, HEAD_DIM, 1)
    expected_cache_shape = (
        int(paged_kv_cache.shape[0]),
        2,
        page_size,
        NUM_HEADS,
        HEAD_DIM,
    )
    expected_cache_stride = (
        2 * page_size * NUM_HEADS * HEAD_DIM,
        page_size * NUM_HEADS * HEAD_DIM,
        NUM_HEADS * HEAD_DIM,
        HEAD_DIM,
        1,
    )

    if device.type != "cuda":
        errors.append(f"paged_kv_cache.device={device}, expected CUDA")
    elif torch.cuda.get_device_capability(device) != (11, 0):
        errors.append(
            "compute_capability="
            f"{torch.cuda.get_device_capability(device)}, expected (11, 0)"
        )
    if tokens_per_frame not in ALLOWED_TOKENS_PER_FRAME:
        errors.append(
            f"tokens_per_frame={tokens_per_frame}, expected one of "
            f"{sorted(ALLOWED_TOKENS_PER_FRAME)}"
        )
    if paged_kv_cache.dtype != torch.bfloat16:
        errors.append(
            f"paged_kv_cache.dtype={paged_kv_cache.dtype}, expected torch.bfloat16"
        )
    if tuple(paged_kv_cache.shape) != expected_cache_shape:
        errors.append(
            f"paged_kv_cache.shape={tuple(paged_kv_cache.shape)}, "
            f"expected {expected_cache_shape}"
        )
    if not paged_kv_cache.is_contiguous() or tuple(paged_kv_cache.stride()) != expected_cache_stride:
        errors.append(
            f"paged_kv_cache.stride={tuple(paged_kv_cache.stride())}, "
            f"expected {expected_cache_stride}"
        )
    errors.extend(
        _tensor_contract_error(
            "k", k, shape=expected_kv_shape, dtype=torch.bfloat16,
            device=device, stride=expected_kv_stride,
        )
    )
    errors.extend(
        _tensor_contract_error(
            "v", v, shape=expected_kv_shape, dtype=torch.bfloat16,
            device=device, stride=expected_kv_stride,
        )
    )
    errors.extend(
        _tensor_contract_error(
            "batch_indices", batch_indices, shape=(tokens_per_frame,),
            dtype=torch.int32, device=device, stride=(1,),
        )
    )
    errors.extend(
        _tensor_contract_error(
            "positions", positions, shape=(tokens_per_frame,),
            dtype=torch.int32, device=device, stride=(1,),
        )
    )
    if kv_indices.ndim != 1 or kv_indices.numel() < 2:
        errors.append(
            f"kv_indices.shape={tuple(kv_indices.shape)}, expected one-dimensional length >= 2"
        )
    if kv_indices.dtype != torch.int32 or kv_indices.device != device or not kv_indices.is_contiguous():
        errors.append(
            f"kv_indices contract dtype={kv_indices.dtype}, device={kv_indices.device}, "
            f"stride={tuple(kv_indices.stride())}; expected contiguous CUDA int32"
        )
    errors.extend(
        _tensor_contract_error(
            "kv_indptr", kv_indptr, shape=(3,), dtype=torch.int32,
            device=device, stride=(1,),
        )
    )
    errors.extend(
        _tensor_contract_error(
            "kv_last_page_len", kv_last_page_len, shape=(2,),
            dtype=torch.int32, device=device, stride=(1,),
        )
    )
    if k.requires_grad or v.requires_grad or paged_kv_cache.requires_grad:
        errors.append("Combined KV append is inference-only")
    if errors:
        raise RuntimeError("Combined KV append append call rejected: " + "; ".join(errors))


def append_paged_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    paged_kv_cache: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
) -> None:
    """Write patch and special tokens with one mutating FlashInfer operation."""
    import flashinfer.page

    validate_direct_append_call(
        k,
        v,
        paged_kv_cache,
        batch_indices,
        positions,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
    )
    flashinfer.page.append_paged_kv_cache(
        k,
        v,
        batch_indices,
        positions,
        paged_kv_cache,
        kv_indices,
        kv_indptr,
        kv_last_page_len,
        kv_layout="NHD",
    )
