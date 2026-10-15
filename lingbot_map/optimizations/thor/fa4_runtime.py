"""Fail-closed loader for the FA4 beta15 overlay.

The FA4 overlay is a process-wide runtime choice because Python and the CuTe DSL
cache ``flash_attn`` modules and compiled kernels globally.  A process may run
either the installed FA4 baseline or the vendored beta15 overlay, never both.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


ENV_FLAG = "LINGBOT_THOR_FA4_QUERY_STAGING"
PAGED_KV_AFFINE_ENV = "LINGBOT_THOR_PAGED_KV_AFFINE"
EXPECTED_VERSION = "4.0.0b15"
OVERLAY_ROOT = (
    Path(__file__).resolve().parents[2]
    / "_vendor"
    / "flash_attn_4_0_0b15"
)

_PROCESS_ENABLED: bool | None = None
_PREPARED_METADATA: dict | None = None


def _requested_enabled() -> bool:
    value = os.environ.get(ENV_FLAG, "")
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{ENV_FLAG} must be unset, '0', or '1'; got {value!r}")
    return value == "1"


def paged_kv_affine_requested() -> bool:
    value = os.environ.get(PAGED_KV_AFFINE_ENV, "")
    if value not in ("", "0", "1"):
        raise RuntimeError(
            f"{PAGED_KV_AFFINE_ENV} must be unset, '0', or '1'; got {value!r}"
        )
    return value == "1"


def fa4_overlay_enabled() -> bool:
    """Return the process-fixed route, rejecting in-process route changes."""
    global _PROCESS_ENABLED

    enabled = _requested_enabled()
    if _PROCESS_ENABLED is None:
        _PROCESS_ENABLED = enabled
    elif _PROCESS_ENABLED is not enabled:
        raise RuntimeError(
            f"{ENV_FLAG} is process-immutable after first use; start a fresh "
            "process to change the FA4 overlay route"
        )
    return enabled


def _loaded_flash_attn_paths() -> list[Path]:
    paths = []
    for name, module in tuple(sys.modules.items()):
        if name != "flash_attn" and not name.startswith("flash_attn."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is not None:
            paths.append(Path(module_file).resolve())
    return paths


def _remove_failed_overlay_imports(
    overlay_root: Path,
    modules_before: set[str],
) -> None:
    for name in tuple(sys.modules):
        if name in modules_before:
            continue
        if name != "flash_attn" and not name.startswith("flash_attn."):
            continue
        module = sys.modules.get(name)
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            sys.modules.pop(name, None)
            continue
        try:
            is_overlay_module = Path(module_file).resolve().is_relative_to(overlay_root)
        except (OSError, RuntimeError):
            is_overlay_module = False
        if is_overlay_module:
            sys.modules.pop(name, None)


def prepare_fa4_overlay() -> dict | None:
    """Select and validate packaged FA4 beta15 before its first import."""
    global _PREPARED_METADATA

    enabled = fa4_overlay_enabled()
    paged_kv_affine_enabled = paged_kv_affine_requested()
    if paged_kv_affine_enabled and not enabled:
        raise RuntimeError(
            f"{PAGED_KV_AFFINE_ENV}=1 requires {ENV_FLAG}=1"
        )
    overlay_root = Path(OVERLAY_ROOT)
    overlay_text = str(overlay_root.resolve())
    if not enabled:
        if overlay_text in sys.path:
            raise RuntimeError(
                "FA4 overlay is present in sys.path while its flag is disabled"
            )
        for loaded_path in _loaded_flash_attn_paths():
            if loaded_path.is_relative_to(overlay_root.resolve()):
                raise RuntimeError(
                    "FA4 overlay beta15 is already loaded while its flag is disabled; "
                    "start a fresh process"
                )
        return None

    if _PREPARED_METADATA is not None:
        return dict(_PREPARED_METADATA)

    from .options import housekeeping_route
    housekeeping_mode = housekeeping_route()

    if overlay_root.is_symlink():
        raise RuntimeError(f"FA4 overlay root must not be a symlink: {overlay_root}")
    overlay_root = overlay_root.resolve()
    cute_root = overlay_root / "flash_attn" / "cute"
    version_file = overlay_root / "VERSION"
    required_files = (
        overlay_root / "flash_attn" / "__init__.py",
        cute_root / "__init__.py",
        cute_root / "interface.py",
        cute_root / "flash_fwd.py",
        cute_root / "flash_fwd_sm100.py",
        cute_root / "paged_kv.py",
        cute_root / "tile_scheduler.py",
        version_file,
    )
    invalid_files = [
        str(path)
        for path in required_files
        if not path.is_file() or path.is_symlink()
    ]
    if invalid_files:
        raise RuntimeError(
            "FA4 beta15 overlay is incomplete or non-regular: "
            + ", ".join(invalid_files)
        )
    version = version_file.read_text().strip()
    if version != EXPECTED_VERSION:
        raise RuntimeError(
            f"FA4 overlay version={version!r}, "
            f"expected {EXPECTED_VERSION!r}"
        )

    loaded_names = [
        name
        for name in sys.modules
        if name == "flash_attn" or name.startswith("flash_attn.")
    ]
    if loaded_names:
        loaded_paths = _loaded_flash_attn_paths()
        observed = str(loaded_paths[0]) if loaded_paths else loaded_names[0]
        raise RuntimeError(
            "FA4 overlay must select beta15 before flash_attn is imported; "
            f"already loaded {observed}"
        )

    inserted = False
    modules_before = set(sys.modules)
    try:
        if overlay_text not in sys.path:
            sys.path.insert(0, overlay_text)
            inserted = True
            importlib.invalidate_caches()
        interface = importlib.import_module("flash_attn.cute.interface")
        interface_path = Path(interface.__file__).resolve()
        if not interface_path.is_relative_to(cute_root):
            raise RuntimeError(
                "FA4 source root rejected: "
                f"observed={interface_path.parent}, expected={cute_root}"
            )
    except Exception:
        if inserted:
            try:
                sys.path.remove(overlay_text)
            except ValueError:
                pass
        _remove_failed_overlay_imports(overlay_root, modules_before)
        importlib.invalidate_caches()
        raise

    # Retain diagnostic keys used by existing measurement records.
    _PREPARED_METADATA = {
        "enabled": True,
        "version": version,
        "overlay_root": str(overlay_root),
        "observed_cute_root": str(interface_path.parent),
        "candidate021_route": housekeeping_mode,
        "candidate023_pagedkv_requested": paged_kv_affine_enabled,
        "optimization": "sm110_blackwell_q_stage_2",
        "process_route_immutable": True,
        "source_contract": "packaged_vendor_version_and_regular_files",
    }
    return dict(_PREPARED_METADATA)
