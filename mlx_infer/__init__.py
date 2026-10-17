"""MLX inference package for GCTStream on Apple Silicon."""

try:
    import mlx.core  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "mlx_infer requires MLX. Install it with: pip install lingbot-map[mlx]"
    ) from e

from .model import GCTStreamMLX
from .weights import load_checkpoint

__all__ = ["GCTStreamMLX", "load_checkpoint"]
