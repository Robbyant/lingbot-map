"""Public, default-off configuration for the optional Thor inference runtime."""
from dataclasses import asdict, dataclass
import os


FLAGS = {
    "cache_mlp_weights": "LINGBOT_THOR_CACHE_MLP",
    "cache_qkv_weights": "LINGBOT_THOR_CACHE_QKV",
    "cache_projection_weights": "LINGBOT_THOR_CACHE_PROJECTION",
    "merge_kv_append": "LINGBOT_THOR_MERGE_KV_APPEND",
    "fa4_query_staging": "LINGBOT_THOR_FA4_QUERY_STAGING",
    "paged_kv_affine": "LINGBOT_THOR_PAGED_KV_AFFINE",
    "allow_approximate": "LINGBOT_THOR_ALLOW_APPROXIMATE",
}
VISIBLE_ENV = "LINGBOT_THOR_VISIBLE_WINDOW"
_ACTIVE = None


def enabled(name):
    value = os.environ.get(name, "0")
    if value not in ("", "0", "1"):
        raise ValueError(f"{name} must be unset, 0, or 1; got {value!r}")
    return value == "1"


def housekeeping_route():
    projection = enabled(FLAGS["cache_projection_weights"])
    append = enabled(FLAGS["merge_kv_append"])
    if projection and append:
        return "projection_shadow_direct_kv_append"
    if projection:
        return "projection_shadow"
    if append:
        return "direct_kv_append"
    return None


@dataclass(frozen=True)
class ThorOptions:
    cache_mlp_weights: bool = False
    cache_qkv_weights: bool = False
    cache_projection_weights: bool = False
    merge_kv_append: bool = False
    fa4_query_staging: bool = False
    paged_kv_affine: bool = False
    visible_window: int | None = None
    allow_approximate: bool = False

    @classmethod
    def from_env(cls):
        raw = os.environ.get(VISIBLE_ENV, "")
        result = cls(**{key: enabled(value) for key, value in FLAGS.items()},
                     visible_window=int(raw) if raw else None)
        result.validate()
        return result

    def validate(self, physical_window=64):
        for key in FLAGS:
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be bool")
        if self.paged_kv_affine and not self.fa4_query_staging:
            raise ValueError("paged_kv_affine requires the pinned fa4_query_staging overlay")
        if self.visible_window is not None and not self.allow_approximate:
            raise ValueError("Approximate options require allow_approximate=True")
        if self.visible_window is not None:
            if type(self.visible_window) is not int or not 1 <= self.visible_window <= physical_window:
                raise ValueError("visible_window must be an integer in [1, physical_window]")

    def activate(self):
        """Fix the options for one process before optional backend imports."""
        global _ACTIVE
        self.validate()
        stale = sorted(k for k, v in os.environ.items() if k.startswith("LINGBOT_CANDIDATE") and v not in ("", "0"))
        if stale:
            raise ValueError(f"Clear legacy research flags before using ThorOptions: {stale}")
        if _ACTIVE is not None and _ACTIVE != self:
            raise RuntimeError("Thor runtime options are process-immutable; start a fresh process")
        if _ACTIVE is not None:
            if ThorOptions.from_env() != self:
                raise RuntimeError("Thor environment changed after activation; start a fresh process")
            return
        for key, env in FLAGS.items():
            os.environ[env] = "1" if getattr(self, key) else "0"
        if self.visible_window is None:
            os.environ.pop(VISIBLE_ENV, None)
        else:
            os.environ[VISIBLE_ENV] = str(self.visible_window)
        _ACTIVE = self

    def to_dict(self):
        return asdict(self)
