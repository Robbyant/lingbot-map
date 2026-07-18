"""Configuration loading and parameter routing.

Manages YAML configuration files and routes parameters to appropriate
components (datasets, methods, etc.).

Layout (auto-discovered):
  configs/
    base.yaml           -- workspace + global evaluation defaults
                           also contains optional 'datasets' and 'methods' selection lists
    datasets/*.yaml     -- flat config (no top-level key); filename stem is the config name
    methods/*.yaml      -- flat config (no top-level key); filename stem is the config name
"""

import yaml
from pathlib import Path
from typing import Any, Optional, List, Dict


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge two dicts; override values take precedence.

    Args:
        base:     Base dictionary
        override: Override dictionary (values take precedence over base)

    Returns:
        New merged dictionary; inputs are not modified.
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


class ConfigManager:
    """Manages configuration loading and parameter routing.

    Supports the split-directory layout:
      - configs/base.yaml         : workspace + evaluation defaults
                                    + optional 'datasets' / 'methods' selection lists
      - configs/datasets/*.yaml   : dataset configs (auto-discovered, flat — no top-level key,
                                    filename stem is the config name)
      - configs/methods/*.yaml    : method configs  (auto-discovered, flat — no top-level key,
                                    filename stem is the config name)

    Each dataset/method yaml may contain an ``evaluation`` block that overrides
    the global defaults for that specific dataset or method.  Merge order:

        base defaults  →  dataset.evaluation  →  method.evaluation
    """

    def __init__(self, config_path: Path):
        """Initialize configuration manager.

        Args:
            config_path: Path to base YAML configuration file.
        """
        self.config_path = Path(config_path)
        self._base_dir = self.config_path.parent

        # Raw config data
        self._base: Dict[str, Any] = {}
        self._datasets: Dict[str, Any] = {}   # config_name -> raw dict
        self._methods: Dict[str, Any] = {}    # config_name -> raw dict

        self._load_config()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_yaml(self, path: Path) -> dict:
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return data or {}

    def _load_config(self) -> None:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        self._base = self._load_yaml(self.config_path)

        # Auto-discover configs/datasets/*.yaml (flat — filename stem is the config name)
        datasets_dir = self._base_dir / 'datasets'
        if datasets_dir.is_dir():
            for path in sorted(datasets_dir.glob('*.yaml')):
                data = self._load_yaml(path)
                self._datasets[path.stem] = data

        # Auto-discover configs/methods/*.yaml (flat — filename stem is the config name)
        methods_dir = self._base_dir / 'methods'
        if methods_dir.is_dir():
            for path in sorted(methods_dir.glob('*.yaml')):
                data = self._load_yaml(path)
                self._methods[path.stem] = data

    # ------------------------------------------------------------------
    # Public API — selection lists
    # ------------------------------------------------------------------

    def get_selected_dataset_names(self) -> List[str]:
        """Get selected dataset names from the 'datasets' list in base config.

        Returns an empty list if 'datasets' is absent or empty.
        Raises KeyError if any listed name is not a known dataset config.
        """
        selected = self._base.get('datasets', None)
        if not selected:
            return []
        unknown = [n for n in selected if n not in self._datasets]
        if unknown:
            raise KeyError(
                f"Unknown dataset(s) in base config 'datasets' list: {unknown}. "
                f"Available: {sorted(self._datasets.keys())}"
            )
        return list(selected)

    def get_selected_method_names(self) -> List[str]:
        """Get selected method names from the 'methods' list in base config.

        Returns an empty list if 'methods' is absent or empty.
        Raises KeyError if any listed name is not a known method config.
        """
        selected = self._base.get('methods', None)
        if not selected:
            return []
        unknown = [n for n in selected if n not in self._methods]
        if unknown:
            raise KeyError(
                f"Unknown method(s) in base config 'methods' list: {unknown}. "
                f"Available: {sorted(self._methods.keys())}"
            )
        return list(selected)

    # ------------------------------------------------------------------
    # Public API — config access
    # ------------------------------------------------------------------

    def get_workspace(self) -> str:
        """Get workspace path from configuration."""
        if 'workspace' not in self._base:
            raise KeyError("'workspace' must be defined in base config")
        return self._base['workspace']

    def get_dataset_config(self, dataset_config_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset.

        Returns:
            Dictionary with:
            - 'dataset_class': Dataset module name (e.g., 'seven_scenes')
            - 'params':        __init__ kwargs (underscore prefix removed)
            - 'sampling':      Sampling config dict or None
            - 'evaluation':    Raw evaluation override dict (may be empty)
        """
        if dataset_config_name not in self._datasets:
            raise KeyError(
                f"Dataset config '{dataset_config_name}' not found. "
                f"Available: {sorted(self._datasets.keys())}"
            )

        dataset_cfg = self._datasets[dataset_config_name].copy()

        if 'dataset' not in dataset_cfg:
            raise KeyError(f"'dataset' field required in dataset config '{dataset_config_name}'")

        dataset_class_name = dataset_cfg.pop('dataset')
        sampling_config = dataset_cfg.pop('sampling', None)
        eval_override = dataset_cfg.pop('evaluation', {})

        params = {}
        for key, value in dataset_cfg.items():
            if key.startswith('_'):
                params[key[1:]] = value
            else:
                params[key] = value

        return {
            'dataset_class': dataset_class_name,
            'params': params,
            'sampling': sampling_config,
            'evaluation': eval_override,
        }

    def get_method_config(self, method_config_name: str) -> Dict[str, Any]:
        """Get configuration for a specific method.

        Returns:
            Dictionary with:
            - 'method_class': Method module name (e.g., 'da3')
            - 'params':       __init__ kwargs (underscore prefix removed, env excluded)
            - 'env':          conda env name or None
            - 'evaluation':   Raw evaluation override dict (may be empty)
        """
        if method_config_name not in self._methods:
            raise KeyError(
                f"Method config '{method_config_name}' not found. "
                f"Available: {sorted(self._methods.keys())}"
            )

        method_cfg = self._methods[method_config_name].copy()

        if 'model' not in method_cfg:
            raise KeyError(f"'model' field required in method config '{method_config_name}'")

        method_class_name = method_cfg.pop('model')
        env = method_cfg.pop('env', None)
        eval_override = method_cfg.pop('evaluation', {})

        params = {}
        for key, value in method_cfg.items():
            if key.startswith('_'):
                params[key[1:]] = value
            else:
                params[key] = value

        return {
            'method_class': method_class_name,
            'params': params,
            'env': env,
            'evaluation': eval_override,
        }

    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get global evaluation defaults from base config."""
        return self._base.get('evaluation', {})

    def get_merged_evaluation_config(
        self, dataset_config_name: str, method_config_name: str = None
    ) -> Dict[str, Any]:
        """Get evaluation config merged for a specific dataset and method.

        Merge order (later overrides earlier):
          1. base.yaml  evaluation  (global defaults)
          2. dataset yaml evaluation block
          3. method yaml evaluation block

        Args:
            dataset_config_name: Dataset config key (e.g., '7scenes_s10')
            method_config_name:  Optional method config key (e.g., 'slam3r')

        Returns:
            Merged evaluation config dict.
        """
        merged = dict(self._base.get('evaluation', {}))

        if dataset_config_name in self._datasets:
            ds_eval = self._datasets[dataset_config_name].get('evaluation', {})
            merged = _deep_merge(merged, ds_eval)

        if method_config_name and method_config_name in self._methods:
            m_eval = self._methods[method_config_name].get('evaluation', {})
            merged = _deep_merge(merged, m_eval)

        return merged

    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get raw value from base config."""
        return self._base.get(key, default)

    def has_key(self, key: str) -> bool:
        """Check if base config has a key."""
        return key in self._base
