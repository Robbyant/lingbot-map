"""BSS (Benchmark Storage Structure) storage classes.

Two-layer storage abstraction:
  - BSSArtifact: single-directory path template (all fixed paths, completion, eval)
  - BSSManager:  workspace-level navigation (hierarchy, scene/method listing)

Layout:
    workspace/
    └── {dataset_name}/
        └── {scene_safe}/          # '/' replaced by '_' in scene name
            ├── gt/                # BSSArtifact for ground truth
            └── {method_name}/     # BSSArtifact for method output
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class BSSArtifact:
    """Path template for a single BSS directory (gt or method output).

    All well-known paths are fixed properties defined at __init__ time,
    making the complete layout inspectable and easy to modify in one place.

    Layout:  workspace / {dataset} / {scene} / {gt|method} /
    """

    def __init__(self, root: Path):
        self.root = Path(root)

        # ── frame data folders ────────────────────────────────────────────
        self.rgb_dir        = self.root / 'rgb'
        self.depth_dir      = self.root / 'depth'
        self.points_dir     = self.root / 'points'
        self.confidence_dir = self.root / 'confidence'
        self.mask_dir       = self.root / 'mask'

        # ── frame data files (stored as files, not folders) ───────────────
        self.traj_file       = self.root / 'traj.txt'
        self.intrinsics_file = self.root / 'intrinsics.txt'

        # ── metadata / markers ────────────────────────────────────────────
        self.complete_file = self.root / '.complete.json'
        self.sampling_file = self.root / 'sampling.json'

        # ── method-level artifacts ────────────────────────────────────────
        self.global_points_file  = self.root / 'points.ply'

        # ── evaluation directory ──────────────────────────────────────────
        self.eval_dir = self.root / 'eval'
        self.traj_transform_file = self.eval_dir / 'traj_transform.txt'

        # ── evaluation result files ───────────────────────────────────────
        self.eval_traj_file   = self.eval_dir / 'traj.json'
        self.eval_auc_file    = self.eval_dir / 'auc.json'
        self.eval_depth_file  = self.eval_dir / 'depth.json'
        self.eval_points_file = self.eval_dir / 'points.json'

        # ── visualization directories ─────────────────────────────────────
        # Internal filenames within these dirs are managed by the evaluators.
        self.vis_traj_dir   = self.eval_dir / 'traj'
        self.vis_auc_dir    = self.eval_dir / 'auc'
        self.vis_depth_dir  = self.eval_dir / 'depth'
        self.vis_points_dir = self.eval_dir / 'points'

    # ── existence ─────────────────────────────────────────────────────────

    def exists(self) -> bool:
        return self.root.exists()

    def has_frame_key(self, key: str) -> bool:
        """Check whether a frame-level data key is present in this directory."""
        if key == 'pose':       return self.traj_file.exists()
        if key == 'intrinsics': return self.intrinsics_file.exists()
        return (self.root / key).exists()

    # ── completion management ─────────────────────────────────────────────

    def is_complete(self) -> bool:
        return self.complete_file.exists()

    def mark_complete(self, metadata: dict = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        data = {
            "completed_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        with open(self.complete_file, 'w') as f:
            json.dump(data, f, indent=2)

    def mark_incomplete(self) -> None:
        if self.complete_file.exists():
            self.complete_file.unlink()

    def read_metadata(self) -> Optional[dict]:
        if not self.complete_file.exists():
            return None
        try:
            with open(self.complete_file) as f:
                return json.load(f).get("metadata", {})
        except (json.JSONDecodeError, IOError):
            return None

    def clear_incomplete(self) -> None:
        """Remove directory contents only if not yet marked complete."""
        if self.root.exists() and not self.is_complete():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)

    def clear_directory(self) -> None:
        """Remove all directory contents unconditionally (force mode)."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ── eval results ─────────────────────────────────────────────────────

    def has_eval(self, eval_type: str) -> bool:
        """Check if evaluation result exists for given type."""
        return (self.eval_dir / f'{eval_type}.json').exists()

    def save_eval(self, eval_type: str, results: dict) -> None:
        """Save evaluation results to eval/{eval_type}.json."""
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.eval_dir / f'{eval_type}.json'
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True, default=_json_default)

    def load_eval(self, eval_type: str) -> Optional[dict]:
        """Load evaluation results from eval/{eval_type}.json. Returns None if not found."""
        filepath = self.eval_dir / f'{eval_type}.json'
        if not filepath.exists():
            return None
        with open(filepath) as f:
            return json.load(f)

    def clear_eval(self) -> None:
        """Delete entire eval/ directory."""
        if self.eval_dir.exists():
            shutil.rmtree(self.eval_dir)

    def save_traj_transform(self, T: np.ndarray) -> None:
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            self.traj_transform_file, T, fmt='%.10f',
            header=(
                '4x4 Sim(3) alignment transformation matrix\n'
                'Apply as: p_aligned = T @ p_original (homogeneous coords)'
            )
        )

class BSSManager:
    """Manages BSS hierarchy: workspace / dataset / scene / method.

    Responsible only for: path construction at that level, scene/method listing,
    and scene name sanitization. All per-directory operations go through BSSArtifact.
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def resolve_scene_safe(scene: str) -> str:
        """Sanitize scene name for use in file system (replaces '/' with '_').

        Example:
            >>> BSSManager.resolve_scene_safe("chess/seq-01")
            'chess_seq-01'
        """
        return scene.replace('/', '_')

    def get_artifact(self, dataset_name: str, scene: str,
                     method_name: Optional[str] = None) -> BSSArtifact:
        """Return the BSSArtifact for a GT or method directory.

        Args:
            dataset_name: Dataset config name
            scene:        Scene name (auto-sanitized)
            method_name:  Method name; None returns the GT artifact

        Returns:
            BSSArtifact pointing at workspace/{dataset}/{scene_safe}/{gt|method}/
        """
        safe = self.resolve_scene_safe(scene)
        sub = 'gt' if method_name is None else method_name
        return BSSArtifact(self.workspace / dataset_name / safe / sub)

    def list_scenes(self, dataset_name: str) -> List[str]:
        """List all scenes in a dataset directory (sanitized, flat format)."""
        dataset_dir = self.workspace / dataset_name
        if not dataset_dir.exists():
            return []
        return sorted(d.name for d in dataset_dir.iterdir()
                      if d.is_dir() and d.name != 'eval')

    def list_methods(self, dataset_name: str, scene: str) -> List[str]:
        """List all methods that have output directories for a scene."""
        scene_dir = self.workspace / dataset_name / self.resolve_scene_safe(scene)
        if not scene_dir.exists():
            return []
        return sorted(d.name for d in scene_dir.iterdir()
                      if d.is_dir() and d.name not in ('gt', 'eval'))

    def get_scene_index(self, dataset_name: str, scene: str) -> int:
        """Return the 0-based index of scene in the sorted scene list, or -1."""
        scenes = self.list_scenes(dataset_name)
        safe = self.resolve_scene_safe(scene)
        try:
            return scenes.index(safe)
        except ValueError:
            return -1

    def has_traj_pair(self, dataset_name: str, scene: str,
                      method_name: str) -> bool:
        """Return True if both GT and method traj.txt files exist."""
        return (
            self.get_artifact(dataset_name, scene).traj_file.exists() and
            self.get_artifact(dataset_name, scene, method_name).traj_file.exists()
        )

    # ── Layer 2/3 eval aggregation ────────────────────────────────────────

    def _save_eval_entry(self, filepath: Path, method_name: str, results: dict) -> None:
        """Read-modify-write: read existing json, update method key, write back."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if filepath.exists():
            with open(filepath) as f:
                existing = json.load(f)
        existing[method_name] = results
        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2, sort_keys=True, default=_json_default)

    # ── Layer 2: scene-level cross-method comparison ──

    def get_scene_eval_dir(self, dataset_name: str, scene: str) -> Path:
        """Return workspace/{dataset}/{scene_safe}/eval/"""
        safe = self.resolve_scene_safe(scene)
        return self.workspace / dataset_name / safe / 'eval'

    def save_scene_eval(self, dataset_name: str, scene: str,
                        method_name: str, eval_type: str, results: dict) -> None:
        """Save scene-level cross-method comparison entry."""
        filepath = self.get_scene_eval_dir(dataset_name, scene) / f'{eval_type}.json'
        self._save_eval_entry(filepath, method_name, results)

    # ── Layer 3: dataset-level aggregation ──

    def get_dataset_eval_dir(self, dataset_name: str) -> Path:
        """Return workspace/{dataset}/eval/"""
        return self.workspace / dataset_name / 'eval'

    def save_dataset_eval(self, dataset_name: str, method_name: str,
                          eval_type: str, results: dict) -> None:
        """Save dataset-level aggregation entry."""
        filepath = self.get_dataset_eval_dir(dataset_name) / f'{eval_type}.json'
        self._save_eval_entry(filepath, method_name, results)
