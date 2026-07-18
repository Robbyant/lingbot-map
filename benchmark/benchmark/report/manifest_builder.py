"""Build static report data from evaluated BSS artifacts."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from benchmark.core.storage import BSSManager


@dataclass(frozen=True)
class ArtifactCopy:
    src: Path
    dest: str


@dataclass
class BuildPlan:
    manifest: dict[str, Any]
    dataset_overviews: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_scenes_indexes: dict[str, dict[str, Any]] = field(default_factory=dict)
    scene_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    method_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifact_copies: list[ArtifactCopy] = field(default_factory=list)


class ManifestBuilder:
    """Scans Layer 1/2/3 eval outputs and prepares static report payloads."""

    LAYER3_CATEGORIES = ("traj", "auc_micro", "auc_macro", "depth", "points")
    LAYER2_CATEGORIES = ("traj", "auc", "depth", "points")
    LAYER1_CATEGORIES = ("traj", "auc", "depth", "points")
    IMAGE_SUFFIXES = {
        ".apng", ".avif", ".gif", ".jpg", ".jpeg", ".png", ".svg", ".webp"
    }
    EXCLUDED_DATASET_DIRS = {"report", "logs", ".git"}

    def __init__(self, bss_manager: BSSManager):
        self.bss = bss_manager
        self.workspace = bss_manager.workspace

    def build(self, dataset_filter: Optional[str] = None) -> BuildPlan:
        datasets = [dataset_filter] if dataset_filter else self._list_datasets()
        plan = BuildPlan(
            manifest={
                "version": "1",
                "generated_at": datetime.now().isoformat(),
                "datasets": [],
                "metric_definitions": self._metric_definitions(),
            }
        )

        seen_copies: set[str] = set()
        for dataset in datasets:
            dataset_dir = self.workspace / dataset
            if not dataset_dir.is_dir():
                continue

            dataset_entry = self._build_dataset(dataset, plan, seen_copies)
            if dataset_entry is not None:
                plan.manifest["datasets"].append(dataset_entry)

        return plan

    def _build_dataset(
        self,
        dataset: str,
        plan: BuildPlan,
        seen_copies: set[str],
    ) -> Optional[dict[str, Any]]:
        dataset_dir = self.workspace / dataset
        overview_metrics, overview_sources = self._read_layer_jsons(
            dataset_dir / "eval",
            self.LAYER3_CATEGORIES,
            f"artifacts/{dataset}/eval",
            plan,
            seen_copies,
        )

        scene_entries: list[dict[str, Any]] = []
        all_methods: set[str] = set()
        scene_categories: set[str] = set()

        for scene_dir in sorted(dataset_dir.iterdir()):
            if not scene_dir.is_dir() or scene_dir.name == "eval":
                continue
            scene_entry = self._build_scene(dataset, scene_dir.name, plan, seen_copies)
            if scene_entry is None:
                continue
            scene_entries.append(scene_entry)
            all_methods.update(scene_entry["methods"])
            scene_categories.update(scene_entry["available_categories"])

        if not overview_metrics and not scene_entries:
            return None

        dataset_overview_path = f"data/datasets/{dataset}/overview.json"
        dataset_scenes_path = f"data/datasets/{dataset}/scenes.json"

        plan.dataset_overviews[dataset_overview_path] = {
            "dataset": dataset,
            "metrics": overview_metrics,
            "sources": overview_sources,
        }
        plan.dataset_scenes_indexes[dataset_scenes_path] = {
            "dataset": dataset,
            "scenes": scene_entries,
        }

        metric_categories = sorted(set(overview_metrics.keys()) | scene_categories)
        return {
            "id": dataset,
            "label": dataset,
            "overview_path": dataset_overview_path,
            "scenes_path": dataset_scenes_path,
            "methods": sorted(all_methods),
            "metric_categories": metric_categories,
        }

    def _build_scene(
        self,
        dataset: str,
        scene: str,
        plan: BuildPlan,
        seen_copies: set[str],
    ) -> Optional[dict[str, Any]]:
        scene_dir = self.workspace / dataset / scene
        summary_metrics, summary_sources = self._read_layer_jsons(
            scene_dir / "eval",
            self.LAYER2_CATEGORIES,
            f"artifacts/{dataset}/{scene}/eval",
            plan,
            seen_copies,
        )

        method_entries: dict[str, dict[str, str]] = {}
        method_names: set[str] = set()
        detail_categories: set[str] = set()

        for method_dir in sorted(scene_dir.iterdir()):
            if not method_dir.is_dir() or method_dir.name in {"gt", "eval"}:
                continue
            detail_path, categories = self._build_method(
                dataset,
                scene,
                method_dir.name,
                plan,
                seen_copies,
            )
            if detail_path is None:
                continue
            method_entries[method_dir.name] = {"detail_path": detail_path}
            method_names.add(method_dir.name)
            detail_categories.update(categories)

        available_categories = sorted(set(summary_metrics.keys()) | detail_categories)
        if not summary_metrics and not method_entries:
            return None

        summary_path = f"data/datasets/{dataset}/scenes/{scene}/summary.json"
        plan.scene_summaries[summary_path] = {
            "dataset": dataset,
            "scene": scene,
            "metrics": summary_metrics,
            "methods": method_entries,
            "sources": summary_sources,
        }

        return {
            "id": scene,
            "label": scene,
            "summary_path": summary_path,
            "methods": sorted(method_names),
            "available_categories": available_categories,
        }

    def _build_method(
        self,
        dataset: str,
        scene: str,
        method: str,
        plan: BuildPlan,
        seen_copies: set[str],
    ) -> tuple[Optional[str], set[str]]:
        method_eval_dir = self.workspace / dataset / scene / method / "eval"
        if not method_eval_dir.is_dir():
            return None, set()

        metrics: dict[str, Any] = {}
        json_artifacts: dict[str, str] = {}
        image_artifacts: dict[str, list[str]] = {}

        for category in self.LAYER1_CATEGORIES:
            json_path = method_eval_dir / f"{category}.json"
            if json_path.exists():
                raw = self._read_json(json_path)
                metrics[category] = self._sanitize_auc(raw) if category == "auc" else self._jsonable(raw)
                dest = f"artifacts/{dataset}/{scene}/{method}/eval/{category}.json"
                self._add_artifact_copy(plan, seen_copies, json_path, dest)
                json_artifacts[category] = dest

            artifact_dir = method_eval_dir / category
            images = self._collect_artifact_dir(
                artifact_dir,
                f"artifacts/{dataset}/{scene}/{method}/eval/{category}",
                plan,
                seen_copies,
            )
            if images:
                image_artifacts[category] = images

        if not metrics and not json_artifacts and not image_artifacts:
            return None, set()

        detail_path = f"data/datasets/{dataset}/scenes/{scene}/methods/{method}.json"
        plan.method_details[detail_path] = {
            "dataset": dataset,
            "scene": scene,
            "method": method,
            "metrics": metrics,
            "artifacts": {
                "json": json_artifacts,
                "images": image_artifacts,
            },
        }
        return detail_path, set(metrics.keys()) | set(image_artifacts.keys())

    def _read_layer_jsons(
        self,
        eval_dir: Path,
        categories: tuple[str, ...],
        artifact_prefix: str,
        plan: BuildPlan,
        seen_copies: set[str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        metrics: dict[str, Any] = {}
        sources: dict[str, str] = {}

        if not eval_dir.is_dir():
            return metrics, sources

        for category in categories:
            path = eval_dir / f"{category}.json"
            if not path.exists():
                continue
            raw = self._read_json(path)
            metrics[category] = self._sanitize_auc(raw) if category.startswith("auc") else self._jsonable(raw)
            dest = f"{artifact_prefix}/{category}.json"
            self._add_artifact_copy(plan, seen_copies, path, dest)
            sources[category] = dest

        return metrics, sources

    def _collect_artifact_dir(
        self,
        src_dir: Path,
        dest_prefix: str,
        plan: BuildPlan,
        seen_copies: set[str],
    ) -> list[str]:
        if not src_dir.is_dir():
            return []

        image_paths: list[str] = []
        for src in sorted(path for path in src_dir.rglob("*") if path.is_file()):
            relative = src.relative_to(src_dir).as_posix()
            dest = f"{dest_prefix}/{relative}"
            self._add_artifact_copy(plan, seen_copies, src, dest)
            if src.suffix.lower() in self.IMAGE_SUFFIXES:
                image_paths.append(dest)
        return sorted(image_paths)

    def _add_artifact_copy(
        self,
        plan: BuildPlan,
        seen_copies: set[str],
        src: Path,
        dest: str,
    ) -> None:
        if dest in seen_copies:
            return
        seen_copies.add(dest)
        plan.artifact_copies.append(ArtifactCopy(src=src, dest=dest))

    def _read_json(self, path: Path) -> Any:
        with open(path, "r") as f:
            return json.load(f)

    def _sanitize_auc(self, payload: Any) -> Any:
        stripped = self._remove_keys(payload, {"rError", "tError"})
        return self._jsonable(stripped)

    def _remove_keys(self, payload: Any, keys_to_remove: set[str]) -> Any:
        if isinstance(payload, dict):
            return {
                key: self._remove_keys(value, keys_to_remove)
                for key, value in payload.items()
                if key not in keys_to_remove
            }
        if isinstance(payload, list):
            return [self._remove_keys(item, keys_to_remove) for item in payload]
        return payload

    def _jsonable(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            return {str(key): self._jsonable(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._jsonable(value) for value in payload]
        if isinstance(payload, tuple):
            return [self._jsonable(value) for value in payload]
        if isinstance(payload, float):
            return payload if math.isfinite(payload) else None
        return copy.deepcopy(payload)

    def _list_datasets(self) -> list[str]:
        if not self.workspace.exists():
            return []
        datasets = []
        for item in sorted(self.workspace.iterdir()):
            if not item.is_dir() or item.name in self.EXCLUDED_DATASET_DIRS:
                continue
            has_eval = (item / "eval").is_dir()
            has_scene = any(
                child.is_dir() and child.name != "eval"
                for child in item.iterdir()
            )
            if has_eval or has_scene:
                datasets.append(item.name)
        return datasets

    def _metric_definitions(self) -> dict[str, dict[str, str]]:
        definitions: dict[str, dict[str, str]] = {
            "traj.ate": {"label": "ATE", "direction": "lower"},
            "traj.rpe_trans": {"label": "RPE Trans", "direction": "lower"},
            "traj.rpe_rot": {"label": "RPE Rot", "direction": "lower"},
            "depth.abs_rel": {"label": "Abs Rel", "direction": "lower"},
            "depth.sq_rel": {"label": "Sq Rel", "direction": "lower"},
            "depth.rmse": {"label": "RMSE", "direction": "lower"},
            "depth.log_rmse": {"label": "Log RMSE", "direction": "lower"},
            "depth.delta_1_25": {"label": "δ<1.25", "direction": "higher"},
            "depth.delta_1_25_2": {"label": "δ<1.25²", "direction": "higher"},
            "depth.delta_1_25_3": {"label": "δ<1.25³", "direction": "higher"},
            "points.chamfer": {"label": "Chamfer", "direction": "lower"},
            "points.accuracy": {"label": "Accuracy", "direction": "lower"},
            "points.completeness": {"label": "Completeness", "direction": "lower"},
            "points.precision": {"label": "Precision", "direction": "higher"},
            "points.recall": {"label": "Recall", "direction": "higher"},
            "points.f1": {"label": "F1", "direction": "higher"},
        }
        for threshold in (3, 5, 15, 30):
            definitions[f"auc.AUC_{threshold}"] = {
                "label": f"AUC@{threshold}",
                "direction": "higher",
            }
            definitions[f"auc.Racc_{threshold}"] = {
                "label": f"Racc@{threshold}",
                "direction": "higher",
            }
            definitions[f"auc.Tacc_{threshold}"] = {
                "label": f"Tacc@{threshold}",
                "direction": "higher",
            }
        return definitions
