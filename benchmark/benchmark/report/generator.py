"""Static report generator for benchmark eval outputs."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Optional

from benchmark.core.storage import BSSManager
from .manifest_builder import BuildPlan, ManifestBuilder

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates a self-contained static report from Layer 1/2/3 eval outputs."""

    def __init__(self, bss_manager: BSSManager, output_dir: Path):
        self.bss = bss_manager
        self.output_dir = Path(output_dir)
        self.builder = ManifestBuilder(bss_manager)

    def generate(self, dataset: Optional[str] = None, clean: bool = False) -> BuildPlan:
        if clean:
            self.cleanup_old_reports()
        else:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        plan = self.builder.build(dataset_filter=dataset)
        self._write_plan(plan)
        self._copy_static_assets()
        self._write_readme()

        logger.info("Generated static report at %s", self.output_dir)
        return plan

    def generate_all_reports(self, clean: bool = False) -> BuildPlan:
        return self.generate(clean=clean)

    def generate_dataset_report(self, dataset_name: str, clean: bool = False) -> BuildPlan:
        return self.generate(dataset=dataset_name, clean=clean)

    def cleanup_old_reports(self) -> None:
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_plan(self, plan: BuildPlan) -> None:
        self._write_json("data/manifest.json", plan.manifest)

        for relative_path, payload in plan.dataset_overviews.items():
            self._write_json(relative_path, payload)
        for relative_path, payload in plan.dataset_scenes_indexes.items():
            self._write_json(relative_path, payload)
        for relative_path, payload in plan.scene_summaries.items():
            self._write_json(relative_path, payload)
        for relative_path, payload in plan.method_details.items():
            self._write_json(relative_path, payload)

        for item in plan.artifact_copies:
            self._copy_artifact(item.src, item.dest)

    def _write_json(self, relative_path: str, payload: Any) -> None:
        output_path = self.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _copy_artifact(self, src_path: Path, relative_dest: str) -> None:
        output_path = self.output_dir / relative_dest
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, output_path)

    def _copy_static_assets(self) -> None:
        templates_dir = Path(__file__).parent / "templates"
        assets_dir = self.output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        index_src = templates_dir / "index.html"
        if index_src.exists():
            shutil.copy2(index_src, self.output_dir / "index.html")

        for asset_name in ("app.js", "styles.css"):
            src = templates_dir / asset_name
            if src.exists():
                shutil.copy2(src, assets_dir / asset_name)

    def _write_readme(self) -> None:
        readme_path = self.output_dir / "README.md"
        readme_path.write_text(
            "# Benchmark Static Report\n\n"
            "This directory is a self-contained static viewer for benchmark eval outputs.\n\n"
            "## View locally\n\n"
            "```bash\n"
            "python -m http.server 8000\n"
            "```\n\n"
            "Open `http://localhost:8000` from this report directory. The page only "
            "depends on files under `index.html`, `assets/`, `data/`, and `artifacts/`.\n",
            encoding="utf-8",
        )
