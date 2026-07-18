#!/usr/bin/env python3
"""Build a self-contained static report from benchmark eval outputs."""

import argparse
import logging
import sys
from pathlib import Path

from benchmark.core.storage import BSSManager
from benchmark.report.generator import ReportGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark static report")
    parser.add_argument("--workspace", type=Path, required=True, help="Path to BSS workspace directory")
    parser.add_argument("--output", type=Path, help="Output directory for report; defaults to <workspace>/report")
    parser.add_argument("--dataset", help="Generate report for a specific dataset only")
    parser.add_argument("--clean", action="store_true", help="Remove existing report directory first")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        workspace = args.workspace
        if not workspace.exists():
            raise FileNotFoundError(f"Workspace directory not found: {workspace}")

        output_dir = Path(args.output) if args.output else workspace / "report"
        generator = ReportGenerator(BSSManager(workspace), output_dir)
        plan = generator.generate(dataset=args.dataset, clean=args.clean)

        dataset_count = len(plan.manifest.get("datasets", []))
        print("Report generation complete.")
        print(f"Report location: {output_dir}")
        print(f"Datasets included: {dataset_count}")
        print("To view: cd to the report directory and run `python -m http.server 8000`.")

    except Exception as exc:
        logger.exception("Failed to generate report: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
