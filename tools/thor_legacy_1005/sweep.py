"""Validate first, then run a fresh-process forward/reverse 1000-frame ablation."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from .records import RUNNER_CONTRACT_ID, compare_records, validate_record

ROWS = (
    ("all_off", ()),
    ("mlp_qkv_cache", ("--cache-mlp-weights", "--cache-qkv-weights")),
    ("projection_append", ("--cache-mlp-weights", "--cache-qkv-weights",
                           "--cache-projection-weights", "--merge-kv-append")),
    ("query_staging", ("--cache-mlp-weights", "--cache-qkv-weights",
                       "--cache-projection-weights", "--merge-kv-append", "--fa4-query-staging")),
    ("single_page_affine", ("--cache-mlp-weights", "--cache-qkv-weights",
                           "--cache-projection-weights", "--merge-kv-append",
                           "--fa4-query-staging", "--paged-kv-affine")),
)
ROOT = Path(__file__).resolve().parents[2]


def clean_env():
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LINGBOT_CANDIDATE", "LINGBOT_THOR"))}
    for name in ("CUBLAS_WORKSPACE_CONFIG", "TORCH_LOGS", "PYTHONPATH"):
        env.pop(name, None)
    env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run(directory, name, flags, frames, benchmark=False):
    output, log = directory / f"{name}.json", directory / f"{name}.log"
    if output.exists() or log.exists():
        raise ValueError(f"Use a new output directory; run already exists: {name}")
    command = [sys.executable, "-u", "-m", "tools.thor_legacy_1005.run",
               "--frames", str(frames), "--out", str(output), *flags]
    if benchmark:
        command.append("--benchmark")
    write_new(directory / f"{name}.command.json", {"argv": command, "environment_policy": "fresh_process_explicit_flags"})
    print(f"START {name} ({frames} frames)", flush=True)
    with log.open("x") as stream:
        subprocess.run(command, cwd=ROOT, env=clean_env(), stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    record = json.loads(output.read_text())
    validate_record(record, correctness=not benchmark)
    print(f"PASS {name}", flush=True)
    return record


def load_validation_record(path, expected_frames):
    record = json.loads(path.read_text())
    validate_record(record, correctness=True)
    if record["frames"] != expected_frames:
        raise RuntimeError(
            f"Validation frame-count mismatch in {path.name}: "
            f"expected {expected_frames}, got {record['frames']}"
        )
    if record["runner_contract_id"] != RUNNER_CONTRACT_ID:
        raise RuntimeError(f"Validation runner contract mismatch in {path.name}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "benchmark"))
    parser.add_argument("--scope", choices=("endpoints", "ablation"), default="ablation")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--validation-frames", type=int, default=200)
    args = parser.parse_args()
    if args.validation_frames < 19:
        parser.error("Validation needs at least 19 frames")
    directory = args.out_dir.resolve()
    rows = (ROWS[0], ROWS[-1]) if args.scope == "endpoints" else ROWS
    validation = directory / "validation"
    if args.mode == "validate":
        run(validation, "preflight_discard", (), args.validation_frames)
        baseline = run(validation, "all_off", (), args.validation_frames)
        candidates = {name: run(validation, name, flags, args.validation_frames) for name, flags in rows[1:]}
        repeat = run(validation, "all_off_repeat", (), args.validation_frames)
        checks = {name: compare_records(baseline, record, repeat) for name, record in candidates.items()}
        write_new(validation / "comparison.json", checks)
        if any(check["status"] != "pass" for check in checks.values()):
            raise RuntimeError("Output comparison failed; do not benchmark")
        return

    # Revalidate the source records, not a user-editable PASS field.
    baseline = load_validation_record(validation / "all_off.json", args.validation_frames)
    repeat = load_validation_record(validation / "all_off_repeat.json", args.validation_frames)
    for name, flags in rows[1:]:
        candidate = load_validation_record(validation / f"{name}.json", args.validation_frames)
        expected = {flag[2:].replace("-", "_") for flag in flags}
        actual = {key for key, value in candidate["options"].items() if value is True}
        if actual != expected or compare_records(baseline, candidate, repeat)["status"] != "pass":
            raise RuntimeError(f"No passing validation for {name}")
    records = []
    for order, selected in (("forward", rows), ("reverse", tuple(reversed(rows)))):
        for name, flags in selected:
            record = run(directory / "timing", f"{name}_{order}", flags, 1000, benchmark=True)
            records.append({"configuration": name, "order": order, **record["timing"]})
    aggregates = [{"configuration": name, "fps": 1000 / statistics.median(
        r["host_per_requested_frame"] for r in records if r["configuration"] == name)} for name, _ in rows]
    write_new(directory / "timing" / "summary.json", {"rows": aggregates, "runs": records})
    for row in aggregates:
        print(f"{row['configuration']}: {row['fps']:.3f} FPS", flush=True)


if __name__ == "__main__":
    main()
