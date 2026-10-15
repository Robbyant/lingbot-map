"""Compare streaming synthetic outputs with a stable baseline repeat."""
import argparse
import json
from pathlib import Path

from tools.thor_legacy_1005.records import compare_records, validate_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        result = compare_records(*[
            json.loads(path.read_text())
            for path in (args.baseline, args.optimized, args.repeat)
        ])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        result = {"status": "invalid_evidence", "reason": str(error)}
    text = json.dumps(result, indent=2) + "\n"
    print(text, end="")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
