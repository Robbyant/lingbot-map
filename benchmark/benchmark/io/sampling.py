"""Sampling metadata I/O utilities."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def write_sampling_json(
    sampling_file: Path,
    frame_ids: List[int],
    sampling_config: Optional[Dict[str, Any]] = None
) -> None:
    """Write sampling.json file with frame IDs and metadata.

    Args:
        sampling_file: Output file path
        frame_ids: List of sampled frame IDs
        sampling_config: Sampling configuration dict (will be stored as metadata)

    Note:
        File format:
        {
            "num_frames": <count>,
            "frames": [<sorted frame IDs>],
            "sampling": {<original config>}
        }
    """
    data = {
        'num_frames': len(frame_ids),
        'frames': sorted(frame_ids),
    }

    if sampling_config:
        data['sampling'] = sampling_config

    with open(sampling_file, 'w') as f:
        json.dump(data, f, indent=2)


def read_sampling_json(sampling_file: Path) -> Dict[str, Any]:
    """Read sampling.json file.

    Args:
        sampling_file: Sampling metadata file path

    Returns:
        Dictionary with 'num_frames', 'frames', and optional 'sampling' keys

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(sampling_file, 'r') as f:
        data = json.load(f)

    return data
