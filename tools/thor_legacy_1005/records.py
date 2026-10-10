"""CPU-only validation and timing arithmetic for the streaming benchmark."""
import math
import re
import statistics

from lingbot_map.optimizations.thor.options import ThorOptions

from .contracts import (
    COMPILE_COUNTER_KEYS, PROTOCOL_VERSION,
    validate_capture_depth_contract, validate_replay_contract,
)

RUNNER_CONTRACT_ID = "thor_1005_streaming_protocol_v4"


def require_zero_compile(delta):
    expected = {f"{section}.{name}": 0 for section, name in COMPILE_COUNTER_KEYS}
    if (not isinstance(delta, dict) or delta != expected
            or any(type(value) is not int for value in delta.values())):
        raise ValueError("Missing zero-compile evidence")


def summarize_samples(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values)):
        raise ValueError("Expected finite, positive timing samples")
    ordered = sorted(values)
    position = 0.9 * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return {
        "count": len(values), "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": ordered[low] + (ordered[high] - ordered[low]) * (position - low),
        "min": ordered[0], "max": ordered[-1],
    }


def _strict_shape(value, expected):
    return (type(value) is list and len(value) == len(expected)
            and all(type(actual) is int and actual == wanted
                    for actual, wanted in zip(value, expected, strict=True)))


def _validate_input_signature(signature, frames):
    if not isinstance(signature, dict):
        raise ValueError("Missing input signature")
    pixels = 378 * 518
    expected_stride = [frames * 3 * pixels, 3 * pixels, pixels, 518, 1]
    expected = {
        "source": ([1, frames, 3, 378, 518], None),
        "S8": ([1, 8, 3, 378, 518], 1),
        "S1": ([1, 1, 3, 378, 518], 10),
    }
    if set(signature) != set(expected):
        raise ValueError("Input signature views are incomplete")
    for view, (shape, calls) in expected.items():
        value = signature.get(view)
        if (not isinstance(value, dict) or not _strict_shape(value.get("shape"), shape)
                or not _strict_shape(value.get("stride"), expected_stride)
                or value.get("dtype") != "torch.bfloat16"
                or value.get("device_type") != "cuda"):
            raise ValueError(f"Invalid {view} input signature")
        if calls is not None and (type(value.get("calls")) is not int
                                  or value["calls"] != calls):
            raise ValueError(f"Invalid {view} input signature call count")


def validate_contract(contract):
    expected = {
        "formal_protocol_version": PROTOCOL_VERSION,
        "capture_boundary": "aggregator_and_depth_only",
        "camera_execution": "uncaptured_original_dynamic_history",
        "temporal_positions": "original_rope_copied_each_frame",
        "formal_scale_no_recompile": True,
        "formal_warm_no_recompile": True,
        "scale_matches_prewarm": True,
        "warm_frames_match_prewarm": True,
    }
    if not isinstance(contract, dict):
        raise ValueError("Missing streaming capture contract")
    for key, value in expected.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError(f"Wrong or missing streaming contract: {key}")
    for key in ("formal_scale_compile_counter_delta", "formal_warm_compile_counter_delta",
                "replay_compile_counter_delta"):
        require_zero_compile(contract.get(key))
    for key in ("steady_state_prime", "cuda_graph_capture"):
        phase = contract.get(key)
        if (not isinstance(phase, dict) or phase.get("depth_impl") != "compiled"
                or phase.get("deterministic_algorithms") is not True):
            raise ValueError(f"Invalid streaming {key} contract")
    validate_capture_depth_contract(contract.get("capture_depth_contract"))
    signature = contract.get("input_signature")
    source = signature.get("source") if isinstance(signature, dict) else None
    shape = source.get("shape") if isinstance(source, dict) else None
    if not isinstance(shape, list) or len(shape) != 5:
        raise ValueError("Missing complete streaming input signature")
    frames = shape[1]
    if type(frames) is not int or frames < 19:
        raise ValueError("Invalid streaming input frame count")
    _validate_input_signature(signature, frames)
    validate_replay_contract(contract, frames)


def _require_finite_samples(values, label):
    try:
        summarize_samples(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label} samples") from error


def _validate_raw_result(result, frames):
    if not isinstance(result, (list, tuple)) or len(result) != 10:
        raise ValueError("Missing complete streaming result")
    gpu, scale, scale_gpu, warm, warm_gpu, scale_wall, warm_wall, wall = result[:8]
    _require_finite_samples(gpu, "GPU replay")
    _require_finite_samples(wall, "host replay")
    if (type(scale) is not int or type(warm) is not int
            or (scale, warm) != (8, 10)
            or len(gpu) != frames - 18 or len(wall) != len(gpu)):
        raise ValueError("Samples do not match 8 scale + 10 warm + remaining replays")
    for value, label in ((scale_gpu, "scale GPU"), (warm_gpu, "warm GPU"),
                         (scale_wall, "scale host"), (warm_wall, "warm host")):
        _require_finite_samples([value], label)
    phase_digests = result[8]
    if not isinstance(phase_digests, dict):
        raise ValueError("Missing phase output digests")
    for phase, count in (("scale", 8), ("warm_tail", 1), ("replay_tail", 1)):
        value = phase_digests.get(phase)
        if not isinstance(value, dict):
            raise ValueError(f"Missing {phase} output digests")
        for name in ("pose_enc", "depth"):
            digest = value.get(name)
            shape = [1, count, 9] if name == "pose_enc" else [1, count, 378, 518, 1]
            if (not isinstance(digest, dict) or not _strict_shape(digest.get("shape"), shape)
                    or digest.get("dtype") != "torch.float32"
                    or digest.get("finite") is not True
                    or not isinstance(digest.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest.get("sha256", ""))):
                raise ValueError(f"Invalid {phase} {name} output digest")
    contract = result[9]
    validate_contract(contract)
    _validate_input_signature(contract.get("input_signature"), frames)


def summarize_timing(result, frames):
    if type(frames) is not int or frames < 19:
        raise ValueError("Expected at least 19 frames and complete capture evidence")
    _validate_raw_result(result, frames)
    gpu, scale, scale_gpu, warm, warm_gpu, scale_wall, warm_wall, wall = result[:8]
    host_per_frame = (scale_wall + warm_wall + sum(wall)) / frames
    gpu_per_frame = (scale_gpu + warm_gpu + sum(gpu)) / frames
    if not math.isfinite(host_per_frame) or not math.isfinite(gpu_per_frame):
        raise ValueError("Timing totals are not finite")
    return {
        "unit": "ms", "fps": 1000 / host_per_frame,
        "host_per_requested_frame": host_per_frame,
        "gpu_per_requested_frame": gpu_per_frame,
        "host_replay": summarize_samples(wall), "gpu_replay": summarize_samples(gpu),
        "setup_compile_capture_excluded": True,
        "host_replay_includes": [
            "input_update", "static_input_copy", "position_update", "cache_preparation",
            "aggregator_graph_replay", "dynamic_camera", "depth_graph_replay", "synchronize",
        ],
        "output_collection_excluded": True,
    }


def validate_record(record, *, correctness=False):
    if not isinstance(record, dict):
        raise ValueError("Missing record")
    required = ("runner_contract_id", "frames", "image_shape", "tokens_per_frame", "seed", "checkpoint",
                "initialization", "scale_frames", "warm_frames", "replay_frames",
                "options", "result")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Missing record fields: {missing}")
    if record["runner_contract_id"] != RUNNER_CONTRACT_ID:
        raise ValueError("Record was generated by an incompatible runner contract")
    frames = record["frames"]
    if type(frames) is not int or frames < 19:
        raise ValueError("Invalid frame count")
    if (not _strict_shape(record["image_shape"], [378, 518])
            or type(record["tokens_per_frame"]) is not int
            or record["tokens_per_frame"] != 1005):
        raise ValueError("Expected the 378x518, 1005-token workload")
    if (record.get("checkpoint") is not None or type(record.get("seed")) is not int
            or record.get("seed") != 42
            or record.get("initialization") != "cuda_fp32_input_to_bf16_then_random_model"):
        raise ValueError("Wrong initialization protocol")
    if (type(record["scale_frames"]) is not int or type(record["warm_frames"]) is not int
            or type(record["replay_frames"]) is not int
            or (record["scale_frames"], record["warm_frames"], record["replay_frames"])
            != (8, 10, frames - 18)):
        raise ValueError("Wrong phase accounting")
    if not isinstance(record["options"], dict):
        raise ValueError("Missing Thor options")
    try:
        options = ThorOptions(**record["options"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid Thor options") from error
    options.validate()
    _validate_raw_result(record["result"], frames)
    state = record.get("model_state", {})
    if not isinstance(state, dict):
        raise ValueError("Missing model-state checks")
    for key in ("original_fp32_parameters_intact", "state_dict_keys_unchanged"):
        if state.get(key) is not True:
            raise ValueError(f"Missing model-state check: {key}")
    if not correctness:
        if record.get("kind") != "synthetic_profile" or record.get("per_frame_output_collection") is not False:
            raise ValueError("Expected a timing run without per-frame output collection")
        calculated = summarize_timing(record["result"], frames)
        if record.get("timing") != calculated:
            raise ValueError("Timing summary differs from raw samples")
        return
    if record.get("kind") != "synthetic_correctness_smoke" or record.get("per_frame_output_collection") is not True:
        raise ValueError("Expected a correctness run with per-frame output collection")
    if options.allow_approximate or options.visible_window is not None:
        raise ValueError("Approximate windows are excluded from lossless comparison")
    rows = record.get("outputs")
    if not isinstance(rows, list):
        raise ValueError("Missing output rows")
    expected_frames = [0, *range(8, frames)]
    observed_frames = [row.get("frame_index") if isinstance(row, dict) else None
                       for row in rows]
    if (len(observed_frames) != len(expected_frames)
            or any(type(actual) is not int for actual in observed_frames)
            or observed_frames != expected_frames):
        raise ValueError("Missing, duplicate, or out-of-order output frame")
    for row in rows:
        count = 8 if row["frame_index"] == 0 else 1
        for name, shape in (("pose_enc", [1, count, 9]), ("depth", [1, count, 378, 518, 1])):
            value = row.get(name)
            if (not isinstance(value, dict) or not _strict_shape(value.get("shape"), shape)
                    or value.get("dtype") != "torch.float32"
                    or value.get("finite") is not True
                    or not isinstance(value.get("sha256"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
                raise ValueError(f"Invalid {name} output at frame {row['frame_index']}")


def compare_records(baseline, optimized, repeat):
    for record in (baseline, optimized, repeat):
        validate_record(record, correctness=True)
        for key in ("runner_contract_id", "frames", "image_shape", "seed", "initialization", "checkpoint", "tokens_per_frame",
                    "scale_frames", "warm_frames", "replay_frames"):
            if record[key] != baseline[key]:
                raise ValueError(f"Comparison protocol mismatch: {key}")
    if baseline["options"] != ThorOptions().to_dict() or repeat["options"] != baseline["options"]:
        raise ValueError("Baseline and repeat must both have every switch off")

    def differences(other):
        return [{"frame_index": a["frame_index"], "tensor": key}
                for a, b in zip(baseline["outputs"], other["outputs"], strict=True)
                for key in ("pose_enc", "depth") if a[key] != b[key]]

    repeat_diff, candidate_diff = differences(repeat), differences(optimized)
    return {
        "status": "baseline_unstable" if repeat_diff else "output_mismatch" if candidate_diff else "pass",
        "frames": baseline["frames"], "checked_output_batches": len(baseline["outputs"]),
        "baseline_repeat_equal": not repeat_diff, "baseline_optimized_equal": not candidate_diff,
        "baseline_repeat_differences": repeat_diff, "baseline_optimized_differences": candidate_diff,
        "scope": "streaming_synthetic",
    }
