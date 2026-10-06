"""Address-free validation for the corrected Thor streaming benchmark."""

PROTOCOL_VERSION = 4
COMPILE_COUNTER_KEYS = (
    ("frames", "total"),
    ("stats", "unique_graphs"),
    ("inductor", "fxgraph_cache_miss"),
    ("inductor", "fxgraph_cache_bypass"),
    ("aot_autograd", "total"),
)


def _zero_compile_counters():
    return {
        f"{section}.{name}": 0
        for section, name in COMPILE_COUNTER_KEYS
    }


def validate_capture_depth_contract(contract):
    expected_route = {
        "scope": "model._predict_depth",
        "strict": True,
        "warn_only": False,
        "eager_swap": False,
        "depth_impl": "compiled",
        "depth_calls": 1,
        "scope_enters": 1,
        "scope_restores": 1,
        "compiled_callable_unchanged": True,
        "aggregator_deterministic_algorithms": False,
        "camera_deterministic_algorithms": False,
    }
    if not isinstance(contract, dict):
        raise ValueError("Missing capture contract")
    for phase in ("prime", "capture"):
        route = contract.get(phase)
        if (
            not isinstance(route, dict)
            or route != expected_route
            or any(
                type(route[key]) is not type(value)
                for key, value in expected_route.items()
            )
        ):
            raise ValueError(
                f"{phase} requires exactly one strict compiled depth call"
            )
    zeros = _zero_compile_counters()
    for key in ("compile_counter_delta", "aggregator_compile_counter_delta"):
        delta = contract.get(key)
        if (
            not isinstance(delta, dict)
            or delta != zeros
            or any(type(value) is not int for value in delta.values())
        ):
            raise ValueError(
                f"Capture compiled or recompiled instead of hitting cache: {key}"
            )
    required = {
        "replay_uses_captured_strict_depth": True,
        "host_deterministic_flag_during_replay": False,
        "capture_boundary": "aggregator_and_depth_only",
        "camera_execution": "uncaptured_original_dynamic_history",
        "temporal_positions": "original_rope_copied_each_frame",
        "prime_consumes_camera_history": False,
    }
    for key, value in required.items():
        if contract.get(key) != value or type(contract.get(key)) is not type(value):
            raise ValueError(f"Missing or invalid streaming capture evidence: {key}")


def validate_replay_contract(contract, frames):
    """Require complete streaming state and no compilation during replay."""
    if type(frames) is not int or frames < 19 or not isinstance(contract, dict):
        raise ValueError("Invalid completed streaming contract")
    state = contract.get("completed_streaming_state")
    expected = {
        "aggregator_frames": frames,
        "camera_frames": frames,
        "replayed_frames": frames - 18,
        "camera_cache_lengths": [frames] * 16,
    }
    if (
        not isinstance(state, dict)
        or state != expected
        or any(
            type(state[key]) is not int
            for key in expected
            if key != "camera_cache_lengths"
        )
        or any(type(length) is not int for length in state["camera_cache_lengths"])
    ):
        raise ValueError("Missing or incomplete camera/aggregator streaming state")
    delta = contract.get("replay_compile_counter_delta")
    zeros = _zero_compile_counters()
    if (
        not isinstance(delta, dict)
        or delta != zeros
        or any(type(value) is not int for value in delta.values())
    ):
        raise ValueError("Missing no-recompile evidence for measured replay")
