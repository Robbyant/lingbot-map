"""Corrected streaming capture protocol for the Thor synthetic benchmark."""
from contextlib import contextmanager
import hashlib
import os
import time

import torch

from lingbot_map.optimizations.thor.projection import (
    projection_cache_enabled, validate_housekeeping_runtime,
)
from .contracts import (
    COMPILE_COUNTER_KEYS,
    PROTOCOL_VERSION,
    validate_capture_depth_contract,
    validate_replay_contract,
)

FORMAL_SCALE_FRAMES = 8
FORMAL_WARM_FRAMES = 10
FORMAL_PROTOCOL_VERSION = PROTOCOL_VERSION
FORMAL_WARM_DEPTH_MODE = "compiled_depth_only_deterministic_cache_hit"
FORMAL_DETERMINISTIC_SCOPE = "model._predict_depth"
FORMAL_COMPILE_COUNTER_KEYS = COMPILE_COUNTER_KEYS


def validate_formal_cublas_workspace_unset():
    """Fail closed: the strict depth-only route does not use a cuBLAS override."""
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if observed is not None:
        raise RuntimeError(
            f"formal protocol v{FORMAL_PROTOCOL_VERSION} requires "
            "CUBLAS_WORKSPACE_CONFIG unset, "
            f"observed {observed!r}"
        )
    return None


def compile_counter_snapshot():
    """Capture Dynamo/Inductor counters that advance on compile or recompile."""
    counters = torch._dynamo.utils.counters
    return {
        f"{section}.{name}": int(counters[section][name])
        for section, name in FORMAL_COMPILE_COUNTER_KEYS
    }


def compile_counter_delta(before, after):
    return {
        name: int(after[name] - before[name])
        for name in before
    }


def validate_zero_compile_counter_delta(delta, *, phase):
    expected_keys = {
        f"{section}.{name}"
        for section, name in FORMAL_COMPILE_COUNTER_KEYS
    }
    if (
        not isinstance(delta, dict)
        or set(delta) != expected_keys
        or any(value != 0 for value in delta.values())
    ):
        raise RuntimeError(
            f"formal measured {phase} compiled or recompiled instead of "
            f"hitting cache: delta={delta!r}"
        )
    return delta


@contextmanager
def replay_without_recompile():
    """Keep setup outside timed frames and reject replay-time compilation."""
    before = compile_counter_snapshot()
    evidence = {}
    with torch._dynamo.config.patch(
        error_on_recompile=True,
        fail_on_recompile_limit_hit=True,
    ):
        yield evidence
    evidence.update(
        compile_counter_delta(before, compile_counter_snapshot())
    )
    validate_zero_compile_counter_delta(evidence, phase="replay")


def formal_input_signature(images):
    """Bind prewarm and formal timing to identical S=8 and S=1 tensor views."""
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.shape[1] < FORMAL_SCALE_FRAMES + FORMAL_WARM_FRAMES:
        raise RuntimeError("formal input signature requires at least 18 frames")

    def signature(value):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device_type": value.device.type,
        }

    return {
        "source": signature(images),
        "S8": {
            "calls": 1,
            **signature(images[:, :FORMAL_SCALE_FRAMES]),
        },
        "S1": {
            "calls": FORMAL_WARM_FRAMES,
            **signature(
                images[:, FORMAL_SCALE_FRAMES:FORMAL_SCALE_FRAMES + 1]
            ),
        },
    }


def validate_pose_depth_output_digests(digests, *, label):
    if not isinstance(digests, dict):
        raise RuntimeError(f"missing {label} output digests")
    for output in ("pose_enc", "depth"):
        digest = digests.get(output, {})
        if (
            not isinstance(digest, dict)
            or not isinstance(digest.get("sha256"), str)
            or len(digest["sha256"]) != 64
        ):
            raise RuntimeError(f"missing {label} {output} digest")
    return digests


def validate_warm_frame_output_digests(digests):
    if not isinstance(digests, list) or len(digests) != FORMAL_WARM_FRAMES:
        raise RuntimeError("formal protocol requires ten warm-frame output digests")
    for offset, item in enumerate(digests):
        expected_frame = FORMAL_SCALE_FRAMES + offset
        if not isinstance(item, dict) or item.get("frame_index") != expected_frame:
            raise RuntimeError(
                f"warm-frame digest index mismatch at offset {offset}"
            )
        validate_pose_depth_output_digests(
            item,
            label=f"frame {expected_frame}",
        )
    return digests


def assert_compiled_depth_route(model, compiled_depth_forward_impl, *, phase):
    """Bind every formal phase to the one callable produced by compile_default()."""
    depth_head = getattr(model, "depth_head", None)
    if depth_head is None or not hasattr(depth_head, "_forward_impl"):
        raise RuntimeError(f"{phase} requires depth_head._forward_impl")
    if depth_head._forward_impl is not compiled_depth_forward_impl:
        raise RuntimeError(f"{phase} did not use the compiled depth implementation")


def assert_compiled_nondeterministic_route(
    model,
    compiled_depth_forward_impl,
    *,
    phase,
):
    assert_compiled_depth_route(
        model,
        compiled_depth_forward_impl,
        phase=phase,
    )
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            f"{phase} requires deterministic algorithms disabled"
        )


@contextmanager
def use_strict_deterministic_compiled_depth_only(
    model,
    *,
    compiled_depth_forward_impl,
    expected_depth_calls,
):
    """Enable strict determinism only while ``model._predict_depth`` executes.

    The wrapper never replaces ``depth_head._forward_impl``: it calls the saved
    bf16 predictor, which still dispatches to the exact callable installed by
    ``compile_default()``.  Aggregator and camera launches occur outside this
    wrapper with deterministic algorithms disabled.
    """
    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="strict depth-only wrapper entry",
    )
    if expected_depth_calls <= 0:
        raise ValueError("strict depth-only wrapper requires a positive call count")
    if not hasattr(model, "_predict_depth"):
        raise RuntimeError("strict depth-only wrapper requires model._predict_depth")

    original_predict_depth = model._predict_depth
    entry_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    route = {
        "scope": FORMAL_DETERMINISTIC_SCOPE,
        "strict": True,
        "warn_only": False,
        "eager_swap": False,
        "depth_impl": "compiled",
        "depth_calls": 0,
        "scope_enters": 0,
        "scope_restores": 0,
        "compiled_callable_unchanged": True,
        "aggregator_deterministic_algorithms": False,
        "camera_deterministic_algorithms": False,
    }

    def strict_predict_depth(*args, **kwargs):
        assert_compiled_depth_route(
            model,
            compiled_depth_forward_impl,
            phase="strict deterministic depth call",
        )
        if torch.are_deterministic_algorithms_enabled():
            raise RuntimeError(
                "strict depth-only call must enter with deterministic algorithms disabled"
            )
        previous_warn_only = (
            torch.is_deterministic_algorithms_warn_only_enabled()
        )
        torch.use_deterministic_algorithms(True, warn_only=False)
        try:
            if (
                not torch.are_deterministic_algorithms_enabled()
                or torch.is_deterministic_algorithms_warn_only_enabled()
            ):
                raise RuntimeError(
                    "depth call did not enter strict deterministic mode"
                )
            route["depth_calls"] += 1
            route["scope_enters"] += 1
            return original_predict_depth(*args, **kwargs)
        finally:
            torch.use_deterministic_algorithms(
                False,
                warn_only=previous_warn_only,
            )
            if torch.are_deterministic_algorithms_enabled():
                raise RuntimeError(
                    "depth call did not restore deterministic algorithms disabled"
                )
            route["scope_restores"] += 1

    model._predict_depth = strict_predict_depth
    try:
        yield route
        if route["depth_calls"] != expected_depth_calls:
            raise RuntimeError(
                "strict depth-only route observed "
                f"{route['depth_calls']} calls, expected {expected_depth_calls}"
            )
        if (
            route["scope_enters"] != expected_depth_calls
            or route["scope_restores"] != expected_depth_calls
        ):
            raise RuntimeError(
                "strict depth-only route did not enter and restore exactly once "
                "per depth call"
            )
    finally:
        model._predict_depth = original_predict_depth
        if torch.are_deterministic_algorithms_enabled():
            torch.use_deterministic_algorithms(
                False,
                warn_only=entry_warn_only,
            )
        if model.depth_head._forward_impl is not compiled_depth_forward_impl:
            route["compiled_callable_unchanged"] = False
            raise RuntimeError(
                "strict depth-only wrapper changed the compiled depth callable"
            )


def prewarm_compiled_strict_depth_only_formal_route(
    model,
    images,
    dtype,
    *,
    compiled_depth_forward_impl,
):
    """Populate strict deterministic compiled depth variants outside timing.

    This mirrors the measured 8-frame scale plus 10 graph-mode streaming calls,
    including ``prepare_frame_for_graph``.  Only ``_predict_depth`` enables
    deterministic algorithms; aggregator and camera remain disabled.  No eager
    depth or aggregator route is installed.
    """
    validate_formal_cublas_workspace_unset()
    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="strict depth-only prewarm entry",
    )
    if images.ndim == 4:
        images = images.unsqueeze(0)
    if images.shape[1] < FORMAL_SCALE_FRAMES + FORMAL_WARM_FRAMES + 1:
        raise RuntimeError("deterministic prewarm requires at least 19 input frames")

    device = next(model.parameters()).device
    manager = model.aggregator.kv_cache_manager
    previous_graph_mode = manager._graph_mode
    previous_aggregator_nvtx = getattr(model.aggregator, "_nvtx_profile", False)
    previous_manager_nvtx = getattr(manager, "_nvtx_profile", False)
    autocast = torch.amp.autocast("cuda", dtype=dtype)
    input_signature = formal_input_signature(images)
    completed_passes = 0
    depth_route = None
    scale_output_digests = None
    warm_frame_output_digests = []
    first_pass_compile_counter_delta = None
    verification_compile_counter_delta = None

    for pass_index in (1, 2):
        verification_pass = pass_index == 2
        pass_warm_frame_output_digests = []
        model.clean_kv_cache()
        try:
            manager._graph_mode = False
            model.aggregator._nvtx_profile = False
            manager._nvtx_profile = False
            counters_before = compile_counter_snapshot()
            compile_guard = {
                "fail_on_recompile_limit_hit": True,
            }
            if verification_pass:
                compile_guard["error_on_recompile"] = True
            with torch._dynamo.config.patch(**compile_guard):
                with use_strict_deterministic_compiled_depth_only(
                    model,
                    compiled_depth_forward_impl=compiled_depth_forward_impl,
                    expected_depth_calls=1 + FORMAL_WARM_FRAMES,
                ) as pass_depth_route:
                    with torch.no_grad(), autocast:
                        scale_output = model.forward(
                            images[:, :FORMAL_SCALE_FRAMES].to(device),
                            num_frame_for_scale=FORMAL_SCALE_FRAMES,
                            num_frame_per_block=FORMAL_SCALE_FRAMES,
                            causal_inference=True,
                        )
                    if verification_pass:
                        torch.cuda.synchronize()
                        scale_output_digests = pose_depth_digests(scale_output)

                    manager._graph_mode = True
                    for frame in range(
                        FORMAL_SCALE_FRAMES,
                        FORMAL_SCALE_FRAMES + FORMAL_WARM_FRAMES,
                    ):
                        manager.prepare_frame_for_graph(frame)
                        with torch.no_grad(), autocast:
                            warm_output = model.forward(
                                images[:, frame:frame + 1].to(device),
                                num_frame_for_scale=FORMAL_SCALE_FRAMES,
                                num_frame_per_block=1,
                                causal_inference=True,
                            )
                        if verification_pass:
                            torch.cuda.synchronize()
                            pass_warm_frame_output_digests.append(
                                {
                                    "frame_index": frame,
                                    **pose_depth_digests(warm_output),
                                }
                            )
                    torch.cuda.synchronize()
                    assert_compiled_depth_route(
                        model,
                        compiled_depth_forward_impl,
                        phase="strict depth-only streaming prewarm",
                    )
            counters_after = compile_counter_snapshot()
            pass_delta = compile_counter_delta(counters_before, counters_after)
            if verification_pass:
                validate_zero_compile_counter_delta(
                    pass_delta,
                    phase="prewarm verification",
                )
                verification_compile_counter_delta = pass_delta
                depth_route = pass_depth_route
                warm_frame_output_digests = (
                    pass_warm_frame_output_digests
                )
            else:
                first_pass_compile_counter_delta = pass_delta
            completed_passes += 1
        finally:
            manager._graph_mode = previous_graph_mode
            model.aggregator._nvtx_profile = previous_aggregator_nvtx
            manager._nvtx_profile = previous_manager_nvtx
            model.clean_kv_cache()

    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="strict depth-only prewarm restore",
    )
    if manager._graph_mode != previous_graph_mode:
        raise RuntimeError("deterministic prewarm did not restore graph mode")
    if completed_passes != 2:
        raise RuntimeError("strict depth-only prewarm did not complete two passes")
    if (
        depth_route is None
        or first_pass_compile_counter_delta is None
        or verification_compile_counter_delta is None
    ):
        raise RuntimeError("strict depth-only prewarm metadata is missing")
    if scale_output_digests is None:
        raise RuntimeError("strict depth-only prewarm scale digest is missing")
    validate_pose_depth_output_digests(
        scale_output_digests,
        label="prewarm scale",
    )
    validate_warm_frame_output_digests(warm_frame_output_digests)
    return {
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "timed": False,
        "scope": FORMAL_DETERMINISTIC_SCOPE,
        "strict": True,
        "warn_only": False,
        "eager_swap": False,
        "depth_impl": "compiled",
        "depth_calls": depth_route["depth_calls"],
        "scope_enters": depth_route["scope_enters"],
        "scope_restores": depth_route["scope_restores"],
        "compiled_callable_unchanged": depth_route[
            "compiled_callable_unchanged"
        ],
        "aggregator_deterministic_algorithms": False,
        "camera_deterministic_algorithms": False,
        "shape_calls": {"S8": 1, "S1": FORMAL_WARM_FRAMES},
        "input_signature": input_signature,
        "scale_output_digests": scale_output_digests,
        "warm_frame_output_digests": warm_frame_output_digests,
        "passes": 2,
        "first_pass_compile_counter_delta": dict(
            first_pass_compile_counter_delta
        ),
        "first_compile_observed": any(
            first_pass_compile_counter_delta.values()
        ),
        "verification_compile_counter_delta": dict(
            verification_compile_counter_delta
        ),
        "verification_cache_hit": True,
        "verification_error_on_recompile": True,
        "fail_on_recompile_limit_hit": True,
        "scale_frames": FORMAL_SCALE_FRAMES,
        "warm_frames": FORMAL_WARM_FRAMES,
        "warm_frame_start": FORMAL_SCALE_FRAMES,
        "warm_frame_end_inclusive": (
            FORMAL_SCALE_FRAMES + FORMAL_WARM_FRAMES - 1
        ),
        "graph_mode_for_scale": False,
        "graph_mode_for_streaming": True,
        "prepare_frame_for_graph": True,
        "cache_hit_required_for_formal_warm": True,
        "deterministic_algorithms_restored": True,
        "predict_depth_wrapper_restored": True,
        "graph_mode_restored": True,
        "cache_cleaned_after": True,
    }


def formal_warm_depth_runtime_contract(
    *,
    scale_frames,
    warm_frames,
    deterministic_prewarm,
    scale_depth_route,
    warm_depth_route,
    input_signature,
    scale_compile_counter_delta,
    warm_compile_counter_delta,
    scale_matches_prewarm,
    formal_warm_frame_output_digests,
    warm_frames_match_prewarm,
    warm_depth_ptrs_unique,
    warm_tail_matches_prewarm,
    deterministic_restored_before_prime,
    compiled_confirmed_before_prime,
    predict_depth_wrapper_restored_before_prime,
    capture_depth_contract,
):
    """Return stable, address-free metadata for the compiled formal route."""
    if scale_frames != FORMAL_SCALE_FRAMES or warm_frames != FORMAL_WARM_FRAMES:
        raise ValueError(
            "formal warm-depth contract requires exactly 8 scale + 10 warm frames"
        )
    if (
        not isinstance(deterministic_prewarm, dict)
        or deterministic_prewarm.get("formal_protocol_version")
        != FORMAL_PROTOCOL_VERSION
        or deterministic_prewarm.get("timed") is not False
        or deterministic_prewarm.get("scope") != FORMAL_DETERMINISTIC_SCOPE
        or deterministic_prewarm.get("strict") is not True
        or deterministic_prewarm.get("warn_only") is not False
        or deterministic_prewarm.get("eager_swap") is not False
        or deterministic_prewarm.get("depth_impl") != "compiled"
        or deterministic_prewarm.get("depth_calls") != 11
        or deterministic_prewarm.get("scope_enters") != 11
        or deterministic_prewarm.get("scope_restores") != 11
        or deterministic_prewarm.get("compiled_callable_unchanged") is not True
        or deterministic_prewarm.get(
            "aggregator_deterministic_algorithms"
        ) is not False
        or deterministic_prewarm.get(
            "camera_deterministic_algorithms"
        ) is not False
        or deterministic_prewarm.get("cache_hit_required_for_formal_warm")
        is not True
        or deterministic_prewarm.get("shape_calls") != {"S8": 1, "S1": 10}
        or deterministic_prewarm.get("input_signature") != input_signature
        or deterministic_prewarm.get("passes") != 2
        or not isinstance(
            deterministic_prewarm.get("first_compile_observed"),
            bool,
        )
        or deterministic_prewarm.get("verification_cache_hit") is not True
        or deterministic_prewarm.get("verification_error_on_recompile")
        is not True
        or deterministic_prewarm.get("fail_on_recompile_limit_hit") is not True
    ):
        raise ValueError(
            "formal warm-depth contract requires an unmeasured strict "
            "depth-only 8+10 prewarm"
        )
    validate_pose_depth_output_digests(
        deterministic_prewarm.get("scale_output_digests"),
        label="prewarm scale",
    )
    validate_zero_compile_counter_delta(
        deterministic_prewarm.get("verification_compile_counter_delta"),
        phase="prewarm verification",
    )
    if (
        not isinstance(scale_depth_route, dict)
        or scale_depth_route.get("scope") != FORMAL_DETERMINISTIC_SCOPE
        or scale_depth_route.get("strict") is not True
        or scale_depth_route.get("warn_only") is not False
        or scale_depth_route.get("eager_swap") is not False
        or scale_depth_route.get("depth_impl") != "compiled"
        or scale_depth_route.get("depth_calls") != 1
        or scale_depth_route.get("scope_enters") != 1
        or scale_depth_route.get("scope_restores") != 1
        or scale_depth_route.get("compiled_callable_unchanged") is not True
        or scale_depth_route.get("aggregator_deterministic_algorithms")
        is not False
        or scale_depth_route.get("camera_deterministic_algorithms") is not False
    ):
        raise ValueError(
            "formal scale contract requires one strict depth-only call"
        )
    if (
        not isinstance(warm_depth_route, dict)
        or warm_depth_route.get("scope") != FORMAL_DETERMINISTIC_SCOPE
        or warm_depth_route.get("strict") is not True
        or warm_depth_route.get("warn_only") is not False
        or warm_depth_route.get("eager_swap") is not False
        or warm_depth_route.get("depth_impl") != "compiled"
        or warm_depth_route.get("depth_calls") != FORMAL_WARM_FRAMES
        or warm_depth_route.get("scope_enters") != FORMAL_WARM_FRAMES
        or warm_depth_route.get("scope_restores") != FORMAL_WARM_FRAMES
        or warm_depth_route.get("compiled_callable_unchanged") is not True
        or warm_depth_route.get("aggregator_deterministic_algorithms")
        is not False
        or warm_depth_route.get("camera_deterministic_algorithms") is not False
    ):
        raise ValueError(
            "formal warm-depth contract requires ten strict depth-only calls"
        )
    validate_warm_frame_output_digests(
        deterministic_prewarm.get("warm_frame_output_digests")
    )
    validate_warm_frame_output_digests(formal_warm_frame_output_digests)
    if (
        formal_warm_frame_output_digests
        != deterministic_prewarm.get("warm_frame_output_digests")
        or warm_frames_match_prewarm is not True
    ):
        raise ValueError(
            "formal measured warm per-frame digests do not match prewarm"
        )
    if warm_depth_ptrs_unique is not True:
        raise ValueError("formal measured warm depth storage was reused")
    validate_zero_compile_counter_delta(
        scale_compile_counter_delta,
        phase="scale",
    )
    validate_zero_compile_counter_delta(
        warm_compile_counter_delta,
        phase="warm",
    )
    if scale_matches_prewarm is not True:
        raise ValueError("formal scale did not match the prewarm reference")
    if warm_tail_matches_prewarm is not True:
        raise ValueError("formal warm tail did not match the prewarm reference")
    if not (
        deterministic_restored_before_prime
        and compiled_confirmed_before_prime
        and predict_depth_wrapper_restored_before_prime
    ):
        raise ValueError(
            "formal warm-depth contract requires restored deterministic, "
            "compiled-depth, and predictor-wrapper state before frame 18"
        )
    validate_capture_depth_contract(capture_depth_contract)
    capture_frame = scale_frames + warm_frames
    return {
        "formal_protocol_version": FORMAL_PROTOCOL_VERSION,
        "mode": FORMAL_WARM_DEPTH_MODE,
        "deterministic_scope": FORMAL_DETERMINISTIC_SCOPE,
        "strict": True,
        "warn_only": False,
        "cublas_workspace_config": validate_formal_cublas_workspace_unset(),
        "eager_swap": False,
        "compiled_callable_unchanged": True,
        "input_signature": dict(input_signature),
        "scale_depth_impl": "compiled",
        "scale_deterministic_algorithms": True,
        "scale_depth_deterministic_algorithms": True,
        "scale_aggregator_deterministic_algorithms": False,
        "scale_camera_deterministic_algorithms": False,
        "scale_depth_route": dict(scale_depth_route),
        "scale_scope_enters": scale_depth_route["scope_enters"],
        "scale_scope_restores": scale_depth_route["scope_restores"],
        "scale_toggle_in_timing": True,
        "scale_digest_in_timing": False,
        "scale_matches_prewarm": True,
        "formal_scale_cache_hit": True,
        "formal_scale_no_recompile": True,
        "formal_scale_compile_counter_delta": dict(
            scale_compile_counter_delta
        ),
        "prewarm_first_compile_observed": deterministic_prewarm[
            "first_compile_observed"
        ],
        "prewarm_verification_cache_hit": True,
        "fail_on_recompile_limit_hit": True,
        "warm_depth_impl": "compiled",
        "warm_depth_deterministic_algorithms": True,
        "warm_aggregator_deterministic_algorithms": False,
        "warm_camera_deterministic_algorithms": False,
        "warm_depth_route": dict(warm_depth_route),
        "warm_scope_enters": warm_depth_route["scope_enters"],
        "warm_scope_restores": warm_depth_route["scope_restores"],
        "formal_scope_enters": (
            scale_depth_route["scope_enters"]
            + warm_depth_route["scope_enters"]
            + capture_depth_contract["prime"]["scope_enters"]
            + capture_depth_contract["capture"]["scope_enters"]
        ),
        "formal_scope_restores": (
            scale_depth_route["scope_restores"]
            + warm_depth_route["scope_restores"]
            + capture_depth_contract["prime"]["scope_restores"]
            + capture_depth_contract["capture"]["scope_restores"]
        ),
        "warm_toggle_in_timing": True,
        "warm_digest_in_timing": False,
        "warm_depth_ptrs_unique": True,
        "warm_frames_match_prewarm": True,
        "post_warm_deterministic_algorithms": False,
        "warm_tail_matches_prewarm": True,
        "formal_warm_cache_hit": True,
        "formal_warm_no_recompile": True,
        "formal_warm_compile_counter_delta": dict(
            warm_compile_counter_delta
        ),
        "warm_frames": warm_frames,
        "warm_frame_start": scale_frames,
        "warm_frame_end_inclusive": capture_frame - 1,
        "deterministic_prewarm": dict(deterministic_prewarm),
        "formal_warm_frame_output_digests": list(
            formal_warm_frame_output_digests
        ),
        "warm_frame_output_digests_source": (
            "measured_warm_refs_hashed_after_timing"
        ),
        "deterministic_restored_before_steady_state_prime": bool(
            deterministic_restored_before_prime
        ),
        "compiled_depth_confirmed_before_steady_state_prime": bool(
            compiled_confirmed_before_prime
        ),
        "predict_depth_wrapper_restored_before_steady_state_prime": bool(
            predict_depth_wrapper_restored_before_prime
        ),
        "first_formal_single_frame_compiled_depth_frame": capture_frame,
        "formal_phase_deterministic_algorithms": {
            "scale": {"aggregator": False, "camera": False, "depth": True},
            "measured_warm": {
                "aggregator": False,
                "camera": False,
                "depth": True,
            },
            "post_warm": {
                "aggregator": False,
                "camera": False,
                "depth": False,
            },
            "frame18_prime": {
                "aggregator": False,
                "camera": False,
                "depth": True,
            },
            "cuda_graph_capture": {
                "aggregator": False,
                "camera": False,
                "depth": True,
            },
            "graph_replay": {
                "aggregator": False,
                "camera": False,
                "depth": True,
            },
        },
        "steady_state_prime": {
            "depth_impl": "compiled",
            "deterministic_algorithms": True,
        },
        "cuda_graph_capture": {
            "depth_impl": "compiled",
            "deterministic_algorithms": True,
        },
        "capture_depth_contract": capture_depth_contract,
        "replay_determinism_source": (
            "kernels_selected_under_strict_depth_during_capture"
        ),
        "host_deterministic_flag_during_replay": False,
        "phase_tail_digest_snapshot": {
            "scale": "after_phase_sync_before_depth_wrapper_restore",
            "warm_tail": "after_phase_sync_before_depth_wrapper_restore",
            "replay_tail": "after_final_replay_sync",
        },
    }


def patch_compile_camera_trunk(model):
    """Compile camera blocks with dynamic cache dimensions during warmup."""
    if model.camera_head is None:
        return
    for i in range(model.camera_head.trunk_depth):
        model.camera_head.trunk[i] = torch.compile(
            model.camera_head.trunk[i], dynamic=True
        )


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": digest.hexdigest(),
        "finite": bool(torch.isfinite(value).all()),
    }


def pose_depth_digests(output):
    return {
        name: tensor_digest(output[name])
        for name in ("pose_enc", "depth")
    }


def prime_and_capture_depth_route(
    model,
    static_input,
    *,
    capture_frame,
    scale_frames,
    side_stream,
    dtype,
    compiled_depth_forward_impl,
    use_nvtx=False,
):
    """Capture static stages while preserving dynamic camera history."""
    from .streaming_capture import CapturedStreamingStep

    step = CapturedStreamingStep(
        model,
        static_input,
        capture_frame=capture_frame,
        scale_frames=scale_frames,
        side_stream=side_stream,
        dtype=dtype,
        compiled_depth_forward_impl=compiled_depth_forward_impl,
        use_nvtx=use_nvtx,
    )
    return step, step.output, step.contract


def profile_with_capture_camcompile(
    model,
    images,
    num_frames,
    dtype,
    num_warm=10,
    use_nvtx=False,
    deterministic_captured_heads=True,
    compiled_depth_forward_impl=None,
    deterministic_prewarm=None,
    output_collector=None,
):
    validate_formal_cublas_workspace_unset()
    if not deterministic_captured_heads:
        raise ValueError("This benchmark requires serial camera then depth")
    if torch._dynamo.config.fail_on_recompile_limit_hit is not True:
        raise RuntimeError(
            "formal protocol requires fail_on_recompile_limit_hit=True"
        )
    if projection_cache_enabled():
        validate_housekeeping_runtime(model)
    if compiled_depth_forward_impl is None:
        raise RuntimeError("formal profile requires the compiled depth implementation")
    if not isinstance(deterministic_prewarm, dict):
        raise RuntimeError("formal profile requires deterministic prewarm metadata")
    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="formal scale entry",
    )
    device = next(model.parameters()).device
    if images.ndim == 4:
        images = images.unsqueeze(0)
    images = images.to(dtype)
    input_signature = formal_input_signature(images)
    if deterministic_prewarm.get("input_signature") != input_signature:
        raise RuntimeError(
            "deterministic depth prewarm input signature differs from formal input"
        )
    prewarm_warm_frame_output_digests = validate_warm_frame_output_digests(
        deterministic_prewarm.get("warm_frame_output_digests")
    )
    prewarm_scale_output_digests = validate_pose_depth_output_digests(
        deterministic_prewarm.get("scale_output_digests"),
        label="prewarm scale",
    )
    S = min(images.shape[1], num_frames)
    scale_frames = min(8, S)
    if scale_frames != FORMAL_SCALE_FRAMES or num_warm != FORMAL_WARM_FRAMES:
        raise RuntimeError("formal profile requires exactly 8 scale + 10 warm frames")

    autocast = torch.amp.autocast('cuda', dtype=dtype)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    manager = model.aggregator.kv_cache_manager
    manager._graph_mode = False
    model.clean_kv_cache()
    model.aggregator._nvtx_profile = False
    manager._nvtx_profile = False

    # Phase 1 (uncaptured): aggregator/camera remain deterministic-false while
    # the one compiled S=8 depth call uses the strict depth-only cache entry.
    scale_batch = images[:, :scale_frames].to(device)
    formal_scale_predict_depth_impl = model._predict_depth
    scale_compile_counters_before = compile_counter_snapshot()
    with torch._dynamo.config.patch(
        error_on_recompile=True,
        fail_on_recompile_limit_hit=True,
    ), use_strict_deterministic_compiled_depth_only(
        model,
        compiled_depth_forward_impl=compiled_depth_forward_impl,
        expected_depth_calls=1,
    ) as scale_depth_route:
        phase1_wall_start = time.perf_counter()
        start_ev.record()
        with torch.no_grad(), autocast:
            scale_output = model.forward(
                scale_batch,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        end_ev.record()
        torch.cuda.synchronize()
        phase1_wall_ms = (time.perf_counter() - phase1_wall_start) * 1000.0
        phase1_ms = start_ev.elapsed_time(end_ev)
        scale_output_digests = pose_depth_digests(scale_output)
    scale_compile_counters_after = compile_counter_snapshot()
    scale_compile_counter_delta = compile_counter_delta(
        scale_compile_counters_before,
        scale_compile_counters_after,
    )
    validate_zero_compile_counter_delta(
        scale_compile_counter_delta,
        phase="scale",
    )
    if model._predict_depth is not formal_scale_predict_depth_impl:
        raise RuntimeError("formal scale did not restore the depth predictor")
    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="formal scale exit",
    )
    scale_matches_prewarm = scale_output_digests == prewarm_scale_output_digests
    if not scale_matches_prewarm:
        raise RuntimeError(
            "formal scale differs from the exact-signature prewarm reference"
        )
    if output_collector is not None:
        output_collector(0, scale_output)
    print(f"  Phase 1: {phase1_ms:.1f} ms for {scale_frames} scale frames")

    # Switch to graph mode for measured streaming warm.  Only _predict_depth
    # enters strict deterministic mode; aggregator and camera stay disabled and
    # the depth head remains on its prewarmed compiled callable.
    manager._graph_mode = True
    manager._nvtx_profile = False
    actual_warm = min(num_warm, S - scale_frames - 1)
    if actual_warm != FORMAL_WARM_FRAMES:
        raise RuntimeError("formal profile requires ten measured warm frames")
    warm_start_ev = torch.cuda.Event(enable_timing=True)
    warm_end_ev = torch.cuda.Event(enable_timing=True)
    warm_output = None
    warm_output_refs = []
    formal_predict_depth_impl = model._predict_depth
    warm_compile_counters_before = compile_counter_snapshot()
    with torch._dynamo.config.patch(
        error_on_recompile=True,
        fail_on_recompile_limit_hit=True,
    ), use_strict_deterministic_compiled_depth_only(
        model,
        compiled_depth_forward_impl=compiled_depth_forward_impl,
        expected_depth_calls=FORMAL_WARM_FRAMES,
    ) as warm_depth_route:
        warm_wall_start = time.perf_counter()
        warm_start_ev.record()
        for i in range(actual_warm):
            f = scale_frames + i
            manager.prepare_frame_for_graph(f)
            with torch.no_grad(), autocast:
                warm_output = model.forward(
                    images[:, f:f+1].to(device),
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            warm_output_refs.append(warm_output)
        warm_end_ev.record()
        torch.cuda.synchronize()
        warm_ms = warm_start_ev.elapsed_time(warm_end_ev)
        warm_wall_ms = (time.perf_counter() - warm_wall_start) * 1000.0
        if warm_output is None:
            raise RuntimeError("profile requires at least one sequence-warm frame")
        if len(warm_output_refs) != FORMAL_WARM_FRAMES:
            raise RuntimeError("formal warm did not retain ten output references")
        depth_ptrs = [
            int(output["depth"].data_ptr())
            for output in warm_output_refs
        ]
        if len(set(depth_ptrs)) != FORMAL_WARM_FRAMES:
            raise RuntimeError(
                "formal warm depth outputs reuse storage; deferred digests unsafe"
            )
        formal_warm_frame_output_digests = [
            {
                "frame_index": scale_frames + offset,
                **pose_depth_digests(output),
            }
            for offset, output in enumerate(warm_output_refs)
        ]
        validate_warm_frame_output_digests(
            formal_warm_frame_output_digests
        )
        warm_output_digests = {
            output: formal_warm_frame_output_digests[-1][output]
            for output in ("pose_enc", "depth")
        }

    if output_collector is not None:
        for offset, output in enumerate(warm_output_refs):
            output_collector(scale_frames + offset, output)

    warm_compile_counters_after = compile_counter_snapshot()
    warm_compile_counter_delta = compile_counter_delta(
        warm_compile_counters_before,
        warm_compile_counters_after,
    )
    validate_zero_compile_counter_delta(
        warm_compile_counter_delta,
        phase="warm",
    )
    warm_frames_match_prewarm = (
        formal_warm_frame_output_digests
        == prewarm_warm_frame_output_digests
    )
    warm_tail_matches_prewarm = warm_frames_match_prewarm
    if not warm_frames_match_prewarm:
        raise RuntimeError(
            "formal warm per-frame outputs differ from exact-signature prewarm"
        )
    deterministic_restored_before_prime = (
        not torch.are_deterministic_algorithms_enabled()
    )
    compiled_confirmed_before_prime = (
        model.depth_head._forward_impl is compiled_depth_forward_impl
    )
    predict_depth_wrapper_restored_before_prime = (
        model._predict_depth is formal_predict_depth_impl
    )
    assert_compiled_nondeterministic_route(
        model,
        compiled_depth_forward_impl,
        phase="steady-state prime entry",
    )
    print(
        "  Graph-mode warmup: "
        f"{actual_warm} streaming frames "
        "(strict deterministic compiled depth only; aggregator/camera disabled)"
    )

    # Camera runs on the side stream; depth waits for it before execution.
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    probe_head_mode = "serial_camera_then_depth"
    if deterministic_captured_heads:
        print("  Deterministic captured heads enabled: serial_camera_then_depth")

    capture_frame = scale_frames + actual_warm
    static_input = images[:, capture_frame:capture_frame+1].to(device).clone()

    g, static_output, capture_depth_contract = prime_and_capture_depth_route(
        model, static_input, capture_frame=capture_frame, scale_frames=scale_frames,
        side_stream=side_stream, dtype=dtype,
        compiled_depth_forward_impl=compiled_depth_forward_impl, use_nvtx=use_nvtx,
    )
    print(
        f"  Captured aggregator/depth at frame {capture_frame} "
        f"(head_mode={probe_head_mode}, uncaptured dynamic camera, "
        "strict bf16 depth)"
    )
    warm_depth_contract = formal_warm_depth_runtime_contract(
        scale_frames=scale_frames,
        warm_frames=actual_warm,
        deterministic_prewarm=deterministic_prewarm,
        scale_depth_route=scale_depth_route,
        warm_depth_route=warm_depth_route,
        input_signature=input_signature,
        scale_compile_counter_delta=scale_compile_counter_delta,
        warm_compile_counter_delta=warm_compile_counter_delta,
        scale_matches_prewarm=scale_matches_prewarm,
        formal_warm_frame_output_digests=(
            formal_warm_frame_output_digests
        ),
        warm_frames_match_prewarm=warm_frames_match_prewarm,
        warm_depth_ptrs_unique=(
            len(set(depth_ptrs)) == FORMAL_WARM_FRAMES
        ),
        warm_tail_matches_prewarm=warm_tail_matches_prewarm,
        deterministic_restored_before_prime=deterministic_restored_before_prime,
        compiled_confirmed_before_prime=compiled_confirmed_before_prime,
        predict_depth_wrapper_restored_before_prime=(
            predict_depth_wrapper_restored_before_prime
        ),
        capture_depth_contract=capture_depth_contract,
    )
    warm_depth_contract["capture_boundary"] = capture_depth_contract[
        "capture_boundary"
    ]
    warm_depth_contract["camera_execution"] = capture_depth_contract[
        "camera_execution"
    ]
    warm_depth_contract["temporal_positions"] = capture_depth_contract[
        "temporal_positions"
    ]

    per_frame_ms = []
    per_frame_wall_ms = []
    measure_start = capture_frame
    with replay_without_recompile() as replay_compile_delta:
        for f in range(measure_start, S):
            replay_wall_start = time.perf_counter()
            static_input.copy_(images[:, f:f+1].to(device, non_blocking=True))
            g.prepare(f)
            start_ev.record()
            if use_nvtx:
                torch.cuda.nvtx.range_push(f"replay_{f}")
            g.replay()
            if use_nvtx:
                torch.cuda.nvtx.range_pop()
            end_ev.record()
            torch.cuda.synchronize()
            per_frame_ms.append(start_ev.elapsed_time(end_ev))
            per_frame_wall_ms.append(
                (time.perf_counter() - replay_wall_start) * 1000.0
            )
            if output_collector is not None:
                output_collector(f, static_output)

    warm_depth_contract["replay_compile_counter_delta"] = dict(
        replay_compile_delta
    )
    warm_depth_contract["completed_streaming_state"] = {
        "aggregator_frames": model.aggregator.total_frames_processed,
        "camera_frames": model.camera_head.frame_idx,
        "replayed_frames": g.next_frame - measure_start,
        "camera_cache_lengths": [
            int(cache[f"k_{block}"].shape[2])
            for cache in model.camera_head.kv_cache
            for block in range(model.camera_head.trunk_depth)
        ],
    }
    validate_replay_contract(warm_depth_contract, S)

    phase_output_digests = {
        "scale": scale_output_digests,
        "warm_tail": warm_output_digests,
        "replay_tail": pose_depth_digests(static_output),
    }
    return (
        per_frame_ms,
        scale_frames,
        phase1_ms,
        actual_warm,
        warm_ms,
        phase1_wall_ms,
        warm_wall_ms,
        per_frame_wall_ms,
        phase_output_digests,
        warm_depth_contract,
    )
