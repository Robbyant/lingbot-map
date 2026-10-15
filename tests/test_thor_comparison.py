"""Streaming synthetic comparison tests with stale-record rejection."""
import copy
import io
import json
import unittest
from unittest.mock import patch

from tools import compare_thor_runs as wrapper
from tools.thor_legacy_1005 import records
from lingbot_map.optimizations.thor.options import ThorOptions

if __package__:
    from .test_thor_measurements import output_digests, timing_result
else:
    from test_thor_measurements import output_digests, timing_result


def make_record(frames=19):
    return {
        "runner_contract_id": records.RUNNER_CONTRACT_ID,
        "kind": "synthetic_correctness_smoke",
        "per_frame_output_collection": True,
        "frames": frames, "scale_frames": 8, "warm_frames": 10,
        "replay_frames": frames - 18, "image_shape": [378, 518],
        "tokens_per_frame": 1005, "seed": 42, "checkpoint": None,
        "initialization": "cuda_fp32_input_to_bf16_then_random_model",
        "options": ThorOptions().to_dict(),
        "model_state": {"original_fp32_parameters_intact": True,
                        "state_dict_keys_unchanged": True},
        "outputs": [{"frame_index": index,
                     **output_digests(8 if index == 0 else 1)}
                    for index in [0, *range(8, frames)]],
        "result": timing_result(frames),
    }


class ThorComparisonTest(unittest.TestCase):
    def setUp(self):
        self.baseline = make_record()
        self.optimized = make_record()
        self.optimized["options"]["cache_mlp_weights"] = True
        self.repeat = make_record()

    def compare(self):
        return wrapper.compare_records(self.baseline, self.optimized, self.repeat)

    def test_wrapper_reuses_record_validator_and_comparison(self):
        self.assertIs(wrapper.compare_records, records.compare_records)
        self.assertIs(wrapper.validate_record, records.validate_record)

    def test_equal_outputs_pass_in_streaming_synthetic_scope(self):
        result = self.compare()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["scope"], "streaming_synthetic")
        self.assertEqual(result["checked_output_batches"], 12)
        self.assertTrue(result["baseline_repeat_equal"])
        self.assertTrue(result["baseline_optimized_equal"])

    def test_unstable_baseline_takes_precedence(self):
        self.repeat["outputs"][-1]["depth"]["sha256"] = "c" * 64
        self.optimized["outputs"][-1]["pose_enc"]["sha256"] = "d" * 64
        result = self.compare()
        self.assertEqual(result["status"], "baseline_unstable")
        self.assertFalse(result["baseline_repeat_equal"])
        self.assertFalse(result["baseline_optimized_equal"])

    def test_output_difference_is_reported(self):
        self.optimized["outputs"][-1]["depth"]["sha256"] = "c" * 64
        result = self.compare()
        self.assertEqual(result["status"], "output_mismatch")
        self.assertEqual(result["baseline_optimized_differences"],
                         [{"frame_index": 18, "tensor": "depth"}])

    def test_protocol_and_phase_mismatches_rejected(self):
        changes = (("frames", 20), ("frames", True), ("seed", 0),
                   ("seed", 42.0), ("checkpoint", "/model.pt"),
                   ("image_shape", [378, 504]), ("image_shape", [378.0, 518]),
                   ("tokens_per_frame", 978), ("tokens_per_frame", 1005.0),
                   ("scale_frames", 7), ("warm_frames", 9),
                   ("replay_frames", 2), ("replay_frames", True),
                   ("initialization", "model_then_input"))
        for key, value in changes:
            self.optimized = make_record()
            self.optimized[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.compare()

    def test_input_signature_mismatch_rejected(self):
        for view in ("source", "S8", "S1"):
            for key, value in (("device_type", "cpu"), ("dtype", "torch.float32"),
                               ("shape", [1, 1, 3, 518, 378]), ("stride", [1] * 5)):
                self.optimized = make_record()
                self.optimized["result"][9]["input_signature"][view][key] = value
                with self.subTest(view=view, key=key), self.assertRaises(ValueError):
                    self.compare()
        for view in ("S8", "S1"):
            self.optimized = make_record()
            self.optimized["result"][9]["input_signature"][view]["calls"] = 2
            with self.subTest(view=view), self.assertRaises(ValueError):
                self.compare()
        self.optimized = make_record()
        del self.optimized["result"][9]["input_signature"]
        with self.assertRaises(ValueError):
            self.compare()

    def test_individually_valid_different_workloads_cannot_be_compared(self):
        self.optimized = make_record(20)
        records.validate_record(self.optimized, correctness=True)
        with self.assertRaisesRegex(ValueError, "Comparison protocol mismatch: frames"):
            self.compare()

    def test_missing_or_duplicate_outputs_rejected(self):
        for change in (lambda rows: rows.pop(),
                       lambda rows: rows.append(copy.deepcopy(rows[-1])),
                       lambda rows: rows.reverse()):
            self.optimized = make_record()
            change(self.optimized["outputs"])
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.compare()
        for name in ("pose_enc", "depth"):
            self.optimized = make_record()
            del self.optimized["outputs"][-1][name]
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.compare()

    def test_invalid_output_shape_dtype_finite_or_digest_rejected(self):
        for name in ("pose_enc", "depth"):
            for key, value in (("shape", [1]), ("dtype", "torch.bfloat16"),
                               ("finite", False), ("finite", 1), ("finite", None),
                               ("sha256", "g" * 64), ("sha256", "a" * 63),
                               ("sha256", None), ("sha256", 42)):
                self.optimized = make_record()
                self.optimized["outputs"][-1][name][key] = value
                with self.subTest(name=name, key=key), self.assertRaises(ValueError):
                    self.compare()

    def test_output_finite_flag_cannot_be_omitted(self):
        for name in ("pose_enc", "depth"):
            value = make_record()
            del value["outputs"][-1][name]["finite"]
            with self.subTest(name=name), self.assertRaises(ValueError):
                records.validate_record(value, correctness=True)

    def test_approximate_or_benchmark_records_rejected(self):
        for changes in ({"kind": "synthetic_profile"},
                        {"per_frame_output_collection": False},
                        {"options": ThorOptions(allow_approximate=True).to_dict()},
                        {"options": ThorOptions(visible_window=56,
                                                 allow_approximate=True).to_dict()}):
            self.optimized = make_record()
            self.optimized.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.compare()

    def test_baseline_and_repeat_require_all_switches_off(self):
        for arm in ("baseline", "repeat"):
            target = getattr(self, arm)
            target["options"]["cache_qkv_weights"] = True
            with self.subTest(arm=arm), self.assertRaises(ValueError):
                self.compare()
            target["options"]["cache_qkv_weights"] = False

    def test_integrity_flags_are_required_and_true(self):
        for arm in ("baseline", "optimized", "repeat"):
            for key in ("original_fp32_parameters_intact", "state_dict_keys_unchanged"):
                for value in (False, None, 1):
                    self.setUp()
                    getattr(self, arm)["model_state"][key] = value
                    with self.subTest(arm=arm, key=key, value=value), self.assertRaises(ValueError):
                        self.compare()

    def test_missing_required_record_fields_rejected(self):
        for key in ("frames", "checkpoint", "result", "outputs", "model_state", "options"):
            value = make_record()
            del value[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                records.validate_record(value, correctness=True)

    def test_malformed_model_state_rejected(self):
        for state in (None, [], True):
            value = make_record()
            value["model_state"] = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                records.validate_record(value, correctness=True)

    def test_protocol_version_and_capture_scope_are_checked(self):
        for key, value in (("formal_protocol_version", 2),
                           ("formal_protocol_version", 4.0),
                           ("capture_boundary", "whole_frame"),
                           ("camera_execution", "captured_fixed_python_state"),
                           ("temporal_positions", "fixed_at_capture")):
            value_record = make_record()
            value_record["result"][9][key] = value
            self.optimized = value_record
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.compare()

    def test_phase_digest_finite_flags_are_checked(self):
        for phase in ("scale", "warm_tail", "replay_tail"):
            for name in ("pose_enc", "depth"):
                for key, invalid in (("finite", False), ("finite", None), ("finite", 1),
                                     ("sha256", None), ("sha256", 42)):
                    value = make_record()
                    value["result"][8][phase][name][key] = invalid
                    with self.subTest(phase=phase, name=name, key=key), self.assertRaises(ValueError):
                        records.validate_record(value, correctness=True)

    def test_correctness_records_validate_raw_sample_counts_and_values(self):
        for index, invalid in ((0, []), (7, [1, 2]), (2, float("nan")),
                               (4, 0), (5, True), (6, -1), (0, [float("inf")])):
            value = make_record()
            value["result"][index] = invalid
            with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                records.validate_record(value, correctness=True)

    def test_in_memory_tuple_and_json_records_validate_in_both_modes(self):
        for correctness in (False, True):
            value = make_record(200)
            value["result"] = tuple(value["result"])
            if not correctness:
                value.update(kind="synthetic_profile", per_frame_output_collection=False, outputs=[])
                value["timing"] = records.summarize_timing(value["result"], value["frames"])
            for record in (value, json.loads(json.dumps(value))):
                with self.subTest(correctness=correctness, container=type(record["result"]).__name__):
                    records.validate_record(record, correctness=correctness)

    def test_profile_summary_must_match_raw_samples(self):
        value = make_record()
        value.update(kind="synthetic_profile", per_frame_output_collection=False, outputs=[])
        value["timing"] = records.summarize_timing(value["result"], value["frames"])
        value["timing"]["fps"] *= 2
        with self.assertRaisesRegex(ValueError, "Timing summary differs"):
            records.validate_record(value)

    def test_cli_reports_status_and_returns_nonzero_on_invalid_or_unequal_evidence(self):
        for status in ("pass", "invalid_evidence", "baseline_unstable", "output_mismatch"):
            baseline, optimized, repeat = make_record(), make_record(), make_record()
            if status == "invalid_evidence":
                del optimized["outputs"]
            elif status == "baseline_unstable":
                repeat["outputs"][-1]["depth"]["sha256"] = "c" * 64
            elif status == "output_mismatch":
                optimized["outputs"][-1]["depth"]["sha256"] = "c" * 64
            with self.subTest(status=status), \
                 patch("sys.argv", ["compare_thor_runs", "--baseline", "baseline.json",
                                    "--optimized", "optimized.json", "--repeat", "repeat.json"]), \
                 patch.object(wrapper.Path, "read_text", side_effect=[
                     json.dumps(record) for record in (baseline, optimized, repeat)]), \
                 patch("sys.stdout", new_callable=io.StringIO) as output:
                code = wrapper.main()
            self.assertEqual(code, 0 if status == "pass" else 1)
            self.assertEqual(json.loads(output.getvalue())["status"], status)


if __name__ == "__main__":
    unittest.main()
