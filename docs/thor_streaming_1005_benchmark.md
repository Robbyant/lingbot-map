# Thor 1005-Token Streaming Benchmark

Date: 2026-10-06

## Protocol Correction

The old whole-frame graph froze Python camera history and temporal positions
at capture. Its benchmark-only cache adapter also allocated special pages in
a different order from graph reads. That experiment's
[5.145 to 6.844 FPS result](thor_legacy_1005_ablation.md) therefore cannot serve
as the corrected streaming result.

The current runner uses the shared cache manager and captures only the
aggregator and depth stages. Each measured frame updates temporal positions
and cache metadata, replays the aggregator, executes the original camera with
its growing history, then replays depth. Prime/capture does not consume a
camera frame. Strict deterministic compiled depth is used throughout.

No optimization implementation, default model, training or demo path was
changed for this correction.

## Workload And Environment

- NVIDIA Thor / SM110, 120 W (`nvpmodel -q`).
- Python 3.12.13, PyTorch 2.12.0+cu130, CUDA 13.0.
- FlashInfer 0.6.11.post3, CuTe DSL 4.5.2.
- Installed FA4 4.0.0b14 in all-off; packaged 4.0.0b15 with Query staging.
- Batch one, seed 42, CUDA FP32 random images converted to BF16, random model
  weights, no checkpoint.
- Input `[1,1000,3,378,518]`: 999 patch tokens plus six special tokens/frame.
- Eight scale frames, ten warm frames, 982 measured streaming steps.
- Camera/depth enabled, four camera refinements, 64-frame patch window; every
  frame is appended. Camera history and special-token history continue to grow.

The timing numerator is all 1000 requested frames. The denominator is the sum
of measured scale, warm and streaming host time. Streaming host time includes
input copying, temporal-position updates, cache preparation, aggregator graph,
camera, depth graph and synchronization. Input tensors are GPU-resident;
image loading/decoding is not measured. Setup, compilation, rehearsal, capture
and output collection are excluded. GPU-event timing is reported separately;
it covers aggregator/camera/depth after the input and cache updates.

Reported FPS is `1000 / median(run-level host ms per requested frame)`.
This is a synthetic benchmark, not demo throughput or a quality evaluation.

## Correctness

A discarded preflight preceded fresh-process all-off, full-stack and all-off
repeat runs. At 200 frames, all 193 output batches matched exactly for both
pose and depth (one eight-frame scale batch plus 192 single-frame batches).
The full stack enabled all six switches; no approximate visibility mode was used.

Each run also recorded 200 processed frames in both stages, 16 camera K/V
histories of length 200, and zero measured replay recompilation. Original
FP32 parameters and checkpoint keys were unchanged. The test suite passed
97 tests, including eight Thor GPU tests, before measurement.

## Performance

The corrected 1000-frame sweep completed in four fresh processes:
all-off, full stack, full stack, all-off. The full stack enables all six
optimization switches together. Intermediate configurations were not
independently measured in this endpoint sweep.

| Configuration | Runs | Host ms/requested frame | GPU ms/requested frame | FPS |
| --- | ---: | ---: | ---: | ---: |
| All options off | 2 | 199.578 | 198.367 | 5.011 |
| All six optimization switches enabled | 2 | 150.042 | 148.659 | 6.665 |

Throughput increased **33.0%**; host time per frame decreased **24.8%**, saving
**49.536 ms/frame**. The table uses the median run-level time for each endpoint.

| Run | Host ms/requested frame | FPS |
| --- | ---: | ---: |
| All-off, forward | 199.742482 | 5.006446 |
| Full stack, forward | 149.993971 | 6.666935 |
| Full stack, reverse | 150.089719 | 6.662682 |
| All-off, reverse | 199.412954 | 5.014719 |

The run-level host-time range was 0.165% of the baseline median and 0.064%
of the full-stack median. All runs completed 1000 camera/aggregator frames,
with 16 camera histories of length 1000 and zero measured replay recompilation.
Their scale, warm-tail and replay-tail output digests also matched. Per-frame
exact output comparison was performed in the separate 200-frame validation.

The [machine-readable record](thor_streaming_1005_benchmark.json) retains the
unrounded values, timing distributions, commands and validation summary.
Approximate visible-window mode was not measured in this corrected sweep.

## Reproduction And Evidence

From the repository root, in the [Thor environment](thor_inference.md#environment):

```bash
python -m unittest discover -s tests -p 'test_thor*.py' -v
python -m tools.thor_legacy_1005.sweep validate \
  --scope endpoints --out-dir /tmp/thor-streaming
python -m tools.thor_legacy_1005.sweep benchmark \
  --scope endpoints --out-dir /tmp/thor-streaming
```

Local run records: `profile/thor_corrected_streaming_1005_20261006/`.
Each child command, stdout/stderr, output comparison and timing sample is
retained there. Runner contract: `thor_1005_streaming_protocol_v4`.

Source provenance: parent revision `2d035998c562b6b0874b2237dddd57318995c890`
plus the local benchmark correction. These changes have not been committed
or pushed. Upstream base: `849e690bb086103637e44b1e91878d9d43a8bf0c`.
