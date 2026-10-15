# Thor 1005-Token CUDA Graph Ablation

> Historical fixed-state results, superseded for the corrected streaming
> benchmark. This run froze camera history and temporal positions at capture
> and used inconsistent eager/graph special-page allocation. The figures below
> are retained as a record of that experiment, not as streaming performance.
> The current harness rejects its old validation records.

Measured on September 13, 2026: **5.145 to 6.844 FPS (+33.0% throughput)** with
all four lossless groups enabled. The baseline has all six switches off.

| Cumulative configuration | Runs | FPS |
|---|---:|---:|
| All options off | 3 | 5.145 |
| + MLP/QKV weight caching | 2 | 5.762 |
| + Projection caching and combined KV append | 2 | 5.828 |
| + FA4 dual-stage Query staging | 2 | 6.687 |
| + Single-page KV addressing | 3 | 6.844 |

The configurations were measured in forward/reverse sweeps, with a third
all-off/full-stack pair. The all-off run range was 0.08% of its median; the
full-stack range was 0.06%. These are cumulative ablations, not separate
single-switch contributions.

## Workload And Timing

- NVIDIA Thor SM110, 120 W; batch one, seed 42.
- Input: `[1,1000,3,378,518]`, 999 patch tokens and six special tokens per frame.
- CUDA FP32 random input cast to BF16 before constructing the random-weight model.
- Eight scale frames, ten warm frames, 982 graph replays; both heads enabled.
- Historical whole-frame capture with fixed Python camera/temporal state,
  FP32 positional arithmetic and the original special-page handoff.
- FA4 beta14 for the first three rows; packaged beta15 for the final two.
- No pretrained checkpoint, DeltaTok, sparsity or profiler.

Throughput is computed as requested frames divided by measured host sequence
time. For each run, that time includes scale, warmup and all replay steps.
Replay steps include input update, cache preparation and synchronization.
Model setup, compilation, rehearsal, capture and output collection are excluded.
The table uses the reciprocal of the median run-level time per requested frame.

Raw time values remain in the [structured measurement record](thor_legacy_1005_ablation.json)
so the FPS arithmetic can be checked. Displayed performance uses FPS throughout.
This synthetic protocol is not a real-input throughput or quality claim.

## Historical Reproduction

These results belong to the old whole-frame protocol, not the corrected
`thor_1005_streaming_protocol_v4` runner. Follow the
[current instructions](thor_inference.md#reproduction) for streaming validation
and measurement; do not reuse the figures above for that path.

The validation command collects every output in its configured regression
length; it does not infer full-sequence equality from timing records alone.
The historical checkout validation used 200 frames and passed all four cumulative
configurations with the all-off repeat. The endpoint timing revalidation used
1000 frames in forward and reverse order and produced 5.146 FPS for all-off and
6.845 FPS for the full stack, matching the historical headline within 0.03%.
