# Thor Inference Optimizations

Four opt-in optimization groups reduce BF16 weight preparation, KV writes and
FA4 data-loading overhead on NVIDIA Thor.
The input is **518 x 378 (width x height), 1005 tokens/frame**, batch size one.

The optimization modules and benchmark are separate from the default model,
training code and demo. No option is enabled by importing the package.

## Optimization Groups

1. **MLP/QKV weight caching.** Reuse non-persistent BF16 weights and biases
   instead of converting the same FP32 parameters on every frame. The original
   parameters and checkpoint keys are preserved.
2. **Projection caching and combined KV append.** Cache attention output
   projection weights and use one FlashInfer paged-KV append per global block
   for patch and special tokens. The KV-write subgraph goes from 96 to 24 nodes.
   Projection caching and append have separate switches.
3. **FA4 dual-stage Query staging.** Adopt upstream FA4 beta15's SM110 Query
   loading schedule. Upstream source, license and attribution are packaged
   separately from the local integration.
4. **Single-page KV addressing.** Use affine addressing for KV tiles contained
   in one page. Cross-page tiles and tails retain the general loader.

## Performance

The corrected 1000-frame endpoint comparison improves from **5.011 to
6.665 FPS (+33.0%)**, with all six optimization switches enabled together.

See the [corrected streaming measurement report](thor_streaming_1005_benchmark.md)
for the current all-off/full-stack comparison. The previous
[fixed-state ablation](thor_legacy_1005_ablation.md) is retained as historical
evidence only; its FPS figures do not apply to the corrected streaming path.

### Benchmark Contract

Both arms use the same BF16 path, random weights, four camera
refinement iterations, and enabled camera/depth heads. There are eight scale
frames, ten streaming warmup frames and 982 measured streaming steps.
Each step updates the input, temporal RoPE positions and cache page table,
replays the aggregator graph, runs the original camera with growing history,
then replays the depth graph. Input/cache/position updates, both graphs,
camera execution and synchronization are timed. Construction, compilation,
rehearsal, capture and output collection are excluded.

Scale writes and graph reads use the shared cache manager's special-page
allocation. Capture does not consume a camera frame. Depth uses strict
deterministic compiled kernels in every phase; original FP32 parameters and
position arithmetic are retained. This workload appends every frame, without
dynamic keyframe selection.

These are synthetic results, not default-demo throughput or a real-sequence
reconstruction-quality evaluation.

## Environment

The recorded runs used NVIDIA Thor (SM110), 120 W power mode, Python 3.12.13,
PyTorch 2.12.0+cu130, CUDA 13.0, FlashInfer 0.6.11.post3 and CuTe DSL 4.5.2.
Install the matching PyTorch/CUDA build first, then install from the checkout:

```bash
python -m pip install -e '.[thor]'
```

FA4 beta14 is the installed reference. The Query-staging option selects the
packaged beta15 implementation before its first import. Use a fresh process
when changing FA4 or optimization options.

## Switches

| Switch | Purpose |
|---|---|
| `--cache-mlp-weights` | Global, frame and patch MLP weight caches |
| `--cache-qkv-weights` | Global, frame and patch QKV weight caches |
| `--cache-projection-weights` | Attention output projection caches |
| `--merge-kv-append` | Combined patch and special-token append |
| `--fa4-query-staging` | Packaged FA4 beta15 Query schedule |
| `--paged-kv-affine` | Single-page addressing; requires Query staging |

All six switches default to off. The optional `--visible-window 56` additionally
requires `--allow-approximate`; it is not part of the lossless comparison.
It changes attention visibility, not the physical 64-page patch allocation.
The Thor runtime appends every frame. Dynamic keyframes, deferred eviction and
rollback are unsupported and raise an error before modifying the cache.

## Reproduction

Run from the repository root. The harness is a checkout-only benchmark under
`tools/thor_legacy_1005`; no installed inference command or checkpoint is needed.
Its input dimensions, seed, head configuration and capture protocol are fixed.

### Short Reproduction

This command runs a discarded preflight, all-off reference, full stack and an
all-off repeat in separate processes. The default regression length is 200
frames, including 182 streaming steps using the captured static stages.

```bash
python -m tools.thor_legacy_1005.sweep validate \
  --scope endpoints --out-dir /tmp/thor-streaming
```

It compares pose/depth for every collected scale, warm and replay output,
checks finiteness, verifies unchanged FP32 parameters and checkpoint keys,
and rejects incomplete, stale, or unstable reference records. Every validation
record carries the runner contract identifier and expected frame count, so a
benchmark cannot consume results from a different runner or workload.
Records also require complete camera/aggregator progress and zero compilation
during measured execution. Old fixed-state records are rejected.

### 1000-Frame Timing

After successful validation, run the same configurations in forward and reverse
order. The command rechecks the validation records before starting any timing.

```bash
python -m tools.thor_legacy_1005.sweep benchmark \
  --scope endpoints --out-dir /tmp/thor-streaming
```

Each child process explicitly sets its switches, clears inherited experimental
flags and enables the FA4 compile cache. Logs, commands, raw timings and output
comparisons are retained in the output directory. Existing runs are never
overwritten. Use a new directory for another campaign.

For all four cumulative optimization groups, use `--scope ablation` for both
commands in a new output directory.
For a single diagnostic run:

```bash
python -m tools.thor_legacy_1005.run --frames 19 \
  --cache-mlp-weights --cache-qkv-weights \
  --cache-projection-weights --merge-kv-append \
  --fa4-query-staging --paged-kv-affine \
  --out /tmp/thor-full-smoke.json
```

### Tests

```bash
python -m unittest discover -s tests -p 'test_thor*.py' -v
```

The tests cover option isolation, unsupported shapes, append/cache behavior,
static-stage capture, dynamic streaming state and rejection of invalid
timing/comparison records. GPU tests require the Thor environment. The separate FA4 checker also
compares captured kernel outputs with its reference across supported shapes.

## Rollback

Use a fresh process with all six switches off. To avoid the benchmark path
entirely, use the unchanged upstream model or demo. Do not reuse a patched
model for training, checkpoint replacement, device changes or another sequence.
