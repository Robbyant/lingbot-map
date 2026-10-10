# FlashAttention-4 beta15 overlay

This directory contains the Python CuTe source from the official
`flash-attn-4==4.0.0b15` wheel.  It is loaded only when
`LINGBOT_THOR_FA4_QUERY_STAGING=1`; the system installation is not modified.

Upstream: <https://github.com/Dao-AILab/flash-attention>

Release tag: `fa4-v4.0.0.beta15`

The relevant SM110 change treats compute capability 11.x as part of the
Blackwell Q-stage selection.  For LingBot's `q_len > 128` shapes this selects
two Q stages instead of one while preserving the attention arithmetic.

Local modification: `LINGBOT_THOR_PAGED_KV_AFFINE=1` enables a single-page
affine-address fast path in the paged loader. Cross-page and partial-tail tiles
retain the general path. This local modification is separate from the upstream
query-staging change and is disabled by default.

Packaging omits the standalone benchmark helpers, FP8 benchmark, and SM90
configuration-search script. Runtime imports and the original license are
retained, including backward modules imported by the upstream interface. The
upstream AUTHORS file accompanies the license.

## Reviewing The Local Changes

Only three retained Python files differ from the official beta15 wheel:

- `interface.py`: validate the opt-in flag and pass it into dispatch/cache keys.
- `flash_fwd_sm100.py`: carry that option into the paged-KV loader.
- `paged_kv.py`: use affine addressing for a complete tile within one page.

The other retained Python files are unchanged upstream source. Runtime imports
include backward, other-architecture and block-sparse helpers; their presence
does not enable model-weight sparsity or training in the Thor runner.

To inspect or refresh this import, download and extract the pinned wheel without
installing it into the active environment:

```bash
python -m pip download --no-deps 'flash-attn-4==4.0.0b15' -d /tmp/thor-fa4-upstream
python -m zipfile -e /tmp/thor-fa4-upstream/flash_attn_4-4.0.0b15-py3-none-any.whl \
  /tmp/thor-fa4-upstream/extracted
diff -u /tmp/thor-fa4-upstream/extracted/flash_attn/cute/paged_kv.py \
  lingbot_map/_vendor/flash_attn_4_0_0b15/flash_attn/cute/paged_kv.py
```

Review the other two files in the same way. Keep the upstream import separate
from these local edits when updating versions. Retain the upstream license,
update `VERSION` and the loader's pinned version together, and rerun addressing,
capture/state, full-model output, and paired performance checks before adopting
an update. Do not refresh the bundled source from an unpinned branch.
