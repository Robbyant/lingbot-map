# lingbot-map — Depth & Image Normalization

How depth and RGB are scaled through the export pipeline (`demo.py`), and what is / isn't
metric. TL;DR: the **`depth\` PNGs are per-frame min–max normalized for viewing only**;
the **`depth_raw\` `.npy` files are the raw model output** and are what every test uses.

## Depth map (PNG export)

Each frame's depth is written to `depth\<i>.png` with **per-frame min–max normalization**
(`demo.py:478-483`):

```python
d = depth[i, ..., 0]                                   # raw model depth for frame i (H×W)
d_min, d_max = d.min(), d.max()                        # THIS frame's own range
d_norm = (d - d_min) / (d_max - d_min + 1e-8)          # → [0, 1]
d_u8   = (d_norm * 255).clip(0, 255).astype(np.uint8)  # → 0-255 grayscale PNG
```

Properties:
- **Per-frame** scale — each frame is rescaled by its *own* min/max, not a global range.
- `+1e-8` avoids divide-by-zero on a flat (min==max) frame.
- **Visualization only, lossy, non-metric.** A given gray value means *different* depths in
  different frames, so **`depth\` PNGs are NOT comparable across frames** and must not be
  used for geometry.

## Raw depth (the metric source of truth)

`depth_raw\<i>.npy` holds the **unnormalized** depth straight from the model (float, H×W),
saved every `--comparison_stride` frames (default 20; set `1` for every frame). This is what
`self_test_reprojection.py`, `show_reprojection.py`, and `draw_reproj_row.py` unproject —
never the PNGs. `poses.json` documents the same: *"depth PNGs are per-frame min/max
normalized to 0-255 for visualization — not metric distance."*

## Point-cloud colors (separate)

For `point_cloud.ply`, RGB colors are scaled `(colors * 255).clip(0,255).astype(uint8)`
(`demo.py:389`) — a plain 0–1 → 0–255 conversion, independent of the depth normalization.
Point *confidence* (`depth_conf`, `all_points_conf_*`) is a separate signal, unrelated to the
0–255 depth PNG scaling.

## Input RGB (model input)

Frames are converted with `torchvision.transforms.ToTensor()` (`load_fn.py:40,142`), which
maps uint8 `[0,255]` → float `[0,1]`. **No mean/std standardization** is applied at this
stage — just the [0,1] scale plus the crop/resize to `image_size` (canonical crop mode).

## Practical implications

- Want **metric / cross-frame-comparable** depth? Use `depth_raw\*.npy`, not `depth\*.png`.
- Comparing two frames' depth PNGs visually is misleading (independent per-frame rescale).
- To get raw depth for a consecutive-frame range (e.g. the 10-frame reprojection row), run
  with `--comparison_stride 1` so every frame's `.npy` (and RGB) is exported.
