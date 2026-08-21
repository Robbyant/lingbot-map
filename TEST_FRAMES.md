# lingbot-map — Test Frames & Results

Validation of lingbot-map's depth + camera (extrinsic/intrinsic) predictions on
video-game footage, across three datasets.

## Datasets & frames

| Dataset | Source | Runner | Results dir |
|---|---|---|---|
| CS:GO (Dust II DM) | `C:\csgo_data\lingbot_map_example\csgo_dm_1` (500 PNGs from HDF5) | `run_csgo_example.bat` / `run_self_test.bat` | `C:\csgo_data\lingbot_map_example\selftest_results` |
| Trackmania | `C:\workspace\data\left_right\left_right\1.mp4` | `run_trackmania_capture.bat` | `C:\workspace\data\left_right\left_right\1_results` |
| Cyberpunk 2077 | `C:\workspace\data\youtube\cyberpunk.mp4` (4K, 828s, 59.76fps) | `run_cyberpunk_test.bat` | `C:\workspace\data\youtube\cyberpunk_results` |

Each run exports (via `demo.py --export_results`):
- `poses.json` — per-frame extrinsics (c2w 3x4) + intrinsics (K 3x3, fx/fy, fov_h_deg)
- `depth\` — per-frame depth PNGs (min/max normalized, not metric)
- `depth_raw\` — raw depth `.npy` for frames 0 & 20 (every-20th) — used by the tests
- `rgb\` — the sampled input frames 0 & 20
- `point_cloud.ply` — fused 3D point cloud

The self-test uses **21 frames** (0-20); raw depth + RGB are exported every 20th, so the
reprojection tests currently operate on frames **0 and 20**.

## Tests

| Test | Script | What it checks |
|---|---|---|
| Convention self-test | `self_test_reprojection.py` | Unproject a frame's depth → world → reproject into the same frame. Error ~0px ⇒ extrinsic/intrinsic convention correct. |
| Reprojection overlay | `show_reprojection.py` | Same-frame = depth-colored overlay (visual PASS). Cross-frame (0→20) = frame 0's geometry seen from frame 20 (exposes depth/pose drift). |
| OpenCV pose compare | `compare_opencv_poses.py` | ORB → essential matrix → recoverPose vs lingbot extrinsics (rotation geodesic + translation-direction error). |
| OpenCV FOV compare | `compare_opencv_fov.py` | Focal self-calibration from the fundamental matrix vs lingbot's FOV. |

One-shot wrappers: `run_self_test.bat` (CS:GO), `run_self_test_all.bat` (all datasets,
reprojection only), `run_cyberpunk_test.bat` (cyberpunk extract + self-test + overlays).

## Results

### Convention self-test — PASS on all datasets
`mean = max = median = 0.0000px` for CS:GO, Trackmania, and Cyberpunk. The extrinsic
(c2w) / intrinsic convention is correct. **Caveat:** this only proves the coordinate math
is self-consistent, NOT that depth is accurate — a flat 2D menu round-trips to 0px too.

### Extrinsics vs OpenCV (CS:GO)
- Rotation: **median 0.55°** (sub-degree on 16/20 frame pairs) — strong agreement.
- Translation: direction-only (monocular), median ~16° — reasonable.
- The 4 bad pairs are OpenCV VO failures (fast motion, as few as 29 inliers), not lingbot.

### Intrinsics / FOV
- OpenCV focal self-calibration is **degenerate** on this footage (rotation-dominant /
  low-parallax) — pins to the search edge, so there is **no valid OpenCV FOV** to compare.
- lingbot's own FOV is **not temporally stable**: e.g. CS:GO steps from ~43° (frames 0-11)
  to ~70° (frames 13-20) at a shot change. Each frame's K is self-consistent, but the focal
  drifts frame-to-frame.

### Depth quality caveats
- **Sky is wrong** (Trackmania): the sky is assigned near/mid depth (~0.79) instead of the
  far maximum (3.13) — collapsed onto a near plane. Known monocular failure; use `--mask_sky`.
- **Cyberpunk intro is a menu**: the first ~60s of `cyberpunk.mp4` is the PS4 dashboard, not
  gameplay — depth there is meaningless. Use `--start_frame 3600` (~60s) to skip it.

## Reproducing

```
cd /d C:\workspace\world\lingbot-map
call .\.venv\Scripts\activate.bat
run_self_test.bat                 REM CS:GO 21-frame self-test (needs GPU)
run_self_test_all.bat             REM reprojection check across all datasets (no GPU)
run_cyberpunk_test.bat            REM cyberpunk gameplay extract + self-test + overlays
```

Memory-safe defaults are baked into `demo.py` (`--kv_cache_sliding_window 32`,
`--camera_num_iterations 1`); `expandable_segments` is a **no-op on Windows**.
Use `--no_viser` for headless runs (skips the blocking point-cloud viewer).
