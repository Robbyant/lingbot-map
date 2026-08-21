# lingbot-map runs on game/video footage — camera + confidence stats

Testing whether lingbot-map (SLAM/streaming 3D reconstruction, RGB-only) works on
video-game footage instead of real-world capture. All runs use `lingbot-map.pt`
(balanced checkpoint), `--use_sdpa` (no FlashInfer installed).

Point-cloud confidence stats (`point_cloud_confidence` in `poses.json`) were added to
`demo.py`'s `export_results` after these three runs completed, so none of them have it
yet -- re-run to get confidence numbers for a given clip.

## Bug: `point_cloud.ply`'s depth-fallback path used c2w where w2c was required

`export_point_cloud_ply`'s fallback (triggers whenever `world_points` isn't in the
model's raw output, e.g. streaming mode -- happened for at least the Trackmania run)
called `unproject_depth_map_to_point_map(depth, predictions["extrinsic"], intrinsic)`.
That function (via `depth_to_world_coords_points`) requires a **w2c** (world-to-camera)
extrinsic. `predictions["extrinsic"]` is **c2w** (camera-to-world) -- see
`postprocess()`'s own "Convert w2c to c2w" comment. No inversion was applied before
the call.

**Why it was invisible in a same-frame self-test**: unprojecting and reprojecting a
frame with the *same* (even wrong) transform always round-trips near-perfectly onto
itself -- `self_test_reprojection.py --frame 0` passed with 0.0000px error despite the
bug being live, because it only tests one frame against itself. The bug only surfaces
when combining points **from different frames**, which is exactly what the merged
`point_cloud.ply` does. Confirmed by reprojecting the full 500-frame CS:GO
`point_cloud.ply` into frame 200's camera view: **0 of ~5.6M points landed inside the
frame** (`reproject_pointcloud.py`).

**Isolating and confirming the fix**: `self_test_reprojection.py` (uses
`closed_form_inverse_se3` to properly invert c2w->w2c before calling
`depth_to_world_coords_points`) passed cleanly, proving the *math* was right and the
bug was specifically the missing inversion in `export_point_cloud_ply`. Verified
`closed_form_inverse_se3_general` (used in the actual fix, batched variant) matches
`closed_form_inverse_se3` to ~1e-10 on the same data before trusting it. A focused
cross-frame test (frame 0's points unprojected buggy vs. fixed, both reprojected into
frame 20's camera) showed a real, visible shift between the two -- not just noise --
confirming the fix changes results meaningfully.

**Fix**: `export_point_cloud_ply` now inverts `predictions["extrinsic"]` via
`closed_form_inverse_se3_general` before passing it to
`unproject_depth_map_to_point_map`.

**Caveat -- fix is necessary but not sufficient**: even with the corrected transform,
frame 0's points reprojected into frame 20's view still didn't trace a recognizable
outline of frame 20's actual scene (just a small, roughly-positioned blob). Consistent
with the broader finding in this doc that the model's pose/scale accuracy on CS:GO is
limited (weak 0.282 correlation vs. independent OpenCV pose estimation) -- the bug fix
removes one real source of error, but doesn't fix the model's own pose-accuracy
ceiling on this content.

**Status**: all `point_cloud.ply` files generated before this fix landed (session date
2026-08-20) should be considered unreliable and regenerated. Trackmania's `results/0/`
was regenerated once already but that rerun started *before* the fix landed too --
needs another rerun. CS:GO and Youtube runs need re-verification of which ones went
through the fallback path (only affects clips that hit it) vs. used the model's native
`world_points` (unaffected).

## Camera trajectory + intrinsics

| | CS:GO (500 fr) | Trackmania/left_right 0 (72 fr) | Youtube cyberpunk (6000 fr) |
|---|---|---|---|
| Source | `hdf5_dm_july2021_1.hdf5` frames | `left_right/left_right/0.mp4` | `youtube/cyberpunk.mp4` (fps=20, first 5 min) |
| Translation x range | 2.77 | 0.89 | 15.10 |
| Translation y range | 0.48 | 0.40 | 4.90 |
| Translation z range | 2.79 | 1.47 | 14.03 |
| Trajectory path length | 45.9 | 3.75 | 2656.4 |
| fx range | 383.9–669.8 (mean 462.6, std 43.7) | 212.3–287.5 (mean 243.9, std 26.5) | 224.9–614.1 (mean 309.5, std 46.0) |
| fy range | 385.7–682.5 | 211.9–286.7 | 225.0–621.7 |
| cx / cy (fixed, from image size) | 259 / 140 | 259 / 147 | 259 / 147 |

**Ground-truth check (CS:GO only)**: CS:GO's competitive-play FOV is hardcoded at 90°
(horizontal, cannot be changed in-game). Expected fx at the processed width (518px,
`cx=259`) for 90° FOV: `fx = 259 / tan(45°) = 259.0`. The model's actual estimate is
roughly **1.8x too high** (mean fx=462.6, implying only a 58.5° FOV; range 383.9-669.8
implies 42.3-68.0°) -- so beyond being unstable frame-to-frame, the CS:GO intrinsics
estimate is also substantially wrong in an absolute sense, consistently
overestimating focal length / underestimating FOV.

**Observation**: all three runs show meaningful focal-length drift over the sequence
(std ~10-19% of mean fx) -- lingbot-map's per-frame intrinsic estimate isn't
converging to one stable value for any of these clips, not just the Trackmania one.
The Youtube run's large path length (2656) relative to its translation bounding box
(~15x5x14) suggests jittery/unstable frame-to-frame scale rather than a single clean
trajectory. Root cause not yet diagnosed -- could be low source resolution (CS:GO is
150x280 native), fast/blurry camera motion, or the model being out-of-distribution on
rendered game content vs its real-world training data.

**Ground-truth check (Trackmania)**: known camera-mode FOVs are Camera 1
(Standard/third-person) ~75-80°, Camera 2 (Close) ~65-70°, Camera 3 (First-Person)
~60°. The model's FOV estimate for `left_right/0.mp4` is mean **91.8°**, std 4.2°
(range 84.2-97.4°) -- a different failure pattern than CS:GO: here the model
**overestimates** FOV relative to every plausible camera mode (12-17° over even the
widest, Camera 1) but is comparatively *stable* frame-to-frame (std 4.2° vs. CS:GO's
up to 28.7°) -- confidently wrong rather than unstably wrong.

**OpenCV comparison (Trackmania, `left_right/0.mp4`)**: source frames weren't saved by
this run (predates the `rgb/` export), so frames were re-extracted from `0.mp4` at
fps=10 (matches the 72-frame `poses.json`) to run both comparisons.
- Pose: 71/71 frame pairs matched; per-frame relative rotation OpenCV mean=1.42°
  (std 1.57°) vs. model mean=1.52° (std 1.43°), mean absolute difference 0.78°,
  correlation **0.741** -- much stronger agreement than CS:GO's 0.282. Pose looks
  solid here.
- Intrinsics: OpenCV self-calibration mean=94.7° (std 17.0°) vs. model mean=91.5°
  (std 4.3°) -- means land within 3.2° of each other, but per-frame values don't
  actually track each other (correlation **0.027**, essentially uncorrelated) --
  agreement in the mean looks coincidental, not evidence the two methods are
  measuring the same thing frame-to-frame. Both are well above all three known
  camera-mode FOVs: vs. Camera 1 (75-80°) OpenCV is 17.2° over, model 14.0° over;
  errors grow further for Camera 2/3. Unlike CS:GO (where OpenCV clearly beat the
  model), here neither method matches the plausible ground-truth range, so either the
  clip uses a 4th/different camera mode than the three given, or both methods share
  a correlated failure mode on this content (e.g. NEAR the same lens-distortion or
  aspect-ratio assumption).

**OpenCV self-calibration comparison (CS:GO)**: `estimate_opencv_intrinsics.py` (focal-
length sweep + 3-view triangulate/solvePnP reprojection-error minimization, no
calibration target, no assumed ground truth) gives mean FOV **77.8°** (std 28.7°,
noisy -- 23 usable frames, 10° sweep grid) vs. model's 31.7° (same CS:GO run, std only
3.6° -- stable but wrong). Errors from the true 90°: OpenCV 26.1° vs. model 58.3°. The
independent classical method, despite its own coarseness, lands much closer to ground
truth than the model -- reinforcing that the model's intrinsics head is genuinely
miscalibrated on this content, not just imprecise.

**OpenCV pose (extrinsics) comparison (CS:GO)**: `compare_opencv_pose.py` (ORB +
essential matrix + recoverPose across consecutive frames, using the known-90°-FOV
intrinsics) vs. the model's own extrinsics: 478/499 frame pairs matched; per-frame
relative rotation OpenCV mean=6.09° (std 19.83°) vs. model mean=3.72° (std 10.26°),
mean absolute difference 5.35° (median 0.91°), correlation 0.282 (weak positive).
Unlike intrinsics, pose/extrinsics look roughly plausible -- weak-but-positive
agreement with an independent method, not a clear failure.

**Camera refinement iterations may matter**: a second CS:GO run (`rollout_000`)
produced a notably better FOV estimate (mean 58.8°, range 42.3-68.0°) than the first
(mean 31.7-33.9° across different frame subsets) -- consistent with
`--camera_num_iterations` differing between runs (1, used to dodge an OOM, vs. the
default 4). Not yet confirmed which iteration count `rollout_000` actually used --
worth verifying before concluding this is the lever.

## Confidence stats

Not available for these three runs (see note above). Re-run `run_csgo_example.bat`,
`run_trackmania_capture.bat`, or `run_youtube_depth.bat` (delete the existing
`poses.json` first so the skip-existing check doesn't short-circuit) to populate
`point_cloud_confidence` -- reports overall mean/std/min/max confidence, the fraction
of points kept at `--conf_threshold`, and per-frame `depth_conf_mean`.
