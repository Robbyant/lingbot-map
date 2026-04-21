# Utility Pole Measurement & Classification — Plan

A vision-first tool for measuring and classifying utility poles from phone
video or photo bursts, built on top of the `lingbot-map` streaming 3D
reconstruction model. No LiDAR, no RTK GNSS, no measuring stick — the
Tesla-style approach: cameras only, resolve the rest in software.

---

## 1. What `lingbot-map` provides (the base)

`lingbot-map` is a feed-forward 3D reconstruction foundation model (a
successor to VGGT, DINOv2 backbone + geometric-context transformer). From a
folder of images or a video, a single forward pass produces, per frame:

- `extrinsic` — camera pose (c2w) in a common world frame
- `intrinsic` — focal / principal point
- `depth` + `depth_conf` — dense per-pixel depth
- `world_points` + `world_points_conf` — dense per-pixel 3D points in the
  shared world frame
- ~20 FPS streaming at 518×378; windowed mode for long sequences

This is exactly the primitive needed for vision-first pole measurement:
**camera poses + dense world-frame point cloud from images alone**.
`demo.py` already handles video → frames → point cloud → viser viewer
(`demo.py:45-100`, `demo.py:370-426`). The gaps for a pole-measurement tool
are metric-scale resolution, segmentation/classification, measurement UI,
and georeferencing.

## 2. Why vision-first ("Tesla approach") fits

Existing utility-pole tools (Katapult Pro, IKE GeoSpatial, SPIDA, Pointivo,
Osmose, Alden One) rely on one of:

- **LiDAR + GNSS** (Katapult "stick", IKE): accurate, expensive, requires
  field crews/trucks.
- **Measuring stick + manual photo annotation** (Katapult, SPIDA field):
  cheap but slow and error-prone.
- **RTK GNSS + photogrammetry** (Pointivo): semi-automated, still needs
  survey-grade hardware.

A vision-first pipeline replaces all of those with *"walk around the pole
with a phone, upload, get measurements."* The tradeoff is monocular metric
scale, which is solvable with cheap cues (see §4).

## 3. End-to-end pipeline

```
User capture (phone video / photo burst)
        │
        ▼
[1] Frame extraction + pose estimation (lingbot-map)   ← exists
        │   per-frame extrinsics, intrinsics, depth, world_points
        ▼
[2] Metric scale resolution                            ← NEW
        │   uplift reconstruction to real-world meters
        ▼
[3] Pole detection & segmentation (SAM 2 + DINOv2)     ← NEW
        │   2D masks per frame: pole body, crossarm, attachments
        ▼
[4] Mask → 3D fusion                                   ← NEW
        │   3D point-cloud subsets per object class
        ▼
[5] Pole-axis fit + ground-plane fit (RANSAC)          ← NEW
        │   vertical pole line in world frame
        ▼
[6] Measurements                                        ← NEW
        │   pole height, attachment heights, lean, diameter
        ▼
[7] Classification                                      ← NEW
        │   pole material/class, attachment types
        ▼
[8] Export                                              ← NEW
        │   GeoJSON / KML / Katapult-compatible JSON, PDF report
        ▼
[9] Browser app (extend viser viewer into a measurement UI)
```

## 4. Solving metric scale (the critical unlock)

Monocular feed-forward reconstruction is **scale-ambiguous** by default.
Four viable cues, in order of preference — support all, fall back
gracefully:

1. **Phone AR session data (ARKit / ARCore)** — if users capture via a
   companion mobile app, visual-inertial odometry gives metric translations.
   Best UX.
2. **EXIF GNSS baseline** — modern phones embed per-photo lat/lon/alt. Fit
   Sim(3) between `lingbot-map` camera translations and EXIF positions;
   recovers scale when the user walks ≥1–2 m around the pole. Zero extra
   effort for users.
3. **Known reference in scene** — user taps two points on a standard
   reference (meter stick, printed AprilTag, known crossarm length). Manual
   one-time calibration per capture.
4. **Learned pole-diameter prior** — mean class diameter (e.g., Class-4 40′
   wood ≈ 9.5″ at 6′ above grade) as a weak prior when nothing else
   exists; flagged as "estimated" in the report.

**Recommendation:** ship #2 as default + #3 as a tap-to-calibrate override.

## 5. Where Meta models fit

Beyond the DINOv2 trunk already inside `lingbot-map`, three Meta models
slot in cleanly:

- **SAM 2** — video-native segmentation. User clicks the pole once in one
  frame; SAM 2 propagates the mask across the whole video. Same for each
  attachment. This is the unlock for labor-free 2D → 3D object isolation.
- **DINOv2 features (reused from the aggregator)** — attach a small
  linear/MLP head for pole classification (wood/steel/concrete/composite,
  class 1–6) and attachment-type classification. No new backbone needed;
  we tap into features already computed during reconstruction. Significant
  compute win.
- **CoTracker3** (optional) — dense point tracking for cross-frame mask
  consistency and pole-lean / pole-sway detection between frames.

Non-Meta but complementary: **Depth Anything v2** as a fallback for
single-still uploads (`lingbot-map` needs ≥2 views).

## 6. Measurement primitives

Once we have a scaled point cloud + per-class masks:

- **Ground plane** — RANSAC plane fit restricted to points within R meters
  of the pole base, normal near +Z.
- **Pole axis** — RANSAC 3D line through pole-masked points with
  near-vertical prior.
- **Pole height** — distance along axis from ground-plane intersection to
  top-of-pole cluster.
- **Attachment height** — for each attachment segment, project centroid (or
  lowest point for clearance) onto axis, report distance from ground.
- **Pole lean** — angle between pole axis and gravity (phone IMU if
  available, else ground-plane normal).
- **Diameter at breast height** — cylinder fit on pole points in the 4.5′
  band.
- **Uncertainty** — propagate `depth_conf` / `world_points_conf` through
  every measurement; report ±σ.

## 7. Georeferencing & export

- Anchor the world frame to WGS84 via the Sim(3) solved in §4 step 2
  (EXIF GNSS). Every measured 3D point gets lat/lon/alt.
- Export targets:
  - **GeoJSON / KML** — QGIS, Google Earth, ArcGIS.
  - **Katapult Pro-compatible JSON** — reverse-engineer from a sample
    export; this is the integration that matters for utilities already on
    Katapult.
  - **SPIDAcalc XML** — for structural-loading handoff.
  - **PDF inspection report** — annotated photos + measurement table,
    signable.

## 8. Competitive positioning

| Tool            | Input                  | Scale source         | Automation | Cost                 |
|-----------------|------------------------|----------------------|------------|----------------------|
| Katapult Pro    | Photos + measuring stick | Manual stick        | Low        | Seat + field labor   |
| IKE GeoSpatial  | LiDAR + RTK GNSS       | RTK GNSS             | Medium     | Hardware + service   |
| Pointivo        | Multi-image + RTK      | RTK GNSS             | Medium     | Hardware + service   |
| SPIDA field     | Photo + stick          | Manual               | Low        | Labor                |
| **This tool**   | **Phone video**        | **EXIF / AR / tag**  | **High**   | **App only**         |

Pitch: **"90% of Katapult's measurement output from a 30-second phone
walk-around, no stick, no RTK"** — with graceful accuracy degradation
rather than a hard hardware gate.

## 9. Phased build

### Phase 0 — Verify the base (1–2 days)
- Run `demo.py` (and `scripts/test_pole.py`) on a real utility-pole video.
- Confirm dense point-cloud quality, pose stability, and whether the pole
  reconstructs as a clean vertical structure.
- **This is the first gate.** Thin vertical structures are a known hard
  case for feed-forward MVS. If the pole is disintegrating, evaluate:
  stage-1 VGGT checkpoint, keyframe density, windowed mode, or pairing
  with a monocular depth prior.

### Phase 1 — Measurement MVP (2–3 weeks)
- EXIF GNSS Sim(3) scale solver.
- SAM 2 single-click pole segmentation (video-propagated).
- Ground-plane + pole-axis fit, height measurement, single attachment
  height.
- Extend the viser viewer with a "measure" panel: click-to-measure on the
  3D point cloud, ruler overlay.

### Phase 2 — Classification & bulk attachments (3–4 weeks)
- DINOv2-head classifier for pole material + class and attachment types.
  Start zero-shot via CLIP for the long tail, fine-tune on a small labeled
  set as data accrues.
- Multi-attachment segmentation (SAM 2 + detector bootstrapped from
  CLIP/Grounding-DINO prompts like "transformer", "insulator",
  "streetlight").
- Confidence/uncertainty reporting.

### Phase 3 — Georeferencing & export (2 weeks)
- WGS84 anchoring, GeoJSON/KML export, PDF report, Katapult JSON.

### Phase 4 — Capture app + cloud (ongoing)
- Thin iOS/Android capture app recording video + ARKit/ARCore pose + GNSS,
  uploads to a cloud worker running the pipeline; web app shows
  interactive 3D + measurements.

## 10. Risks to validate early

- **Thin vertical structures in feed-forward MVS.** Biggest unknown.
  Validate in Phase 0 before committing to the architecture.
- **Metric scale drift on short baselines.** Users standing still won't
  give a GNSS baseline. Detect and prompt the user to walk around the pole.
- **Occluded bases.** Grass, fences, parked cars. Ground-plane fit needs
  robust handling; may require a user-assisted "tap the ground" step.
- **Daylight / backlighting.** Poles against bright sky. Sky masking
  exists (`demo.py:286`); extend to general over-exposure handling.
- **Accuracy claims.** Utilities often need survey-grade (±0.1 ft) for
  make-ready engineering. Be explicit about precision tier — position the
  tool as *pre-survey triage + inventory* initially, not as a replacement
  for RTK-grade engineering input.

---

## Phase 0 runbook (local)

```bash
# Env
conda create -n lingbot-map python=3.10 -y
conda activate lingbot-map
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[vis]"

# Checkpoint (one-time, several GB)
huggingface-cli download robbyant/lingbot-map --local-dir ./checkpoints

# Test wrapper: dumps a .ply and prints sanity-check stats
python scripts/test_pole.py \
    --model_path ./checkpoints/lingbot-map.pt \
    --video_path /path/to/pole.MOV \
    --output pole_cloud.ply

# Open pole_cloud.ply in CloudCompare or MeshLab to inspect.
```
