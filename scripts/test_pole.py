#!/usr/bin/env python3
"""
Pole-capture test wrapper for lingbot-map.

Opinionated defaults for short phone video walk-arounds of utility poles:
  - fps=8 (a slow walk, a few frames per second is enough)
  - sky masking on by default
  - confidence filter at the 50th percentile
  - dumps a binary .ply you can open in CloudCompare / MeshLab
  - prints scene extent, frame count, and (if present) EXIF GNSS stats as
    a first sanity check before investing in the full pipeline.

Usage:
    python scripts/test_pole.py \\
        --model_path /path/to/lingbot-map.pt \\
        --video_path /path/to/pole.MOV \\
        --output pole_cloud.ply

This is a Phase 0 smoke test. If the pole reconstructs as a clean vertical
structure here, we have a viable base for the full measurement pipeline
(see PLAN.md).
"""
import argparse
import os
import sys
import time
from argparse import Namespace

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from demo import load_images, load_model, postprocess


def _build_model_args(model_path, image_size=518):
    return Namespace(
        model_path=model_path,
        image_size=image_size,
        patch_size=14,
        mode="streaming",
        enable_3d_rope=True,
        max_frame_num=1024,
        num_scale_frames=8,
        kv_cache_sliding_window=64,
        camera_num_iterations=4,
        use_sdpa=False,
    )


def _write_ply_binary(path, xyz, rgb):
    """Minimal binary PLY writer — no trimesh dependency."""
    assert xyz.shape[0] == rgb.shape[0]
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    verts = np.empty(
        n,
        dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("r", "u1"), ("g", "u1"), ("b", "u1"),
        ],
    )
    verts["x"], verts["y"], verts["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    verts["r"], verts["g"], verts["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def _probe_exif_gnss(image_folder):
    """Report whether source frames carry GPS EXIF — required for Sim(3) scale recovery later."""
    try:
        from PIL import Image, ExifTags
    except ImportError:
        return None
    if not image_folder or not os.path.isdir(image_folder):
        return None
    sample = None
    for name in sorted(os.listdir(image_folder)):
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            sample = os.path.join(image_folder, name)
            break
    if sample is None:
        return None
    try:
        img = Image.open(sample)
        exif = img._getexif() or {}
        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        return tag_map.get("GPSInfo")
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="Phase 0 pole-capture test for lingbot-map.")
    p.add_argument("--model_path", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video_path")
    src.add_argument("--image_folder")
    p.add_argument("--output", default="pole_cloud.ply")
    p.add_argument("--fps", type=int, default=8, help="Frame sample rate when reading video.")
    p.add_argument("--first_k", type=int, default=None, help="Use only the first K frames.")
    p.add_argument("--conf_percentile", type=float, default=50.0,
                   help="Drop points below this world_points_conf percentile.")
    p.add_argument("--mask_sky", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply sky segmentation (recommended for outdoor pole shots).")
    p.add_argument("--image_size", type=int, default=518)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("Warning: CUDA unavailable. Inference will be slow and may OOM.")

    # 1) Frames
    t0 = time.time()
    images, paths, resolved_folder = load_images(
        image_folder=args.image_folder,
        video_path=args.video_path,
        fps=args.fps,
        first_k=args.first_k,
        image_size=args.image_size,
        patch_size=14,
    )
    print(f"Loaded {images.shape[0]} frames in {time.time() - t0:.1f}s (source: {resolved_folder})")

    # 2) GNSS sanity check (future scale solver input)
    gps = _probe_exif_gnss(resolved_folder)
    if gps:
        print("[scale] EXIF GPSInfo present — Sim(3) scale recovery will be feasible in Phase 1.")
    else:
        print("[scale] No EXIF GPSInfo on source frames. Reconstruction will be scale-ambiguous "
              "until a known reference or AR pose stream is supplied.")

    # 3) Model
    model = load_model(_build_model_args(args.model_path, args.image_size), device)

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=dtype)
    else:
        dtype = torch.float32

    images = images.to(device)
    n_frames = images.shape[0]
    keyframe_interval = 1 if n_frames <= 320 else (n_frames + 319) // 320

    # 4) Inference
    print(f"Running streaming inference: {n_frames} frames, "
          f"keyframe_interval={keyframe_interval}, dtype={dtype}")
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(
            images,
            num_scale_frames=8,
            keyframe_interval=keyframe_interval,
            output_device=torch.device("cpu"),
        )
    print(f"Inference finished in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(f"GPU peak: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # 5) Post-process (moves tensors to CPU; pops predictions['images'])
    images_for_post = predictions.get("images", images)
    predictions, images_cpu = postprocess(predictions, images_for_post)

    # 6) Point cloud
    pts = predictions["world_points"].numpy()          # (S, H, W, 3)
    conf = predictions["world_points_conf"].numpy()    # (S, H, W)
    imgs_np = images_cpu.numpy() if hasattr(images_cpu, "numpy") else np.asarray(images_cpu)
    # images_cpu is (S, 3, H, W) in [0, 1]
    rgb_full = np.transpose(imgs_np, (0, 2, 3, 1))

    xyz = pts.reshape(-1, 3)
    col = (rgb_full.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    c = conf.reshape(-1)

    thr = np.percentile(c, args.conf_percentile) if args.conf_percentile > 0 else 0.0
    keep = (c >= thr) & (c > 1e-5)
    xyz, col = xyz[keep], col[keep]
    print(f"Kept {keep.sum():,} / {keep.size:,} points "
          f"(>= p{args.conf_percentile:.0f} confidence).")

    if xyz.size == 0:
        print("No points survived filtering. Lower --conf_percentile or check input quality.")
        return

    # 7) Scene extent (sanity check — is anything reconstructed?)
    lo = np.percentile(xyz, 5, axis=0)
    hi = np.percentile(xyz, 95, axis=0)
    extent = hi - lo
    print(f"Scene 5–95% extent (model units): "
          f"x={extent[0]:.2f}  y={extent[1]:.2f}  z={extent[2]:.2f}")
    print("  (Units are scale-ambiguous until resolved from EXIF / AR / reference.)")

    # 8) Save
    _write_ply_binary(args.output, xyz.astype(np.float32), col)
    print(f"Wrote {args.output}  ({xyz.shape[0]:,} points)")
    print("Next: open in CloudCompare or MeshLab and verify the pole appears as a clean "
          "vertical structure. That's the Phase 0 gate.")


if __name__ == "__main__":
    main()
