#!/usr/bin/env python3
"""Batch inference demo for LingBot-MAP with offline video rendering.

Supported modes:
1. Batch process scenes under ``--input_folder``
2. Process a single ``--video_path`` by extracting frames first
3. Render one or more saved ``--load_predictions`` NPZ files to videos

This script is intentionally scoped to the current ``GCTStream`` model and its
``inference_streaming()`` / ``inference_windowed()`` APIs.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from demo import load_model, postprocess, prepare_for_visualization
from lingbot_map.vis.sky_segmentation import load_or_create_sky_masks
from lingbot_map.utils.load_fn import load_and_preprocess_images


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# render_cuda_ext ships both top-level .so extensions and an inner Python
# wrapper subpackage, so its directory must be on sys.path for the
# `from render_cuda_ext import voxelize_frame` import inside rgbd_render to
# resolve when this script invokes the render pipeline directly.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "render_cuda_ext"))


def compile_model(model, image_size=518, patch_size=14, num_scale_frames=8,
                  img_h=None, img_w=None):
    """Apply torch.compile(mode='reduce-overhead') to model sub-modules.

    Follows the three-phase strategy from the reference profiling script:
      1. Pre-warm RoPE/frequency caches with uncompiled forward passes
      2. Compile specific sub-modules (frame_blocks, DINOv2 blocks, global_block attn_pre/ffn_residual/proj)
      3. Run warmup passes with cudagraph_mark_step_begin() to capture CUDA graphs

    NOT compiled (tested regressions in reference):
      - depth_head: multi-scale upsampling causes too many graph captures (3x slowdown)
      - camera_head trunk: kv_cache dict changes every frame -> constant recompilation
      - norm1/ls1/ls2 separately: graph->eager crossing overhead > savings

    Args:
        model: GCTStream model (already on CUDA device, eval mode).
               Must use FlashInfer backend (not SDPA).
        image_size: Input image size (used as width, and for height if img_h is None)
        patch_size: Patch size (to compute valid image dimensions)
        num_scale_frames: Number of scale frames for warmup
        img_h: Actual image height (if known). Ensures CUDA graphs match real input shape.
        img_w: Actual image width (if known).
    """
    device = next(model.parameters()).device
    if device.type != "cuda":
        print("compile_model: skipping (not on CUDA)")
        return

    # torch.compile with CUDA graphs is only effective for FlashInfer backend.
    # SDPABlock lacks attn_pre/ffn_residual, and the dict-based KV cache causes
    # constant recompilation due to changing tensor identity.
    if getattr(model.aggregator, "use_sdpa", False):
        print("compile_model: skipping (SDPA backend; compile only benefits FlashInfer)")
        return

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)

    # Generate synthetic warmup images.
    # Use actual image dimensions if known, so CUDA graphs match real shapes.
    warmup_h = img_h if img_h is not None else (image_size // patch_size) * patch_size
    warmup_w = img_w if img_w is not None else image_size
    num_warmup = num_scale_frames + 1  # scale frames + 1 streaming frame
    warmup_images = torch.randn(1, num_warmup, 3, warmup_h, warmup_w, device=device, dtype=torch.float32)
    scale = num_scale_frames

    def _mark_step():
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    # ── Phase 1: Pre-warm caches (uncompiled) ────────────────────────────
    # RoPE caches cos/sin tables keyed on (dim, max_pos, device, dtype).
    # Populating before compile ensures CUDA graph capture sees cache hits
    # (no graph-owned tensors in the cache).
    print("  [compile] Phase 1: pre-warming caches (uncompiled)...")
    model.clean_kv_cache()
    with torch.no_grad(), autocast_ctx:
        _ = model.forward(
            warmup_images[:, :scale],
            num_frame_for_scale=scale,
            num_frame_per_block=scale,
            causal_inference=True,
        )
        _ = model.forward(
            warmup_images[:, scale:scale + 1],
            num_frame_for_scale=scale,
            num_frame_per_block=1,
            causal_inference=True,
        )
    torch.cuda.synchronize()

    # ── Phase 2: Compile sub-modules ──────────────────────────────────────
    print("  [compile] Phase 2: compiling sub-modules...")

    # frame_blocks: fixed input shape [B*S, P, C] -> CUDA graph safe
    for i, block in enumerate(model.aggregator.frame_blocks):
        model.aggregator.frame_blocks[i] = torch.compile(block, mode="reduce-overhead")
    print(f"    frame_blocks: {len(model.aggregator.frame_blocks)} blocks compiled")

    # DINOv2 encoder blocks: fixed shape [B*S, num_tokens, embed_dim]
    if hasattr(model.aggregator.patch_embed, "blocks"):
        dino_blocks = model.aggregator.patch_embed.blocks
        for i, block in enumerate(dino_blocks):
            model.aggregator.patch_embed.blocks[i] = torch.compile(block, mode="reduce-overhead")
        print(f"    DINOv2 blocks: {len(dino_blocks)} blocks compiled")

    # global_blocks: compile attn_pre (norm1+qkv+RoPE), ffn_residual (norm2+mlp+ls2), proj
    for block in model.aggregator.global_blocks:
        if hasattr(block, "attn_pre"):
            block.attn_pre = torch.compile(block.attn_pre, mode="reduce-overhead")
        if hasattr(block, "ffn_residual"):
            block.ffn_residual = torch.compile(block.ffn_residual, mode="reduce-overhead")
        block.attn.proj = torch.compile(block.attn.proj, mode="reduce-overhead")
    print(f"    global_blocks: {len(model.aggregator.global_blocks)} blocks (attn_pre + ffn_residual + proj)")

    # ── Phase 3: Warmup (trigger CUDA graph capture) ─────────────────────
    print("  [compile] Phase 3: warmup (capturing CUDA graphs)...")
    model.clean_kv_cache()

    # Scale-frame shape
    _mark_step()
    with torch.no_grad(), autocast_ctx:
        _ = model.forward(
            warmup_images[:, :scale],
            num_frame_for_scale=scale,
            num_frame_per_block=scale,
            causal_inference=True,
        )
    torch.cuda.synchronize()

    # Streaming shape (1 frame)
    _mark_step()
    with torch.no_grad(), autocast_ctx:
        _ = model.forward(
            warmup_images[:, scale:scale + 1],
            num_frame_for_scale=scale,
            num_frame_per_block=1,
            causal_inference=True,
        )
    torch.cuda.synchronize()

    model.clean_kv_cache()
    # Destroy the KV cache manager so it gets lazily re-created with the
    # correct tokens_per_frame when actual (possibly different-resolution) images arrive.
    model.aggregator.kv_cache_manager = None
    del warmup_images
    torch.cuda.empty_cache()
    print("  [compile] Done. CUDA graphs captured.")


def extract_frames_from_video(video_path, output_folder, fps=None):
    """Extract video frames into a folder and return saved frame paths."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps is not None and fps > 0:
        interval = max(1, round(src_fps / fps))
    else:
        interval = 1

    paths = []
    idx = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            path = os.path.join(output_folder, f"frame_{len(paths):06d}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
        idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    print(
        f"Extracted {len(paths)} frames from {video_path} "
        f"({total_frames} total, interval={interval})"
    )
    return paths


def _normalize_ext(ext):
    ext = ext.strip()
    if not ext:
        return ext
    return ext if ext.startswith(".") else f".{ext}"


def list_image_paths(image_folder, image_ext):
    """List image files in a folder."""
    paths = []
    for ext in image_ext.split(","):
        ext = _normalize_ext(ext)
        if not ext:
            continue
        paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(image_folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def apply_image_filters(paths, first_k=None, last_k=None, stride=1, image_range=None):
    """Apply range / stride / first_k / last_k filters to image paths."""
    if image_range:
        parts = image_range.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise ValueError(
                f"Invalid --image_range '{image_range}', expected start:end[:stride]"
            )
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else len(paths)
        range_stride = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        paths = paths[start:end:range_stride]
    else:
        if first_k is not None and first_k > 0:
            paths = paths[:first_k]
        if last_k is not None and last_k > 0:
            paths = paths[-last_k:]

    if stride > 1:
        paths = paths[::stride]

    return paths


def load_images_from_video(video_path, fps=None, target_frames=None,
                           image_size=518, patch_size=14, save_frames_dir=None,
                           first_k=None, stride=1):
    """Load and preprocess frames from video, with optional frame caching.

    If ``save_frames_dir`` is set:
    - First check if frames are already saved there (skip video decoding)
    - Otherwise extract, preprocess, and save frames as PNG for reuse

    If ``first_k`` is set, stop reading after collecting first_k frames.
    If ``stride > 1``, keep only every N-th frame after ``first_k`` (matches
    ``apply_image_filters`` semantics for image-folder input).  Cached / saved
    PNG frames are always the unstrided fps-extracted ones so the cache is
    reusable across different stride settings.
    """
    stride = max(1, int(stride))

    # Try loading cached frames
    if save_frames_dir is not None and os.path.isdir(save_frames_dir):
        cached = sorted(glob.glob(os.path.join(save_frames_dir, 'frame_*.png')))
        if cached:
            if first_k is not None and first_k > 0:
                cached = cached[:first_k]
            if stride > 1:
                cached = cached[::stride]
            print(f"Loading {len(cached)} cached frames from {save_frames_dir}"
                  + (f" (stride={stride})" if stride > 1 else ""))
            images = load_and_preprocess_images(
                cached, mode="crop", image_size=image_size, patch_size=patch_size,
            )
            h, w = images.shape[-2:]
            print(f"Preprocessed to {w}x{h} ({len(images)} frames)")
            return images

    # Decode video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps is None and target_frames is not None and target_frames > 0:
        fps = max(1, round(src_fps * target_frames / total_frames))
        print(f"Auto fps: {fps} (video {total_frames} frames @ {src_fps:.1f}fps -> ~{target_frames} target)")

    interval = max(1, round(src_fps / fps)) if fps is not None and fps > 0 else 1

    max_collect = first_k if (first_k is not None and first_k > 0) else float('inf')
    target_size = image_size
    save_dir_ready = False
    if save_frames_dir is not None:
        os.makedirs(save_frames_dir, exist_ok=True)
        save_dir_ready = True

    images = []
    idx = 0
    collected = 0
    pbar = tqdm(total=total_frames, desc="Reading video", unit="frame")
    while True:
        if idx % interval == 0:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save raw frame for reuse
            if save_dir_ready:
                cv2.imwrite(
                    os.path.join(save_frames_dir, f"frame_{collected:06d}.png"),
                    frame,  # already BGR from cap.read()
                )

            # Preprocess: resize + crop
            h, w = frame_rgb.shape[:2]
            new_width = target_size
            new_height = round(h * (new_width / w) / patch_size) * patch_size
            frame_resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            img = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
            if new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y:start_y + target_size, :]
            images.append(img)

            collected += 1
            if collected >= max_collect:
                pbar.update(1)
                break
        else:
            if not cap.grab():
                break
        idx += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    print(f"Read {collected} frames from {video_path} (interval={interval})")
    if save_dir_ready:
        print(f"Saved {collected} frames to {save_frames_dir}")

    images = torch.stack(images)
    if stride > 1:
        images = images[::stride]
        print(f"Applied stride={stride}: {collected} -> {len(images)} frames")
    h, w = images.shape[-2:]
    print(f"Preprocessed to {w}x{h} ({len(images)} frames)")
    return images


def load_images_from_paths(paths, image_size=518, patch_size=14, num_workers=8):
    """Load and preprocess images from an explicit path list."""
    if not paths:
        raise RuntimeError("No input images found")

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images


def find_image_folder(scene_path, image_ext):
    """Find a folder that contains images inside a scene path."""
    image_exts = [_normalize_ext(x).lower() for x in image_ext.split(",") if x.strip()]
    common_names = ["images", "imgs", "rgb", "color", "frames", "input", "raw"]

    def _has_images(folder):
        for ext in image_exts:
            if glob.glob(os.path.join(folder, f"*{ext}")):
                return True
            if glob.glob(os.path.join(folder, f"*{ext.upper()}")):
                return True
        return False

    if _has_images(scene_path):
        return scene_path

    for name in common_names:
        candidate = os.path.join(scene_path, name)
        if os.path.isdir(candidate) and _has_images(candidate):
            return candidate

    for root, _, _ in os.walk(scene_path):
        depth = root[len(scene_path):].count(os.sep)
        if depth > 2:
            continue
        if _has_images(root):
            return root

    return None


def find_scenes(input_folder, image_ext, min_images=2):
    """Find scenes under an input folder."""
    scenes = []

    root_image_folder = find_image_folder(input_folder, image_ext)
    if root_image_folder:
        count = len(list_image_paths(root_image_folder, image_ext))
        if count >= min_images:
            scenes.append((os.path.basename(os.path.abspath(input_folder)), root_image_folder, count))
            if root_image_folder == input_folder:
                return scenes

    for item in sorted(os.listdir(input_folder)):
        item_path = os.path.join(input_folder, item)
        if not os.path.isdir(item_path):
            continue
        image_folder = find_image_folder(item_path, image_ext)
        if image_folder is None:
            continue
        count = len(list_image_paths(image_folder, image_ext))
        if count >= min_images:
            scenes.append((item, image_folder, count))

    # Deduplicate by scene name / folder pair
    unique = []
    seen = set()
    for scene in scenes:
        key = (scene[0], scene[1])
        if key not in seen:
            seen.add(key)
            unique.append(scene)
    return unique


def run_inference(model, images, args):
    """Run model inference and return visualization-ready numpy predictions."""
    # Autocast must follow the *model* device, not the images.  Images may
    # intentionally live on CPU (very-long-sequence memory mode) and get moved
    # per-window inside the inference methods, but the model's matmuls still
    # run on CUDA in bf16/fp16.
    model_device = next(model.parameters()).device
    device_type = model_device.type
    autocast_ctx = nullcontext()
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        autocast_ctx = torch.amp.autocast("cuda", dtype=dtype)
        images_device = images.device.type
        print(f"Running {args.mode} inference (dtype={dtype}, "
            f"images on {images_device})...")
    else:
        print(f"Running {args.mode} inference on CPU...")

    flow_threshold = getattr(args, "flow_threshold", 0.0)
    max_non_keyframe_gap = getattr(args, "max_non_keyframe_gap", 30)

    if flow_threshold > 0:
        print(
            f"Flow-based keyframe mode: threshold={flow_threshold:.1f}px, "
            f"max_gap={max_non_keyframe_gap} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )
    elif args.keyframe_interval > 1:
        print(
            f"Fixed-interval keyframe mode: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    num_frames = images.shape[0] if images.dim() == 4 else images.shape[1]

    with torch.no_grad(), autocast_ctx:
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu")
            )
        else:
            # Estimate number of windows for fixed-interval mode
            if flow_threshold <= 0:
                ws = min(args.num_scale_frames, num_frames)
                kf_int = max(args.keyframe_interval, 1)
                # Match inference_windowed's eff_overlap resolution order.
                if args.overlap_keyframes is not None:
                    overlap = max(ws, args.overlap_keyframes * kf_int)
                elif args.overlap_size is not None:
                    overlap = args.overlap_size
                else:
                    overlap = ws
                phase2_kf = max(args.window_size - ws, 0)
                actual_win = ws + phase2_kf * kf_int
                eff_win = min(actual_win, num_frames)
                step = max(eff_win - overlap, 1)
                n_windows = max(1, (num_frames - overlap + step - 1) // step) if eff_win < num_frames else 1
                print(f"Windowed: {n_windows} windows (size={args.window_size}, overlap={overlap}, total={num_frames})")

            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                scale_mode=args.scale_mode,
                output_device=torch.device("cpu"),
                keyframe_interval=args.keyframe_interval,
                flow_threshold=flow_threshold,
                max_non_keyframe_gap=max_non_keyframe_gap,
            )

            # Report actual window count (especially useful for flow-based mode)
            chunk_scales = predictions.get("chunk_scales")
            if chunk_scales is not None:
                actual_windows = chunk_scales.shape[1] if chunk_scales.dim() >= 2 else chunk_scales.shape[0]
                print(f"Actual windows used: {actual_windows}")

    predictions, images_cpu = postprocess(predictions, images)
    return prepare_for_visualization(predictions, images_cpu)


def save_predictions_npz(predictions, output_path):
    """Save predictions as per-frame .npz files in a directory (parallel I/O).

    Creates a directory at ``output_path`` (without extension) and writes one
    ``.npz`` per frame: ``frame_000000.npz``, ``frame_000001.npz``, ...
    Each per-frame npz contains the slice along the sequence dimension for
    every array key.  Non-sequence scalars/metadata are saved in ``meta.npz``.

    This is much faster than a single large ``np.savez`` because each frame
    is small and frames are written in parallel.
    """
    from concurrent.futures import ThreadPoolExecutor

    dir_path = output_path
    if dir_path.endswith('.npz'):
        dir_path = dir_path[:-4]
    # Clean stale frame files from previous runs to avoid ghost frames
    if os.path.isdir(dir_path):
        old_frames = glob.glob(os.path.join(dir_path, 'frame_*.npz'))
        if old_frames:
            for f in old_frames:
                os.remove(f)
        old_meta = os.path.join(dir_path, 'meta.npz')
        if os.path.exists(old_meta):
            os.remove(old_meta)
    os.makedirs(dir_path, exist_ok=True)

    # Separate sequence arrays (have a frame dim) from metadata
    # Sequence arrays have shape (S, ...) or (B, S, ...) with B==1
    seq_keys = []
    meta_dict = {}
    S = None
    for key, value in predictions.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.ndim >= 2 and S is None:
            S = value.shape[0]
        if value.ndim >= 2 and value.shape[0] == S:
            seq_keys.append(key)
        else:
            meta_dict[key] = value

    if S is None:
        # Fallback: no sequence arrays found, save everything in one file
        save_dict = {k: v for k, v in predictions.items() if isinstance(v, np.ndarray)}
        np.savez(os.path.join(dir_path, "frame_000000.npz"), **save_dict)
        print(f"Predictions saved to {dir_path}/ (1 file, {len(save_dict)} keys)")
        return dir_path

    def _save_frame(frame_idx):
        frame_dict = {}
        for key in seq_keys:
            frame_dict[key] = predictions[key][frame_idx]
        np.savez(os.path.join(dir_path, f"frame_{frame_idx:06d}.npz"), **frame_dict)

    num_workers = min(32, S)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        list(pool.map(_save_frame, range(S)))

    # Save metadata (non-sequence arrays) separately
    if meta_dict:
        np.savez(os.path.join(dir_path, "meta.npz"), **meta_dict)

    print(f"Predictions saved to {dir_path}/ ({S} frames, {len(seq_keys)} keys/frame)")
    return dir_path


def load_predictions_from_npz(input_path, num_workers=None):
    """Load predictions from NPZ file or directory of per-frame npz files.

    Per-frame NPZ directories are loaded in parallel using a thread pool —
    np.load releases the GIL during disk I/O, so threads scale well even at
    moderate worker counts. Progress is reported via tqdm.

    Args:
        input_path: Path to .npz file or directory of per-frame .npz files.
        num_workers: Thread pool size (default: min(32, num_frames)).
    """
    from concurrent.futures import ThreadPoolExecutor

    if os.path.isdir(input_path):
        frame_files = sorted(glob.glob(os.path.join(input_path, 'frame_*.npz')))
        if not frame_files:
            # Try .npy fallback
            npy_files = sorted(glob.glob(os.path.join(input_path, '*.npy')))
            if npy_files:
                predictions = {}
                for path in tqdm(npy_files, desc="Loading .npy"):
                    key = os.path.splitext(os.path.basename(path))[0]
                    predictions[key] = np.load(path, allow_pickle=False)
            else:
                raise ValueError(f"No frame_*.npz or *.npy files found in {input_path}")
        else:
            # Parallel per-frame load
            n_frames = len(frame_files)
            workers = num_workers if num_workers is not None else min(32, n_frames)

            def _load_one(path):
                data = np.load(path, allow_pickle=False)
                return {key: data[key] for key in data.files}

            with ThreadPoolExecutor(max_workers=workers) as pool:
                frame_dicts = list(tqdm(
                    pool.map(_load_one, frame_files),
                    total=n_frames,
                    desc=f"Loading {n_frames} per-frame NPZs",
                    unit="frame",
                ))

            # Stack all keys (executes after all reads complete)
            all_keys = list(frame_dicts[0].keys())
            predictions = {}
            for key in all_keys:
                predictions[key] = np.stack([fd[key] for fd in frame_dicts], axis=0)

            # Load metadata if present
            meta_path = os.path.join(input_path, 'meta.npz')
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True)
                for key in meta.files:
                    predictions[key] = meta[key]
    else:
        data = np.load(input_path, allow_pickle=True)
        predictions = {key: data[key] for key in data.files}
    print(f"Loaded predictions from {input_path}")
    print(f"  Keys: {list(predictions.keys())}")
    return predictions


def resolve_artifact_dir(base_dir, artifact_name, use_subdirs):
    if base_dir is None:
        return None
    if use_subdirs:
        return os.path.join(base_dir, artifact_name)
    return base_dir


def get_sky_artifact_dirs(args, artifact_name):
    use_subdirs = getattr(args, "use_per_scene_sky_dirs", False)
    sky_mask_dir = resolve_artifact_dir(args.sky_mask_dir, artifact_name, use_subdirs)
    sky_mask_visualization_dir = resolve_artifact_dir(
        args.sky_mask_visualization_dir,
        artifact_name,
        use_subdirs,
    )
    return sky_mask_dir, sky_mask_visualization_dir


def visualize_sky_masks(
    args,
    artifact_name,
    image_folder=None,
    image_paths=None,
    images=None,
):
    if images is None and image_paths is not None:
        print("Loading resized images for sky mask visualization...")
        loaded_images = load_images_from_paths(
            image_paths,
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_workers=args.num_workers,
        )
        images = loaded_images.numpy()

    sky_mask_dir, sky_mask_visualization_dir = get_sky_artifact_dirs(args, artifact_name)
    if sky_mask_visualization_dir is None and sky_mask_dir is None:
        raise ValueError(
            "Sky mask visualization requested, but neither --sky_mask_visualization_dir "
            "nor --sky_mask_dir is set"
        )

    num_frames = None
    if image_paths is not None:
        num_frames = len(image_paths)
    elif images is not None:
        num_frames = images.shape[0]

    sky_masks = load_or_create_sky_masks(
        image_folder=image_folder,
        image_paths=image_paths,
        images=images,
        skyseg_model_path=args.skyseg_model_path,
        sky_mask_dir=sky_mask_dir,
        sky_mask_visualization_dir=sky_mask_visualization_dir,
        num_frames=num_frames,
    )
    if sky_masks is None:
        raise RuntimeError("Sky segmentation failed")

    print(f"Sky masks ready: {sky_masks.shape[0]} frames")
    if sky_mask_dir is not None:
        print(f"  Sky mask dir: {sky_mask_dir}")
    if sky_mask_visualization_dir is not None:
        print(f"  Sky mask visualization dir: {sky_mask_visualization_dir}")

    return {
        "sky_mask_dir": sky_mask_dir,
        "sky_mask_visualization_dir": sky_mask_visualization_dir,
        "num_sky_masks": int(sky_masks.shape[0]),
    }


def export_glb(predictions, output_path, args):
    """Export predictions to GLB."""
    from lingbot_map.vis import predictions_to_glb

    scene = predictions_to_glb(
        predictions,
        conf_thres=args.conf_threshold,
        filter_by_frames="all",
        mask_sky=args.mask_sky,
        target_dir=os.path.dirname(output_path),
        prediction_mode="Predicted Pointmap",
    )
    scene.export(output_path)
    print(f"GLB saved to {output_path}")


def render_with_pipeline(npz_path, output_video, args, artifact_name=None):
    """Render NPZ predictions to video using the rgbd_render pipeline.

    Args:
        npz_path: Path to NPZ file or per-frame NPZ directory.
        output_video: Output video path (.mp4).
        args: Parsed CLI args (batch_demo argparse namespace).
        artifact_name: Scene name for sky mask sub-directories.

    Returns:
        True if rendering succeeded.
    """
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    from rgbd_render.config import (
        PipelineConfig, CameraSegment, _build_segments_from_flat_args,
        _set_nested,
    )
    from rgbd_render.camera import build_camera_path
    from rgbd_render.overlay import build_overlays
    from rgbd_render.pipeline.builder import SceneBuilder
    from rgbd_render.pipeline.offline import OfflinePipeline

    def _log(msg):
        tqdm.write(f"[render] {msg}")

    # YAML preset (if any) seeds defaults; CLI flags override below.
    yaml_path = getattr(args, 'config', None)
    cfg = PipelineConfig.from_yaml(yaml_path) if yaml_path else PipelineConfig()
    user_set = getattr(args, '_user_supplied', None)
    if user_set is None:
        # Legacy callers that don't run main() — apply every CLI value.
        user_set = set(vars(args).keys())

    def _apply(cfg_path, arg_name):
        """Override cfg field from args.

        - Without --config: always applies the CLI value (preserves legacy
          behavior, including baked-in argparse defaults like point_size=0.1).
        - With --config: only applies when the user explicitly passed the
          flag, so YAML values survive untouched otherwise.
        Args set to None are always skipped.
        """
        val = getattr(args, arg_name, None)
        if val is None:
            return
        if yaml_path and arg_name not in user_set:
            return
        _set_nested(cfg, cfg_path, val)

    cfg.input = npz_path
    cfg.output = output_video
    cfg.fast_review = 0

    # Pipeline / I/O
    _apply('fps', 'video_fps')
    _apply('frame_stride', 'render_stride')

    # Render settings
    _apply('render.width', 'video_width')
    _apply('render.height', 'video_height')
    _apply('render.point_size', 'point_size')

    # Preprocess settings
    _apply('preprocess.vis_threshold', 'vis_threshold')
    _apply('preprocess.conf_threshold', 'conf_threshold')
    _apply('preprocess.mask_sky', 'mask_sky')
    _apply('preprocess.sky_model', 'skyseg_model_path')
    artifact_name = artifact_name or os.path.splitext(os.path.basename(output_video))[0]
    sky_mask_dir, sky_mask_viz_dir = get_sky_artifact_dirs(args, artifact_name)
    cfg.preprocess.sky_mask_dir = sky_mask_dir
    cfg.preprocess.sky_mask_visualization_dir = sky_mask_viz_dir

    # Camera segments — keep YAML segments untouched if the user didn't
    # pass any segment-shaping flag.
    _SEG_ARGS = {
        'camera_mode', 'smooth_window', 'back_offset', 'up_offset',
        'look_offset', 'follow_scale_frames', 'birdeye_start',
        'birdeye_duration', 'reveal_height_mult',
    }
    keep_yaml_segments = (
        yaml_path and bool(cfg.camera.segments) and not (user_set & _SEG_ARGS)
    )
    if not keep_yaml_segments:
        camera_mode = getattr(args, 'camera_mode', 'follow') or 'follow'
        has_birdeye_inserts = bool(getattr(args, 'birdeye_start', None)) \
            and bool(getattr(args, 'birdeye_duration', None))

        if camera_mode == 'follow' and has_birdeye_inserts:
            # follow base + birdeye inserts (multi-segment); reuse config helper
            # so this matches rgbd_scan_render.py's behavior 1:1.
            cfg.camera.segments = _build_segments_from_flat_args({
                'smooth_window': args.smooth_window,
                'back_offset': args.back_offset,
                'up_offset': args.up_offset,
                'look_offset': args.look_offset,
                'follow_scale_frames': args.follow_scale_frames,
                'birdeye_start': args.birdeye_start,
                'birdeye_duration': args.birdeye_duration,
                'reveal_height_mult': args.reveal_height_mult,
            })
        else:
            # Single segment covering the whole video
            seg_kwargs = {'mode': camera_mode, 'frames': [0, -1]}
            if camera_mode == 'follow':
                seg_kwargs.update(
                    back_offset=args.back_offset,
                    up_offset=args.up_offset,
                    look_offset=args.look_offset,
                    scale_frames=args.follow_scale_frames,
                )
                if args.smooth_window is not None:
                    seg_kwargs['smooth_window'] = args.smooth_window
            elif camera_mode == 'birdeye' and args.reveal_height_mult is not None:
                seg_kwargs['reveal_height_mult'] = args.reveal_height_mult
            # static / pivot: leave eye/lookat=None → auto-derived from start frame
            cfg.camera.segments = [CameraSegment(**seg_kwargs)]

    _apply('camera.fov', 'fov')
    _apply('camera.transition', 'birdeye_transition')

    # Scene settings
    _apply('scene.downsample', 'downsample_factor')
    cfg.scene.keyframes_only_points = bool(getattr(args, 'keyframes_only_points', False))

    # Overlay settings
    _apply('overlay.camera_vis', 'camera_vis')
    if getattr(args, 'trail_color_ramp', None):
        cfg.overlay.trail_color_ramp = args.trail_color_ramp
    if getattr(args, 'trail_line_width', None) is not None:
        cfg.overlay.trail_line_width = args.trail_line_width
    if getattr(args, 'trail_tail_len', None) is not None:
        cfg.overlay.trail_tail_len = args.trail_tail_len
    if getattr(args, 'head_num_frames', None) is not None:
        cfg.overlay.head_num_frames = args.head_num_frames
    if getattr(args, 'head_point_size', None) is not None:
        cfg.overlay.head_point_size = args.head_point_size
    if getattr(args, 'head_frustum_scale', None) is not None:
        cfg.overlay.head_frustum_scale = args.head_frustum_scale
    if getattr(args, 'head_frustum_line_width', None) is not None:
        cfg.overlay.head_frustum_line_width = args.head_frustum_line_width
    if getattr(args, 'head_frustum_color', None):
        cfg.overlay.head_frustum_color = args.head_frustum_color
    if getattr(args, 'head_texture_alpha', None) is not None:
        cfg.overlay.head_texture_alpha = args.head_texture_alpha
    cfg.overlay.frame_tag = bool(getattr(args, 'frame_tag', False))
    if getattr(args, 'frame_tag_position', None):
        cfg.overlay.frame_tag_position = args.frame_tag_position

    scene = SceneBuilder(cfg, log=_log).load().preprocess().voxelize().build()
    camera_path = build_camera_path(cfg.camera, scene)
    overlays, overlay_specs = build_overlays(cfg, scene)

    _log(f"{len(scene.sorted_xyz):,} voxels, {scene.num_frames} frames")
    _log(f"{len(camera_path)} camera frames")

    OfflinePipeline(scene, camera_path, overlays, cfg,
                    overlay_specs=overlay_specs, log=_log).run()
    scene.destroy()
    return True


def process_scene(args, scene_name, image_folder, model, device, video_images=None):
    """Process one scene: inference, optional NPZ / GLB, video render.

    Args:
        video_images: Pre-loaded video frames tensor (if from --video_path).
                      None means load from image_folder.
    """
    result = {
        "scene_name": scene_name,
        "image_folder": image_folder,
        "success": False,
        "error": None,
        "duration": 0.0,
    }
    start_time = time.time()

    video_path = os.path.join(args.output_folder, f"{scene_name}{args.video_suffix}.mp4")
    npz_path = os.path.join(args.output_folder, f"{scene_name}.npz")
    glb_path = os.path.join(args.output_folder, f"{scene_name}.glb")
    result["output_video"] = video_path
    result["output_npz"] = npz_path if args.save_predictions else None
    result["output_glb"] = glb_path if args.save_glb else None

    try:
        print(f"\n{'=' * 60}")
        print(f"Processing scene: {scene_name}")
        print(f"{'=' * 60}")

        image_paths = None
        if video_images is not None:
            images = video_images
            num_frames = images.shape[0]
            print(f"  Frames: {num_frames} (loaded from video)")
        else:
            image_paths = _get_filtered_image_paths(args, image_folder)
            if not image_paths:
                raise ValueError("No images found after applying filters")
            num_frames = len(image_paths)
            print(f"  Image folder: {image_folder}")
            print(f"  Frames: {num_frames}")
            images = load_images_from_paths(
                image_paths,
                image_size=args.image_size,
                patch_size=args.patch_size,
                num_workers=args.num_workers,
            )
        # Keep images on CPU; inference_streaming / inference_windowed move
        # per-window (or per-frame) slices to the model device just-in-time.
        # This avoids OOM on very long sequences (tens of thousands of frames)
        # where the full tensor would exceed GPU memory.
        if device.type == "cuda":
            # Pinned memory makes per-slice .to(cuda, non_blocking=True) fast.
            images = images.pin_memory() if not images.is_pinned() else images

        t_infer = time.time()
        predictions = run_inference(model, images, args)
        t_infer = time.time() - t_infer
        print(f"Inference done in {t_infer:.1f}s ({num_frames / max(t_infer, 1e-6):.1f} FPS)")

        sky_mask_dir, sky_mask_visualization_dir = get_sky_artifact_dirs(args, scene_name)
        result["sky_mask_dir"] = sky_mask_dir
        result["sky_mask_visualization_dir"] = sky_mask_visualization_dir

        # Always save NPZ (render pipeline reads from disk).
        # save_predictions_npz strips the .npz suffix and writes to a directory
        # of per-frame files; it returns the actual output path.
        saved_npz_path = save_predictions_npz(predictions, npz_path)

        if args.save_glb:
            export_glb(predictions, glb_path, args)

        if not args.no_render:
            print(f"Rendering video to {video_path}...")
            if not render_with_pipeline(
                saved_npz_path, video_path, args, artifact_name=scene_name,
            ):
                raise RuntimeError("Video rendering failed")

        # Clean up NPZ if user didn't explicitly request saving
        if not args.save_predictions and os.path.exists(saved_npz_path):
            if os.path.isdir(saved_npz_path):
                shutil.rmtree(saved_npz_path, ignore_errors=True)
            else:
                os.remove(saved_npz_path)

        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Error processing {scene_name}: {exc}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["duration"] = time.time() - start_time

    return result


def render_npz_file(args, npz_path):
    """Render one NPZ file to video / GLB."""
    result = {
        "npz_path": npz_path,
        "success": False,
        "error": None,
        "duration": 0.0,
    }
    start_time = time.time()

    try:
        predictions = load_predictions_from_npz(npz_path)
        name = os.path.splitext(os.path.basename(npz_path))[0]
        video_path = os.path.join(args.output_folder, f"{name}{args.video_suffix}.mp4")
        glb_path = os.path.join(args.output_folder, f"{name}.glb")
        result["output_video"] = video_path
        result["output_glb"] = glb_path if args.save_glb else None

        if args.save_glb:
            export_glb(predictions, glb_path, args)

        if not args.no_render:
            print(f"Rendering {npz_path} -> {video_path}")
            if not render_with_pipeline(npz_path, video_path, args, artifact_name=name):
                raise RuntimeError("Video rendering failed")

        result["success"] = True
    except Exception as exc:
        result["error"] = str(exc)
        print(f"Error rendering {npz_path}: {exc}")
    finally:
        result["duration"] = time.time() - start_time

    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch streaming/windowed inference + headless video rendering for LingBot-MAP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/batch_demo.py \\
      --input_folder /data/scenes \\
      --output_folder /data/outputs \\
      --model_path /path/to/checkpoint.pt --mode streaming

  python examples/batch_demo.py \\
      --video_path /data/video.mp4 --fps 10 \\
      --output_folder /data/outputs \\
      --model_path /path/to/checkpoint.pt --mode windowed --window_size 64 --overlap_size 16

  python examples/batch_demo.py \\
      --load_predictions pred1.npz pred2.npz \\
      --output_folder /data/outputs
""",
    )

    parser.add_argument("--input_folder", type=str, default=None, help="Root folder containing scene folders")
    parser.add_argument("--video_path", type=str, default=None, help="Input video path")
    parser.add_argument("--fps", type=int, default=None, help="Target FPS for frame extraction from video")
    parser.add_argument("--target_frames", type=int, default=None,
                        help="Target number of frames to extract from video (auto-computes fps)")
    parser.add_argument("--save_frames_dir", type=str, default=None,
                        help="Save extracted video frames as PNG for reuse. "
                             "If the directory already has frames, skip video decoding.")
    parser.add_argument("--load_predictions", type=str, nargs="+", default=None, help="Render saved NPZ predictions")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML preset to seed render/scene/camera/overlay defaults "
             "(e.g. demo_render/config/indoor.yaml). CLI flags still override "
             "any value the user explicitly passes.",
    )

    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument(
        "--mode",
        type=str,
        default="streaming",
        choices=["streaming", "windowed"],
        help="Inference mode: streaming with KV cache, or windowed with overlap alignment",
    )
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--keyframe_interval", type=int, default=1)
    parser.add_argument(
        "--flow_threshold",
        type=float,
        default=0.0,
        help="Flow-based keyframe threshold in pixels. >0 enables flow-based mode "
             "(takes precedence over --keyframe_interval). 0 = disabled.",
    )
    parser.add_argument(
        "--max_non_keyframe_gap",
        type=int,
        default=100,
        help="Max consecutive non-keyframe frames before forcing a keyframe (flow mode only)",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--kv_cache_scale_frames", type=int, default=8)
    parser.add_argument("--use_sdpa", action="store_true", default=False)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Apply torch.compile(mode='reduce-overhead') to frame_blocks, DINOv2 blocks, "
            "and global_block sub-modules for CUDA graph acceleration",
    )
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window in windowed mode")
    parser.add_argument("--overlap_size", type=int, default=None,
                        help="Overlap between adjacent windows in *actual frames*. "
                             "Default: num_scale_frames (overlap = scale frames)")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in *keyframes* (takes precedence over "
                             "--overlap_size). Converted internally to "
                             "max(num_scale_frames, overlap_keyframes * keyframe_interval) "
                             "actual frames.  Recommended when --keyframe_interval > 1.")
    parser.add_argument(
        "--scale_mode",
        type=str,
        default="median",
        choices=["median", "trimmed_mean", "median_all", "trimmed_mean_all"],
        help="Scale estimation mode for window alignment",
    )

    parser.add_argument("--image_extension", type=str, default=".jpg,.jpeg,.png")
    parser.add_argument("--image_stride", type=int, default=1)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--last_k", type=int, default=None)
    parser.add_argument("--image_range", type=str, default=None, help="start:end[:stride]")
    parser.add_argument("--min_images", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--use_point_map", action="store_true")
    parser.add_argument("--downsample_factor", type=int, default=5)
    parser.add_argument("--keyframes_only_points", action="store_true",
                        help="Only unproject keyframe depth into the point cloud. "
                             "Non-keyframes still contribute their camera pose to the "
                             "trajectory/frustum overlay.  Requires the NPZ to carry "
                             "is_keyframe / frame_type (produced by batch_demo.py "
                             "--save_predictions).")
    parser.add_argument("--render_downsample", type=int, default=1)
    parser.add_argument("--point_size", type=float, default=0.1)
    parser.add_argument("--vis_threshold", type=float, default=1.5)
    parser.add_argument("--conf_threshold", type=float, default=0.0)
    parser.add_argument("--mask_sky", action="store_true")
    parser.add_argument("--skyseg_model_path", type=str, default="skyseg.onnx")
    parser.add_argument("--sky_mask_dir", type=str, default=None)
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None)
    parser.add_argument(
        "--visualize_sky_mask_only",
        action="store_true",
        help="Generate sky masks / overlay visualizations and exit without model inference",
    )
    parser.add_argument(
        "--camera_mode",
        type=str,
        default="follow",
        choices=["follow", "birdeye", "static", "pivot"],
        help="Camera segment mode applied to the entire video unless "
             "--birdeye_start/--birdeye_duration carve out birdeye inserts. "
             "follow=trajectory chase cam, birdeye=top-down reveal, "
             "static=fixed eye+lookat, pivot=fixed eye with moving target.",
    )
    parser.add_argument(
        "--zoom_out_local_frames",
        type=int,
        default=50,
        help="[DEPRECATED no-op] zoom_out mode is not implemented in rgbd_render. "
             "Use --camera_mode birdeye, or --birdeye_start/--birdeye_duration "
             "to insert birdeye segments inside a follow path.",
    )
    # Camera segment / path params (mirror rgbd_scan_render.py)
    parser.add_argument("--fov", type=float, default=None,
                        help="Camera FoV in degrees (default 60)")
    parser.add_argument("--smooth_window", type=int, default=None,
                        help="Follow-cam trajectory smoothing window (default 40)")
    parser.add_argument("--birdeye_start", type=str, default=None,
                        help="Comma-separated frame indices for birdeye segment starts "
                             "(e.g. '100,500'). Requires --camera_mode follow.")
    parser.add_argument("--birdeye_duration", type=str, default=None,
                        help="Comma-separated durations matching --birdeye_start "
                             "(e.g. '60,80')")
    parser.add_argument("--birdeye_transition", type=int, default=None,
                        help="Transition frames blending adjacent segments (default 30)")
    parser.add_argument("--reveal_height_mult", type=float, default=None,
                        help="Birdeye camera height multiplier (default 2.0)")
    # Camera / trajectory overlay
    parser.add_argument(
        "--camera_vis",
        type=str,
        default="",
        choices=["", "default", "frustum", "textured", "trail"],
        help="Camera overlay preset.  '' disables the overlay (default).  "
            "'default'=trail+points, 'frustum'=trail+wireframe frusta, "
            "'textured'=trail+image-textured frusta, 'trail'=trail only.",
    )
    parser.add_argument("--trail_color_ramp", type=str, default=None,
                        choices=["cyan_blue", "white", "rainbow", "red",
                                "green", "yellow", "magenta"],
                        help="Trajectory trail color ramp")
    parser.add_argument("--head_frustum_scale", type=float, default=None,
                        help="Frustum wireframe size as a fraction of scene scale "
                            "(frustum/textured presets)")
    parser.add_argument("--head_frustum_color", type=str, default=None,
                        help="Frustum wireframe color (hex or name); '' = follow trail newest color")
    parser.add_argument("--trail_line_width", type=float, default=None,
                        help="Trajectory trail line width")
    parser.add_argument("--trail_tail_len", type=int, default=None,
                        help="Number of past frames drawn as the trail tail")
    parser.add_argument("--head_num_frames", type=int, default=None,
                        help="Number of recent frames rendered as the camera head")
    parser.add_argument("--head_point_size", type=float, default=None,
                        help="Point size for camera head points")
    parser.add_argument("--head_frustum_line_width", type=float, default=None,
                        help="Frustum wireframe line width")
    parser.add_argument("--head_texture_alpha", type=float, default=None,
                        help="Alpha for image-textured frustum (textured preset)")
    parser.add_argument("--frame_tag", action="store_true",
                        help="Stamp '<i> / <N> Frames' counter on the output video")
    parser.add_argument("--frame_tag_position", type=str, default=None,
                        choices=["top_left", "top_right", "bottom_left", "bottom_right"],
                        help="Frame counter position (only with --frame_tag)")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--video_width", type=int, default=1920)
    parser.add_argument("--video_height", type=int, default=1080)
    parser.add_argument("--video_suffix", type=str, default="_pointcloud")
    parser.add_argument("--save_original_video", action="store_true", default=True)
    parser.add_argument("--no_original_video", action="store_true")
    parser.add_argument("--max_render_points", type=int, default=1_000_000_000)
    parser.add_argument("--render_stride", type=int, default=1)
    parser.add_argument("--camera_stride", type=int, default=1)
    parser.add_argument(
        "--point_cloud_stride",
        type=int,
        default=1,
        help="Only sample points from every Nth frame into the point cloud while keeping camera frustums",
    )
    parser.add_argument("--back_offset", type=float, default=0.3)
    parser.add_argument("--up_offset", type=float, default=0.1)
    parser.add_argument("--look_offset", type=float, default=0.5)
    parser.add_argument(
        "--follow_scale_frames",
        type=int,
        default=0,
        help="Use first N frames to estimate follow-camera local scale; 0 uses the full scene",
    )

    parser.add_argument("--save_predictions", action="store_true", help="Save per-scene predictions to NPZ")
    parser.add_argument("--save_glb", action="store_true", help="Export GLB alongside video")
    parser.add_argument("--no_render", action="store_true", help="Skip video rendering")
    parser.add_argument("--npz_image_folder", type=str, default=None, help="Optional image folder for NPZ sky masking")

    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--scenes", type=str, nargs="+", default=None)
    parser.add_argument("--exclude_scenes", type=str, nargs="+", default=None)

    return parser


def _save_results_json(output_folder, results, **extra_fields):
    """Save batch results to JSON and return the number of failures."""
    results_path = os.path.join(output_folder, "batch_results.json")
    payload = {"timestamp": datetime.now().isoformat(), **extra_fields, "results": results}
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {results_path}")
    return sum(1 for r in results if not r.get("success", False))


def _get_filtered_image_paths(args, image_folder):
    """List and filter image paths from a folder using args filters."""
    paths = list_image_paths(image_folder, args.image_extension)
    return apply_image_filters(
        paths,
        first_k=args.first_k,
        last_k=args.last_k,
        stride=args.image_stride,
        image_range=args.image_range,
    )


def _discover_scenes(args, parser, video_images=None):
    """Discover and filter scenes from input_folder or video, returning (scenes, video_images)."""
    if args.video_path is not None:
        if not os.path.isfile(args.video_path):
            parser.error(f"Video file does not exist: {args.video_path}")
        video_name = os.path.splitext(os.path.basename(args.video_path))[0]

        save_frames_dir = args.save_frames_dir
        if save_frames_dir is None:
            save_frames_dir = os.path.join(args.output_folder, f"{video_name}_frames")
        video_images = load_images_from_video(
            args.video_path,
            fps=args.fps,
            target_frames=args.target_frames,
            image_size=args.image_size,
            patch_size=args.patch_size,
            save_frames_dir=save_frames_dir,
            first_k=args.first_k,
            stride=args.image_stride,
        )
        scenes = [(video_name, args.output_folder, video_images.shape[0])]
    else:
        if not os.path.isdir(args.input_folder):
            parser.error(f"Input folder does not exist: {args.input_folder}")
        scenes = find_scenes(args.input_folder, args.image_extension, min_images=args.min_images)

    # Apply scene name filters
    if args.scenes:
        scenes = [s for s in scenes if s[0] in args.scenes]
    if args.exclude_scenes:
        scenes = [s for s in scenes if s[0] not in args.exclude_scenes]

    # Skip existing outputs
    filtered = []
    for scene_name, image_folder, image_count in scenes:
        output_video = os.path.join(args.output_folder, f"{scene_name}{args.video_suffix}.mp4")
        output_npz = os.path.join(args.output_folder, f"{scene_name}.npz")
        should_skip = (
            not args.visualize_sky_mask_only
            and args.skip_existing
            and (
                (not args.no_render and os.path.exists(output_video))
                or (args.save_predictions and os.path.exists(output_npz))
            )
        )
        if should_skip:
            print(f"Skipping {scene_name} (output exists)")
        else:
            filtered.append((scene_name, image_folder, image_count))

    return filtered, video_images


def _run_load_predictions_mode(args):
    """Handle --load_predictions mode: render or visualize saved NPZ files."""
    args.use_per_scene_sky_dirs = len(args.load_predictions) > 1
    results = []
    for npz_path in args.load_predictions:
        if not os.path.exists(npz_path):
            results.append({"npz_path": npz_path, "success": False, "error": "File not found"})
            continue
        output_video = os.path.join(
            args.output_folder,
            f"{os.path.splitext(os.path.basename(npz_path))[0]}{args.video_suffix}.mp4",
        )
        if (
            args.skip_existing
            and not args.visualize_sky_mask_only
            and not args.no_render
            and os.path.exists(output_video)
        ):
            print(f"Skipping {npz_path} (output exists)")
            continue

        if args.visualize_sky_mask_only:
            result = {"npz_path": npz_path, "success": False, "error": None, "duration": 0.0}
            start_time = time.time()
            try:
                predictions = load_predictions_from_npz(npz_path)
                artifact_name = os.path.splitext(os.path.basename(npz_path))[0]
                image_paths = (
                    _get_filtered_image_paths(args, args.npz_image_folder)
                    if args.npz_image_folder else None
                )
                result.update(visualize_sky_masks(
                    args, artifact_name,
                    image_folder=args.npz_image_folder,
                    image_paths=image_paths,
                    images=predictions.get("images"),
                ))
                result["success"] = True
            except Exception as exc:
                result["error"] = str(exc)
                print(f"Error visualizing sky masks for {npz_path}: {exc}")
            finally:
                result["duration"] = time.time() - start_time
            results.append(result)
        else:
            results.append(render_npz_file(args, npz_path))

    mode = "visualize_sky_mask_only" if args.visualize_sky_mask_only else "load_predictions"
    return _save_results_json(args.output_folder, results, mode=mode)


def _run_sky_mask_only_mode(args, scenes):
    """Handle --visualize_sky_mask_only mode for discovered scenes."""
    results = []
    total_start = time.time()
    for scene_name, image_folder, _ in tqdm(scenes, desc="Visualizing sky masks"):
        result = {"scene_name": scene_name, "image_folder": image_folder,
                  "success": False, "error": None, "duration": 0.0}
        start_time = time.time()
        try:
            image_paths = _get_filtered_image_paths(args, image_folder)
            if not image_paths:
                raise ValueError("No images found after applying filters")
            result.update(visualize_sky_masks(
                args, scene_name, image_folder=image_folder, image_paths=image_paths,
            ))
            result["success"] = True
        except Exception as exc:
            result["error"] = str(exc)
            print(f"Error visualizing sky masks for {scene_name}: {exc}")
        finally:
            result["duration"] = time.time() - start_time
        results.append(result)

    return _save_results_json(
        args.output_folder, results,
        input_folder=args.input_folder,
        skyseg_model_path=args.skyseg_model_path,
        total_duration=time.time() - total_start,
        mode="visualize_sky_mask_only",
    )


def _run_inference_mode(args, parser, scenes, video_images=None):
    """Handle main inference mode: load model, optionally compile, process scenes."""
    if args.model_path is None:
        parser.error("--model_path is required for inference modes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args, device)

    if args.compile and device.type == "cuda":
        print("Applying torch.compile acceleration...")
        compile_h, compile_w = None, None
        if video_images is not None:
            compile_h, compile_w = video_images.shape[-2], video_images.shape[-1]
        compile_model(
            model,
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_scale_frames=args.num_scale_frames,
            img_h=compile_h,
            img_w=compile_w,
        )

    results = []
    total_start = time.time()
    for scene_name, image_folder, _ in tqdm(scenes, desc="Processing scenes"):
        results.append(process_scene(args, scene_name, image_folder, model, device,
                                     video_images=video_images))

    return _save_results_json(
        args.output_folder, results,
        input_folder=args.input_folder,
        model_path=args.model_path,
        total_duration=time.time() - total_start,
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Track which CLI args the user explicitly passed (vs. argparse defaults).
    # Used by render_with_pipeline so a YAML preset (--config) only governs
    # values the user did not override on the command line.
    _arg_defaults = {a.dest: a.default for a in parser._actions
                     if a.dest not in ('help',)}
    args._user_supplied = {
        k for k, v in vars(args).items()
        if k != '_user_supplied' and v != _arg_defaults.get(k, None)
    }

    if args.no_original_video:
        args.save_original_video = False

    if args.load_predictions is None and args.input_folder is None and args.video_path is None:
        parser.error("One of --input_folder, --video_path, or --load_predictions is required")

    os.makedirs(args.output_folder, exist_ok=True)
    if args.visualize_sky_mask_only and args.sky_mask_visualization_dir is None:
        args.sky_mask_visualization_dir = os.path.join(args.output_folder, "sky_mask_visualizations")

    # ── Mode 1: Render / visualize saved predictions ──────────────────────
    if args.load_predictions is not None:
        sys.exit(_run_load_predictions_mode(args))

    # ── Discover scenes from video or folder ──────────────────────────────
    scenes, video_images = _discover_scenes(args, parser)
    args.use_per_scene_sky_dirs = len(scenes) > 1

    if not scenes:
        print("No scenes to process.")
        sys.exit(0)

    print("Scenes to process:")
    for idx, (scene_name, image_folder, image_count) in enumerate(scenes, start=1):
        print(f"  {idx}. {scene_name} ({image_count} images) -> {image_folder}")

    if args.dry_run:
        sys.exit(0)

    # ── Mode 2: Sky mask visualization only ───────────────────────────────
    if args.visualize_sky_mask_only:
        sys.exit(_run_sky_mask_only_mode(args, scenes))

    # ── Mode 3: Model inference + rendering ───────────────────────────────
    sys.exit(_run_inference_mode(args, parser, scenes, video_images=video_images))


if __name__ == "__main__":
    main()
