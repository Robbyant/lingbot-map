"""LingBot-MAP demo: streaming 3D reconstruction from images or video.

Usage:
    # Streaming inference (frame-by-frame with KV cache)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/

    # Streaming inference with keyframe KV caching
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --image_folder /path/to/images/ --mode streaming --keyframe_interval 6

    # Windowed inference (for very long sequences, >500 frames)
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10 --mode windowed --window_size 64

    # From video with custom FPS sampling
    python examples/demo.py --model_path /path/to/checkpoint.pt \
        --video_path video.mp4 --fps 10
"""

import argparse
import glob
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Must be set before `import torch` / any CUDA init. Reduces the reserved-vs-allocated
# memory gap by letting the caching allocator grow segments on demand instead of
# pre-reserving fixed-size blocks.
#
# Caveat: `expandable_segments:True` is **incompatible** with torch.compile's
# `cudagraph_trees` (PyTorch ≤2.8) — checkpoint pool state restore assumes the
# classic fixed-segment topology, and trips
# `RuntimeError: Expected curr_block->next == nullptr` during compiled warmup
# / replay. So when `--compile` is requested we skip the env override and let
# the default allocator run.
if "--compile" not in sys.argv:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png,.JPG",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8,
                rotate_clockwise_90=False, start_frame=0):
    """Load images from folder or video and preprocess into a tensor.

    Returns:
        (images, paths, resolved_image_folder): preprocessed tensor, file paths,
        and the folder containing the source images (for sky mask caching etc.).
    """
    if video_path is not None:
        # Read the video directly and sample every interval-th frame IN MEMORY as a
        # PIL Image -- no per-frame temp .jpg written to disk (the old approach broke
        # on a full disk: a write would silently fail mid-extraction, then loading
        # would crash later with FileNotFoundError on the missing frame). first_k is
        # enforced DURING decode as a hard cap, so a long clip is never fully decoded
        # (a 14-min video would otherwise sample thousands of frames and OOM the model).
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        # Seek past the first `start_frame` source frames (skip intros/menus) before sampling.
        if start_frame and start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
            print(f"Seeking to source frame {int(start_frame)} "
                  f"(~{start_frame / src_fps:.1f}s in) before sampling")
        cap_k = first_k if (first_k is not None and first_k > 0) else None
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Reading video", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                # Downsize immediately, before appending -- holding thousands of
                # full-resolution frames in `saved` until the final crop/resize step
                # exhausts system RAM on long clips. load_and_preprocess_images still
                # does the exact crop/resize to image_size; this is just a cheap
                # pre-shrink so the in-memory frame list stays small.
                h, w = frame.shape[:2]
                if max(h, w) > image_size:
                    scale = image_size / max(h, w)
                    frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                saved.append(Image.fromarray(frame_rgb))
                if cap_k is not None and len(saved) >= cap_k:
                    break
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        resolved_folder = None  # no on-disk frame folder -- sky-mask caching needs a real path if used
        print(f"Read {len(paths)} frames from video "
              f"(sampled every {interval} of {total_frames}, cap={cap_k})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        if start_frame and start_frame > 0:
            paths = paths[int(start_frame):]  # skip the first N images (start further into the set)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

    if rotate_clockwise_90:
        rotated_dir = tempfile.mkdtemp(prefix="lingbot_rot_cw90_")
        rotated_paths = []
        # Image.ROTATE_270 = lossless 90° clockwise (270° counter-clockwise) reordering.
        for p in tqdm(paths, desc="Rotating images 90° CW"):
            out_path = os.path.join(rotated_dir, os.path.basename(p))
            Image.open(p).transpose(Image.ROTATE_270).save(out_path)
            rotated_paths.append(out_path)
        paths = rotated_paths
        resolved_folder = rotated_dir
        print(f"Rotated {len(paths)} images 90° clockwise → {rotated_dir}")

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths, resolved_folder


# =============================================================================
# Model loading
# =============================================================================

def load_model(args, device):
    """Load GCTStream model from checkpoint."""
    if getattr(args, "mode", "streaming") == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    print("Building model...")
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )

    if args.model_path:
        print(f"Loading checkpoint: {args.model_path}")
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
        print("  Checkpoint loaded.")

    return model.to(device).eval()


# =============================================================================
# torch.compile (opt-in via --compile)
# =============================================================================

def compile_model(model):
    """Compile hot, fixed-shape modules with mode="reduce-overhead".

    Mirrors the targets in gct_profile.py:compile_model. Unlike the profile script,
    `model.point_head` is **kept** — the demo needs world_points for visualization.
    """
    agg = model.aggregator
    for i, b in enumerate(agg.frame_blocks):
        agg.frame_blocks[i] = torch.compile(b, mode="reduce-overhead")
    for i, b in enumerate(agg.patch_embed.blocks):
        agg.patch_embed.blocks[i] = torch.compile(b, mode="reduce-overhead")
    for b in agg.global_blocks:
        if hasattr(b, 'attn_pre'):
            b.attn_pre = torch.compile(b.attn_pre, mode="reduce-overhead")
        if hasattr(b, 'ffn_residual'):
            b.ffn_residual = torch.compile(b.ffn_residual, mode="reduce-overhead")
        b.attn.proj = torch.compile(b.attn.proj, mode="reduce-overhead")


def _warm_streaming(model, images, scale_frames, warm_stream_n, dtype,
                    passes=1, keyframe_interval=1):
    """Drive `clean_kv_cache → Phase 1 → N streaming forwards` `passes` times.

    Warmup inputs are sliced from the already-preprocessed ``images`` tensor, so
    their **spatial shape (H×W) and number of scale frames adapt to the user's
    input** — this is what makes the captured CUDA graphs match what
    ``inference_streaming`` will replay (reduce-overhead mode keys on shape).

    The streaming loop alternates keyframe / non-keyframe forwards according to
    ``keyframe_interval``, mirroring ``inference_streaming``'s call pattern so
    the ``skip_append`` (defer+append+attend+rollback) path is also captured
    during warmup.  Without this, the first non-keyframe in the real run hits
    cold orchestration code and can confuse cudagraph_trees' allocator
    checkpoint state.
    """
    num_avail = int(images.shape[0])
    scale_frames = max(1, min(int(scale_frames), num_avail))
    # Keep at least one streaming frame for the per-frame compile path; if the
    # user supplied <= scale_frames images, shrink scale to free a stream slot.
    if scale_frames >= num_avail:
        scale_frames = max(1, num_avail - 1)
    warm_stream_n = max(1, min(int(warm_stream_n), num_avail - scale_frames))
    kf_int = max(int(keyframe_interval), 1)

    # images: [S, 3, H, W] on device already; slice + add batch dim, no copy of
    # spatial dims so warmup shape == real inference shape (H, W).
    warm_scale = images[:scale_frames].unsqueeze(0).to(dtype)
    warm_stream = images[scale_frames:scale_frames + warm_stream_n].unsqueeze(0).to(dtype)

    for _ in range(passes):
        model.clean_kv_cache()
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            model.forward(
                warm_scale,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=scale_frames,
                causal_inference=True,
            )
        for i in range(warm_stream_n):
            is_keyframe = (kf_int <= 1) or (i % kf_int == 0)
            if not is_keyframe:
                model._set_skip_append(True)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                model.forward(
                    warm_stream[:, i:i + 1],
                    num_frame_for_scale=scale_frames,
                    num_frame_per_block=1,
                    causal_inference=True,
                )
            if not is_keyframe:
                model._set_skip_append(False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Wipe warmup KV so real inference_streaming starts clean (it also calls
    # clean_kv_cache internally, but this is defensive + makes intent obvious).
    model.clean_kv_cache()


# =============================================================================
# Post-processing
# =============================================================================

_BATCHED_NDIMS = {
    "pose_enc": 3,
    "depth": 5,
    "depth_conf": 4,
    "world_points": 5,
    "world_points_conf": 4,
    "extrinsic": 4,
    "intrinsic": 4,
    "chunk_scales": 2,
    "chunk_transforms": 4,
    "images": 5,
}


def _squeeze_single_batch(key, value):
    """Drop the leading batch dimension for single-sequence demo outputs."""
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions, images):
    """Convert pose encoding to extrinsics (c2w) and move to CPU."""
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    # Convert w2c to c2w
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    extrinsic = extrinsic_4x4[..., :3, :4]

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)

    print("Moving results to CPU...")
    for k in list(predictions.keys()):
        if isinstance(predictions[k], torch.Tensor):
            predictions[k] = _squeeze_single_batch(
                k, predictions[k].to("cpu", non_blocking=True)
            )
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return predictions, images_cpu


def override_intrinsics_with_known_fov(predictions, fov_deg, img_w=None):
    """Replace the model's estimated fx/fy with the value implied by a known
    ground-truth horizontal FOV (e.g. CS:GO's fixed 90 deg), assuming square pixels
    (fx == fy). cx/cy are left as-is (already just the image center). Also drops any
    precomputed world_points so export_point_cloud_ply's depth-based fallback
    re-derives the point cloud using the corrected intrinsics instead of stale ones."""
    intrinsic = predictions["intrinsic"]  # (S, 3, 3)
    if img_w is None:
        img_w = predictions["depth"].shape[2]  # (S, H, W, 1)
    fx_corrected = (img_w / 2) / np.tan(np.radians(fov_deg / 2))
    intrinsic[:, 0, 0] = fx_corrected
    intrinsic[:, 1, 1] = fx_corrected
    predictions.pop("world_points", None)
    predictions.pop("world_points_conf", None)
    print(f"Overrode fx/fy to {fx_corrected:.1f} (from {fov_deg} deg known FOV, "
          f"width={img_w}); dropped precomputed world_points to force depth-based re-unprojection.")


def export_point_cloud_ply(predictions, images, output_path, conf_threshold=1.5, downsample_factor=10):
    """Export a merged (all-frames) point cloud as an ASCII PLY, colored from images,
    with per-point confidence as a 4th vertex property. Returns confidence summary stats."""
    if "world_points" in predictions:
        world_points = predictions["world_points"].numpy()  # (S, H, W, 3)
        conf = predictions.get("world_points_conf")
        if conf is None:
            conf = predictions["depth_conf"]
        conf = conf.numpy()  # (S, H, W)
    else:
        # Point head output wasn't kept (e.g. some checkpoints/paths only produce
        # depth) -- unproject depth via pose+intrinsics instead, same fallback the
        # interactive viewer uses by default (use_point_map=False).
        # unproject_depth_map_to_point_map -> depth_to_world_coords_points requires a
        # w2c (cam-from-world) extrinsic, but predictions["extrinsic"] is c2w (see
        # postprocess()'s "Convert w2c to c2w" comment) -- invert first, same as
        # self_test_reprojection.py, or every point lands in the wrong place.
        extrinsic_w2c = closed_form_inverse_se3_general(predictions["extrinsic"])[:, :3, :]
        world_points = unproject_depth_map_to_point_map(
            predictions["depth"], extrinsic_w2c, predictions["intrinsic"]
        )
        conf = predictions["depth_conf"].numpy()  # (S, H, W)
    colors = images.numpy().transpose(0, 2, 3, 1)  # (S, H, W, 3), float in [0, 1]

    pts = world_points.reshape(-1, 3)
    conf_flat = conf.reshape(-1)
    colors_flat = colors.reshape(-1, 3)

    stats = {
        "all_points_conf_mean": float(conf_flat.mean()),
        "all_points_conf_std": float(conf_flat.std()),
        "all_points_conf_min": float(conf_flat.min()),
        "all_points_conf_max": float(conf_flat.max()),
        "conf_threshold_used": conf_threshold,
    }

    mask = conf_flat >= conf_threshold
    pts = pts[mask]
    conf_kept = conf_flat[mask]
    colors_flat = (colors_flat[mask] * 255).clip(0, 255).astype(np.uint8)

    stats["num_points_total"] = int(conf_flat.shape[0])
    stats["num_points_kept"] = int(pts.shape[0])
    stats["kept_fraction"] = float(pts.shape[0] / max(conf_flat.shape[0], 1))
    stats["kept_points_conf_mean"] = float(conf_kept.mean()) if conf_kept.size else None
    stats["kept_points_conf_std"] = float(conf_kept.std()) if conf_kept.size else None

    if downsample_factor > 1:
        pts = pts[::downsample_factor]
        conf_kept = conf_kept[::downsample_factor]
        colors_flat = colors_flat[::downsample_factor]

    with open(output_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float confidence\n")
        f.write("end_header\n")
        for p, c, cf in zip(pts, colors_flat, conf_kept):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} {cf}\n")

    print(f"Exported {pts.shape[0]} points to {output_path} "
          f"(kept {stats['kept_fraction']*100:.1f}% at conf>={conf_threshold}, "
          f"conf mean={stats['all_points_conf_mean']:.3f} std={stats['all_points_conf_std']:.3f})")
    return stats


def export_results(predictions, images, output_dir, conf_threshold=1.5, downsample_factor=10,
                   comparison_stride=20):
    """Export extrinsic/intrinsic + point-cloud confidence stats to poses.json,
    depth maps as grayscale PNGs, and a merged point cloud PLY (confidence per point)."""
    output_dir = Path(output_dir)
    depth_dir = output_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    extrinsic = predictions["extrinsic"].numpy()  # (S, 3, 4), c2w
    intrinsic = predictions["intrinsic"].numpy()  # (S, 3, 3)
    depth = predictions["depth"].numpy()  # (S, H, W, 1)
    depth_conf = predictions.get("depth_conf")
    depth_conf = depth_conf.numpy() if depth_conf is not None else None

    conf_stats = export_point_cloud_ply(
        predictions, images, output_dir / "point_cloud.ply",
        conf_threshold=conf_threshold, downsample_factor=downsample_factor,
    )

    img_w = images.shape[-1]
    translations = extrinsic[:, :, 3]  # (S, 3), c2w camera positions

    def _frame_entry(i):
        fx, fy = float(intrinsic[i, 0, 0]), float(intrinsic[i, 1, 1])
        entry = {
            "frame": i,
            "extrinsic": extrinsic[i].tolist(),
            "intrinsic": intrinsic[i].tolist(),
            "fx": fx,
            "fy": fy,
            "fov_h_deg": float(np.degrees(2 * np.arctan((img_w / 2) / fx))) if fx > 0 else None,
            "camera_position": translations[i].tolist(),
            "depth_conf_mean": float(depth_conf[i].mean()) if depth_conf is not None else None,
        }
        if i > 0:
            entry["translation_delta"] = float(np.linalg.norm(translations[i] - translations[i - 1]))
        return entry

    poses = {
        "note": "extrinsic is camera-to-world (c2w), 3x4. depth PNGs are per-frame "
                "min/max normalized to 0-255 for visualization -- not metric distance. "
                "fov_h_deg is the horizontal FOV implied by fx and the processed image "
                "width -- compare against a known ground-truth FOV (e.g. CS:GO's fixed "
                "90 deg) to sanity-check the model's intrinsics estimate. "
                "point_cloud.ply carries a per-vertex 'confidence' property (raw model "
                "confidence, unrelated to the 0-255 depth PNG normalization); "
                "point_cloud_confidence summarizes it as an uncertainty proxy -- lower "
                "confidence / higher spread means less trustworthy geometry in that region.",
        "point_cloud_confidence": conf_stats,
        "frames": [_frame_entry(i) for i in range(extrinsic.shape[0])],
    }
    with open(output_dir / "poses.json", "w") as f:
        json.dump(poses, f, indent=2)

    # Every 20th frame also gets its RAW (non-normalized) depth saved as .npy, so
    # reprojection/self-consistency tests have real metric-scale values to work with
    # -- the visualization PNG below is min/max-normalized per frame and lossy, not
    # usable for reconstructing 3D points.
    depth_raw_dir = output_dir / "depth_raw"
    depth_raw_dir.mkdir(parents=True, exist_ok=True)
    for i in range(depth.shape[0]):
        d = depth[i, ..., 0]
        d_min, d_max = d.min(), d.max()
        d_norm = (d - d_min) / (d_max - d_min + 1e-8)
        d_u8 = (d_norm * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(depth_dir / f"{i:06d}.png"), d_u8)
        if i % comparison_stride == 0:
            np.save(str(depth_raw_dir / f"{i:06d}.npy"), d)

    # Every 20th source RGB frame, same {i:06d}.png naming as depth/, for eyeballing
    # a depth prediction against its actual input frame side by side.
    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    images_np = images.numpy()  # (S, 3, H, W), float in [0, 1]
    num_rgb_saved = 0
    for i in range(0, images_np.shape[0], comparison_stride):
        frame = (images_np[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(rgb_dir / f"{i:06d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        num_rgb_saved += 1

    print(f"Exported {extrinsic.shape[0]} poses to {output_dir / 'poses.json'}")
    print(f"Exported {depth.shape[0]} depth maps to {depth_dir}")
    print(f"Exported {num_rgb_saved} comparison RGB frames (every {comparison_stride}) to {rgb_dir}")
    print(f"Exported {num_rgb_saved} raw depth arrays (every {comparison_stride}) to {depth_raw_dir}")


def prepare_for_visualization(predictions, images=None):
    """Convert predictions to the unbatched NumPy format used by vis code."""
    vis_predictions = {}
    for k, v in predictions.items():
        if isinstance(v, torch.Tensor):
            v = _squeeze_single_batch(k, v.detach().cpu())
            vis_predictions[k] = v.numpy()
        elif isinstance(v, np.ndarray):
            vis_predictions[k] = _squeeze_single_batch(k, v)
        else:
            vis_predictions[k] = v

    if images is None:
        images = predictions.get("images")

    if isinstance(images, torch.Tensor):
        images = images.detach().cpu()
    if isinstance(images, np.ndarray):
        images = _squeeze_single_batch("images", images)
    elif isinstance(images, torch.Tensor):
        images = _squeeze_single_batch("images", images).numpy()

    if isinstance(images, torch.Tensor):
        images = images.numpy()

    if images is not None:
        vis_predictions["images"] = images

    return vis_predictions


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LingBot-MAP: Streaming 3D Reconstruction Demo")

    # Input
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--first_k", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Skip this many source frames before sampling (video only) -- "
                            "e.g. to jump past an intro/menu. first_k still counts from there.")
    parser.add_argument("--rotate_clockwise_90", action="store_true",
                        help="Rotate source images 90° clockwise before preprocessing "
                             "(crop/resize then operates on the rotated aspect ratio)")

    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)

    # Inference mode
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "windowed"],
                        help="streaming: frame-by-frame with KV cache; windowed: overlapping windows for long sequences")

    # Streaming options
    parser.add_argument("--enable_3d_rope", action="store_true", default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument(
        "--keyframe_interval",
        type=int,
        default=None,
        help="Every N-th frame after scale frames is kept as a keyframe. 1 = every frame. "
            "Streaming: if unset, auto-selected (1 when num_frames <= 320, else ceil(num_frames / 320)) "
            "to bound KV cache. Windowed: defaults to 1; --window_size counts keyframes, so values >1 "
            "expand each window's actual-frame coverage to "
            "scale_frames + (window_size - scale_frames) * keyframe_interval.",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=32,
                        help="KV-cache sliding window (frames). Default 32 keeps 500+ frame streaming "
                            "within 32GB VRAM; raise to 64 for a bit more temporal context if you have headroom.")
    parser.add_argument("--camera_num_iterations", type=int, default=1,
                        help="Camera head iterative-refinement steps. Default 1 (fits 32GB + faster); "
                            "raise to 4 for slightly higher pose accuracy if VRAM allows.")
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="torch.compile hot modules (reduce-overhead) with a CUDA-graph warmup. "
                            "Streaming mode only; ~5 FPS faster at 518x378. Adds ~30-60 s warmup time.")
    parser.add_argument(
        "--offload_to_cpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Offload per-frame predictions to CPU during inference to cut GPU peak memory "
            "(on by default).  Use --no-offload_to_cpu to keep outputs on GPU.",
    )
    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=16,
                        help="Overlap between windows in *actual frames*")
    parser.add_argument("--overlap_keyframes", type=int, default=None,
                        help="Overlap expressed in *keyframes* (takes precedence over "
                             "--overlap_size). Converted internally to "
                             "max(num_scale_frames, overlap_keyframes * keyframe_interval) "
                             "actual frames.  Recommended when --keyframe_interval > 1.")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_viser", action="store_true", default=False,
                        help="Skip the blocking viser point-cloud viewer (for headless/self-test "
                            "runs that just need the exported results, then exit).")
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--comparison_stride", type=int, default=20,
                        help="Export raw depth (.npy) + RGB every Nth frame. Default 20; set 1 to "
                            "export every frame (needed for consecutive-frame reprojection strips).")
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    parser.add_argument(
        "--sky_model",
        type=str,
        default="skyseg.onnx",
        help="Path to the sky segmentation ONNX model (default: skyseg.onnx; downloaded automatically if missing)",
    )
    parser.add_argument("--sky_mask_dir", type=str, default=None,
                        help="Directory for cached sky masks (default: <image_folder>_sky_masks/)")
    parser.add_argument("--sky_mask_visualization_dir", type=str, default=None,
                        help="Save sky mask visualizations (original | mask | overlay) to this directory")
    parser.add_argument("--export_preprocessed", type=str, default=None,
                        help="Export stride-sampled, resized/cropped images to this folder")
    parser.add_argument("--export_results", type=str, default=None,
                        help="Export per-frame extrinsic/intrinsic to poses.json and depth maps "
                             "as grayscale PNGs (per-frame min/max normalized, not metric) to this folder")
    parser.add_argument("--override_fov_deg", type=float, default=None,
                        help="Override the model's estimated fx/fy with the value implied by this "
                             "known ground-truth horizontal FOV (e.g. 90 for CS:GO), assuming square "
                             "pixels. Use when the source camera's FOV is known and the model's own "
                             "intrinsics estimate is unreliable (see RESULTS.md).")

    args = parser.parse_args()
    assert args.image_folder or args.video_path, \
        "Provide --image_folder or --video_path"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths, resolved_image_folder = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
        rotate_clockwise_90=args.rotate_clockwise_90,
        start_frame=args.start_frame,
    )

    # Export preprocessed images if requested
    if args.export_preprocessed:
        os.makedirs(args.export_preprocessed, exist_ok=True)
        print(f"Exporting {images.shape[0]} preprocessed images to {args.export_preprocessed}...")
        for i in range(images.shape[0]):
            img = (images[i].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(
                os.path.join(args.export_preprocessed, f"{i:06d}.png"),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
            )
        print(f"Exported to {args.export_preprocessed}")

    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    # Pick inference dtype; autocast still runs for the ops that need fp32 (e.g. LayerNorm).
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32

    # Cast the aggregator (DINOv2-style trunk) to the inference dtype to remove the
    # redundant fp32 master weight copy + autocast bf16 weight cache (~2-3 GB saved,
    # no measurable quality change). gct_base._predict_* upcasts inputs to fp32 and
    # runs each head under `autocast(enabled=False)`, so camera/depth/point heads
    # keep fp32 weights automatically.
    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"GPU mem after load: "
            f"alloc={torch.cuda.memory_allocated()/1e9:.2f} GB, "
            f"reserved={torch.cuda.memory_reserved()/1e9:.2f} GB"
        )

    if args.keyframe_interval is None:
        if args.mode == "streaming" and num_frames > 320:
            args.keyframe_interval = (num_frames + 319) // 320
            print(
                f"Auto-selected --keyframe_interval={args.keyframe_interval} "
                f"(num_frames={num_frames} > 320)."
            )
        else:
            args.keyframe_interval = 1

    if args.keyframe_interval > 1:
        if args.mode == "streaming":
            print(
                f"Keyframe streaming enabled: interval={args.keyframe_interval} "
                f"(after the first {args.num_scale_frames} scale frames)."
            )
        else:  # windowed
            actual_per_window = (
                args.num_scale_frames
                + max(0, args.window_size - args.num_scale_frames) * args.keyframe_interval
            )
            print(
                f"Keyframe windowed enabled: interval={args.keyframe_interval}, "
                f"each window covers up to {actual_per_window} actual frames "
                f"(window_size={args.window_size} keyframes, scale={args.num_scale_frames})."
            )

    # ── Optional: torch.compile + CUDA-graph warmup (streaming only) ────────
    if args.compile:
        if args.mode != "streaming":
            print(
                f"--compile only applies to --mode streaming (got {args.mode!r}); "
                "skipping compile."
            )
        else:
            scale_for_warm = min(args.num_scale_frames, num_frames)
            if scale_for_warm >= num_frames:
                scale_for_warm = max(1, num_frames - 1)
            warm_stream_n = min(10, max(1, num_frames - scale_for_warm))
            warm_h, warm_w = int(images.shape[-2]), int(images.shape[-1])
            print(
                f"Warmup eager (scale={scale_for_warm} + {warm_stream_n} streaming, "
                f"shape={warm_h}x{warm_w}, kf_int={args.keyframe_interval})..."
            )
            t_warm = time.time()
            _warm_streaming(
                model, images, scale_for_warm, warm_stream_n, dtype,
                passes=1, keyframe_interval=args.keyframe_interval,
            )
            print(f"  eager warmup: {time.time() - t_warm:.1f}s")

            print("Compiling hot modules...")
            compile_model(model)

            # 3 passes under compile: 1st captures CUDA graphs, 2nd/3rd replay so
            # the caching allocator / graph-address map converge on the state the
            # real inference will see. See gct_profile.py:302-306 for rationale.
            print("Warmup compiled (3x dress rehearsal)...")
            t_warm = time.time()
            _warm_streaming(
                model, images, scale_for_warm, warm_stream_n, dtype,
                passes=3, keyframe_interval=args.keyframe_interval,
            )
            print(f"  compiled warmup: {time.time() - t_warm:.1f}s")

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    output_device = torch.device("cpu") if args.offload_to_cpu else None

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                overlap_keyframes=args.overlap_keyframes,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=output_device
            )

    print(f"Inference done in {time.time() - t0:.1f}s")
    if torch.cuda.is_available():
        print(
            f"GPU peak during inference: "
            f"{torch.cuda.max_memory_allocated()/1e9:.2f} GB "
            f"(reserved peak {torch.cuda.max_memory_reserved()/1e9:.2f} GB)"
        )

    # ── Post-process ─────────────────────────────────────────────────────────
    if args.offload_to_cpu:
        del images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images_for_post = predictions["images"]  # already CPU
    else:
        images_for_post = images

    predictions, images_cpu = postprocess(predictions, images_for_post)

    if args.override_fov_deg is not None:
        override_intrinsics_with_known_fov(predictions, args.override_fov_deg)

    if args.export_results:
        export_results(predictions, images_cpu, args.export_results,
                        conf_threshold=args.conf_threshold, downsample_factor=args.downsample_factor,
                        comparison_stride=args.comparison_stride)

    # ── Visualize ────────────────────────────────────────────────────────────
    if args.no_viser:
        print("Skipping viser viewer (--no_viser); results already exported.")
        return 0
    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            vis_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=resolved_image_folder,
            skyseg_model_path=args.sky_model,
            sky_mask_dir=args.sky_mask_dir,
            sky_mask_visualization_dir=args.sky_mask_visualization_dir,
        )
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
