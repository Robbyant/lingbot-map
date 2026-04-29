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
import os
import time

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm

from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3_general
from lingbot_map.utils.load_fn import load_and_preprocess_images


# =============================================================================
# Image loading
# =============================================================================

def load_images(image_folder=None, video_path=None, fps=10, image_ext=".jpg,.png",
                first_k=None, stride=1, image_size=518, patch_size=14, num_workers=8):
    """Load images from folder or video and preprocess into a tensor."""
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        pbar = tqdm(total=total_frames, desc="Extracting frames", unit="frame")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
            pbar.update(1)
        pbar.close()
        cap.release()
        paths = saved
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)

    if stride > 1:
        paths = paths[::stride]
    if first_k is not None and first_k > 0:
        paths = paths[:first_k]

    print(f"Loading {len(paths)} images...")
    images = load_and_preprocess_images(
        paths,
        mode="crop",
        image_size=image_size,
        patch_size=patch_size,
    )
    h, w = images.shape[-2:]
    print(f"Preprocessed images to {w}x{h} using canonical crop mode")
    return images, paths


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
        kv_cache_scale_frames=args.kv_cache_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
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
        default=1,
        help="Every N-th frame after scale frames is kept as a keyframe. 1 = every frame.",
    )
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
        default=30,
        help="Max consecutive non-keyframe frames before forcing a keyframe (flow mode only)",
    )
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--kv_cache_scale_frames", type=int, default=8)
    parser.add_argument("--use_sdpa", action="store_true", default=False,
                        help="Use SDPA backend (no flashinfer needed). Default: FlashInfer")

    # Windowed options
    parser.add_argument("--window_size", type=int, default=64, help="Frames per window (windowed mode)")
    parser.add_argument("--overlap_size", type=int, default=None,
                        help="Overlap between windows. Default: num_scale_frames (overlap = scale frames)")

    # Visualization
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--conf_threshold", type=float, default=1.0)
    parser.add_argument("--downsample_factor", type=int, default=10)
    parser.add_argument("--point_size", type=float, default=0.005)
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter sky points")
    parser.add_argument("--skyseg_model_path", type=str, default="skyseg.onnx",
                        help="Path to sky segmentation ONNX model")

    # Output
    parser.add_argument("--save_predictions", type=str, default=None, help="Save predictions to .npz file")
    parser.add_argument("--load_predictions", type=str, default=None, help="Load predictions from .npz file (skip inference)")

    args = parser.parse_args()
    assert args.image_folder or args.video_path or args.load_predictions, \
        "Provide --image_folder, --video_path, or --load_predictions"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load from saved predictions ──────────────────────────────────────────
    if args.load_predictions:
        print(f"Loading predictions from {args.load_predictions}...")
        data = np.load(args.load_predictions, allow_pickle=True)
        predictions = {k: torch.from_numpy(data[k]) for k in data.files}
        print(f"  Keys: {list(predictions.keys())}")

        # Load images for visualization (needed for point cloud coloring)
        if args.image_folder or args.video_path:
            images, _ = load_images(
                image_folder=args.image_folder, video_path=args.video_path,
                fps=args.fps, first_k=args.first_k, stride=args.stride,
                image_size=args.image_size, patch_size=args.patch_size,
            )
            images_cpu = images
        else:
            # Try to reconstruct from predictions if images were saved
            if "images" in predictions:
                images_cpu = predictions["images"]
            else:
                print("Warning: no images provided for visualization. Use --image_folder or --video_path.")
                images_cpu = None

        # Jump to visualization
        if images_cpu is not None:
            try:
                from lingbot_map.vis import PointCloudViewer
                viewer = PointCloudViewer(
                    pred_dict=prepare_for_visualization(predictions, images_cpu),
                    port=args.port,
                    init_conf_threshold=args.conf_threshold,
                    downsample_factor=args.downsample_factor,
                    point_size=args.point_size,
                    mask_sky=args.mask_sky,
                    image_folder=args.image_folder,
                )
                print(f"3D viewer at http://localhost:{args.port}")
                viewer.run()
            except ImportError:
                print("viser not installed. Install with: pip install lingbot-map[vis]")
        return

    # ── Load images & model ──────────────────────────────────────────────────
    t0 = time.time()
    images, paths = load_images(
        image_folder=args.image_folder, video_path=args.video_path,
        fps=args.fps, first_k=args.first_k, stride=args.stride,
        image_size=args.image_size, patch_size=args.patch_size,
    )
    model = load_model(args, device)
    print(f"Total load time: {time.time() - t0:.1f}s")

    images = images.to(device)
    num_frames = images.shape[0]
    print(f"Input: {num_frames} frames, shape {tuple(images.shape)}")
    print(f"Mode: {args.mode}")

    if args.flow_threshold > 0:
        print(
            f"Flow-based keyframe mode: threshold={args.flow_threshold:.1f}px, "
            f"max_gap={args.max_non_keyframe_gap} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )
    elif args.keyframe_interval > 1:
        print(
            f"Fixed-interval keyframe mode: interval={args.keyframe_interval} "
            f"(after the first {args.num_scale_frames} scale frames)."
        )

    # ── Inference ────────────────────────────────────────────────────────────
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Running {args.mode} inference (dtype={dtype})...")
    t0 = time.time()

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
            )
        else:  # windowed
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                flow_threshold=args.flow_threshold,
                max_non_keyframe_gap=args.max_non_keyframe_gap,
            )

    t_infer = time.time() - t0
    print(f"Inference done: {t_infer:.1f}s ({num_frames / t_infer:.1f} FPS)")

    # ── Post-process ─────────────────────────────────────────────────────────
    predictions, images_cpu = postprocess(predictions, images)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save_predictions:
        save_dict = {"images": images_cpu.numpy()}
        for k, v in predictions.items():
            if isinstance(v, torch.Tensor):
                save_dict[k] = v.numpy()
        np.savez_compressed(args.save_predictions, **save_dict)
        print(f"Predictions saved to {args.save_predictions} ({len(save_dict)} keys)")

    # ── Visualize ────────────────────────────────────────────────────────────
    try:
        from lingbot_map.vis import PointCloudViewer
        viewer = PointCloudViewer(
            pred_dict=prepare_for_visualization(predictions, images_cpu),
            port=args.port,
            init_conf_threshold=args.conf_threshold,
            downsample_factor=args.downsample_factor,
            point_size=args.point_size,
            mask_sky=args.mask_sky,
            image_folder=args.image_folder,
        )
        print(f"3D viewer at http://localhost:{args.port}")
        viewer.run()
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        print(f"Predictions contain keys: {list(predictions.keys())}")


if __name__ == "__main__":
    main()
