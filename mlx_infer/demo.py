"""
MLX inference demo for GCTStreamMLX.

Usage:
    python mlx_infer/demo.py --checkpoint model.pt --images /path/to/images --output out.npz

The script mirrors demo.py but uses the MLX model for GPU-accelerated inference
on Apple Silicon (unified memory, no Metal OOM accumulation).
"""

import argparse
import os
import sys
import glob
import time
from pathlib import Path

import numpy as np
import mlx.core as mx

# Make parent directory importable when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_infer.model import GCTStreamMLX
from mlx_infer.weights import load_checkpoint


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def _load_images_from_dir(image_dir: str, img_size: int = 518,
                          patch_size: int = 14) -> np.ndarray:
    """Load images from a directory using the same crop preprocessing as demo.py.

    Matches PyTorch demo's load_and_preprocess_images(mode="crop"):
    - Resize so width = img_size, maintaining aspect ratio
    - Center-crop height to img_size if taller than img_size
    - Round height to nearest patch_size multiple

    Returns [S, 3, H, W] float32 in [0, 1].
    """
    from PIL import Image

    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )
    if not paths:
        raise ValueError(f"No jpg/png images found in {image_dir}")

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_w = img_size
        new_h = round(h * (new_w / w) / patch_size) * patch_size
        img = img.resize((new_w, new_h), Image.BICUBIC)
        if new_h > img_size:
            start_y = (new_h - img_size) // 2
            arr = np.array(img, dtype=np.float32)[start_y:start_y + img_size]
        else:
            arr = np.array(img, dtype=np.float32)
        frames.append(arr / 255.0)

    arr = np.stack(frames, axis=0)          # [S, H, W, 3]
    arr = arr.transpose(0, 3, 1, 2)         # [S, 3, H, W]
    return arr


# ---------------------------------------------------------------------------
# Post-processing: MLX outputs → pred_dict for PointCloudViewer
# ---------------------------------------------------------------------------

def postprocess(predictions: dict, image_hw: tuple) -> dict:
    """Convert MLX inference outputs to the pred_dict format expected by PointCloudViewer.

    Parameters
    ----------
    predictions : dict
        Raw output from inference_streaming. Tensors have shape [B, S, ...].
        B=1 is squeezed away here.
    image_hw : tuple
        (H, W) of the input images (needed for FoV → intrinsics conversion).

    Returns
    -------
    pred_dict : dict with numpy arrays ready for PointCloudViewer.
    """
    import torch
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general

    H, W = image_hw

    # Convert MLX → numpy, squeeze batch dim (B=1)
    def to_np(x):
        return np.asarray(x.astype(mx.float32))[0]   # [B, S, ...] → [S, ...]

    pose_enc_np = to_np(predictions["pose_enc"])       # [S, 9]
    images_np   = to_np(predictions["images"])         # [S, 3, H, W]

    # pose_enc → w2c extrinsics + intrinsics (via torch)
    pose_t = torch.from_numpy(pose_enc_np).float().unsqueeze(0)   # [1, S, 9]
    extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(pose_t, (H, W))
    # extrinsic_t: [1, S, 3, 4] w2c  →  convert to c2w
    ext4 = torch.zeros(*extrinsic_t.shape[:-2], 4, 4)
    ext4[..., :3, :4] = extrinsic_t
    ext4[..., 3, 3] = 1.0
    ext4_c2w = closed_form_inverse_se3_general(ext4)             # [1, S, 4, 4] c2w
    extrinsic_np = ext4_c2w[0, :, :3, :4].numpy()               # [S, 3, 4] c2w
    intrinsic_np = intrinsic_t[0].numpy()                        # [S, 3, 3]

    pred_dict = {
        "images":    images_np,      # [S, 3, H, W]
        "extrinsic": extrinsic_np,   # [S, 3, 4] c2w
        "intrinsic": intrinsic_np,   # [S, 3, 3]
    }

    if "depth" in predictions:
        pred_dict["depth"]      = to_np(predictions["depth"])       # [S, H, W, 1]
        pred_dict["depth_conf"] = to_np(predictions["depth_conf"])  # [S, H, W]

    if "world_points" in predictions:
        pred_dict["world_points"]      = to_np(predictions["world_points"])       # [S, H, W, 3]
        pred_dict["world_points_conf"] = to_np(predictions["world_points_conf"])  # [S, H, W]

    return pred_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GCTStream MLX inference demo")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--images", required=True,
                        help="Directory of input images (sorted alphabetically)")
    parser.add_argument("--output", default="mlx_output.npz",
                        help="Output file (.npz)")
    parser.add_argument("--img-size", type=int, default=518,
                        help="Resize images to this square size")
    parser.add_argument("--scale-frames", type=int, default=8,
                        help="Number of initial scale frames (Phase 1)")
    parser.add_argument("--keyframe-interval", type=int, default=1,
                        help="Keyframe interval (1 = every frame)")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                        help="Compute dtype (float16 is faster but less stable)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only process first N frames (default: all)")
    # Visualization
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip 3D visualization server (just save .npz)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser visualization server port")
    parser.add_argument("--conf-threshold", type=float, default=1.5,
                        help="Confidence threshold for point cloud filtering")
    parser.add_argument("--downsample-factor", type=int, default=10,
                        help="Point cloud downsample factor")
    parser.add_argument("--point-size", type=float, default=0.00001,
                        help="Initial point size in the 3D viewer")
    args = parser.parse_args()

    # ---- Build model ----
    print("Building GCTStreamMLX model...")
    model = GCTStreamMLX(
        img_size=args.img_size,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_register_tokens=4,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=args.scale_frames,
        camera_num_iterations=4,
        enable_depth=True,
        enable_point=False,
    )

    # ---- Load weights ----
    print(f"Loading checkpoint: {args.checkpoint}")
    load_checkpoint(model, args.checkpoint, verbose=True)

    # ---- Load images ----
    print(f"Loading images from: {args.images}")
    images_np = _load_images_from_dir(args.images, img_size=args.img_size)
    print(f"Loaded {images_np.shape[0]} frames at {images_np.shape[2]}×{images_np.shape[3]}")

    if args.max_frames is not None:
        images_np = images_np[:args.max_frames]
        print(f"Trimmed to {images_np.shape[0]} frames")

    images = mx.array(images_np)   # [S, 3, H, W]

    if args.dtype == "float16":
        images = images.astype(mx.float16)
        model.apply(lambda x: x.astype(mx.float16) if mx.is_array(x) else x)

    # ---- Streaming inference ----
    print("Running streaming inference...")
    t0 = time.perf_counter()

    predictions = model.inference_streaming(
        images,
        num_scale_frames=args.scale_frames,
        keyframe_interval=args.keyframe_interval,
    )

    mx.eval(predictions)
    elapsed = time.perf_counter() - t0
    S = images_np.shape[0]
    print(f"Inference done: {S} frames in {elapsed:.1f}s  ({S/elapsed:.1f} fps)")

    # ---- Save outputs ----
    out = {}
    for k, v in predictions.items():
        if k == "images":
            continue
        out[k] = np.asarray(v)
    np.savez(args.output, **out)
    print(f"Saved predictions to {args.output}")
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    # ---- Visualization ----
    if args.no_vis:
        return

    try:
        from lingbot_map.vis import PointCloudViewer
    except ImportError:
        print("viser not installed. Install with: pip install lingbot-map[vis]")
        return

    print("Post-processing for visualization...")
    H, W = images_np.shape[2], images_np.shape[3]
    pred_dict = postprocess(predictions, (H, W))

    print(f"Launching 3D viewer at http://localhost:{args.port}")
    viewer = PointCloudViewer(
        pred_dict=pred_dict,
        port=args.port,
        vis_threshold=args.conf_threshold,
        downsample_factor=args.downsample_factor,
        point_size=args.point_size,
    )
    viewer.run()


if __name__ == "__main__":
    main()
