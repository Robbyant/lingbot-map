import argparse
import asyncio
import base64
import glob
import io
import json
import math
import os
import tempfile
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import cv2

import torch
import viser
import viser.transforms as tf
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from live_point_cloud_viewer import LivePointCloudViewer
from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.models.gct_stream_window import GCTStream as GCTStreamWindow
from lingbot_map.utils.geometry import closed_form_inverse_se3_general, unproject_depth_map_to_point_map
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.vis import PointCloudViewer

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


def load_images(
    image_folder: str | None = None,
    video_path: str | None = None,
    fps: int = 10,
    image_ext: str = ".jpg,.png",
    first_k: int | None = None,
    stride: int = 1,
    image_size: int = 518,
    patch_size: int = 14,
):
    if video_path is not None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = max(1, round(src_fps / fps))
        idx, saved = 0, []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                path = os.path.join(out_dir, f"{len(saved):06d}.jpg")
                cv2.imwrite(path, frame)
                saved.append(path)
            idx += 1
        cap.release()
        paths = saved
        resolved_folder = out_dir
        print(f"Extracted {len(paths)} frames from video ({total_frames} total, interval={interval})")
    else:
        exts = image_ext.split(",")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(image_folder, f"*{ext}")))
        paths = sorted(paths)
        resolved_folder = image_folder

    if first_k is not None and first_k > 0:
        paths = paths[:first_k]
    if stride > 1:
        paths = paths[::stride]

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


def load_model(args: argparse.Namespace, device: torch.device):
    model_cls = GCTStreamWindow if getattr(args, "mode", "streaming") == "windowed" else GCTStream
    print("Building model...")
    model = model_cls(
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


def _squeeze_single_batch(key: str, value):
    batched_ndim = _BATCHED_NDIMS.get(key)
    if batched_ndim is None or not hasattr(value, "ndim"):
        return value
    if value.ndim == batched_ndim and value.shape[0] == 1:
        return value[0]
    return value


def postprocess(predictions: dict[str, torch.Tensor], images: torch.Tensor):
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    predictions["extrinsic"] = extrinsic_4x4[..., :3, :4]
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)
    predictions.pop("images", None)
    print("Moving results to CPU...")
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = _squeeze_single_batch(key, predictions[key].to("cpu", non_blocking=True))
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return predictions, images_cpu


def prepare_for_visualization(predictions: dict[str, Any], images=None):
    vis_predictions = {}
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor):
            value = _squeeze_single_batch(key, value.detach().cpu())
            vis_predictions[key] = value.numpy()
        elif isinstance(value, np.ndarray):
            vis_predictions[key] = _squeeze_single_batch(key, value)
        else:
            vis_predictions[key] = value

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


def preprocess_frame_rgb(
    rgb: np.ndarray,
    image_size: int,
    patch_size: int,
) -> torch.Tensor:
    image = Image.fromarray(rgb, mode="RGB")
    width, height = image.size
    new_width = image_size
    new_height = round(height * (new_width / width) / patch_size) * patch_size
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = torch.from_numpy(np.asarray(image).copy()).float() / 255.0
    tensor = tensor.permute(2, 0, 1)
    if new_height > image_size:
        start_y = (new_height - image_size) // 2
        tensor = tensor[:, start_y : start_y + image_size, :]
    return tensor.contiguous()


def tensor_frame_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    frame = tensor.detach().cpu()
    if frame.ndim == 5:
        frame = frame[0, 0]
    if frame.ndim == 4:
        frame = frame[0]
    return frame.permute(1, 2, 0).numpy()


def postprocess_predictions(
    predictions: dict[str, torch.Tensor],
    images: torch.Tensor,
    free_cuda_cache: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsic_4x4 = torch.zeros((*extrinsic.shape[:-2], 4, 4), device=extrinsic.device, dtype=extrinsic.dtype)
    extrinsic_4x4[..., :3, :4] = extrinsic
    extrinsic_4x4[..., 3, 3] = 1.0
    extrinsic_4x4 = closed_form_inverse_se3_general(extrinsic_4x4)
    predictions["extrinsic"] = extrinsic_4x4[..., :3, :4]
    predictions["intrinsic"] = intrinsic
    predictions.pop("pose_enc_list", None)

    processed: dict[str, torch.Tensor] = {}
    for key, value in predictions.items():
        if torch.is_tensor(value):
            processed[key] = value.to("cpu", non_blocking=True)
        else:
            processed[key] = value
    images_cpu = images.to("cpu", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if free_cuda_cache:
            torch.cuda.empty_cache()
    return processed, images_cpu


def slice_frame_prediction(predictions: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in predictions.items():
        if not torch.is_tensor(value):
            sliced[key] = value
            continue
        if value.ndim >= 2 and value.shape[1] > index:
            sliced[key] = value[:, index : index + 1]
        else:
            sliced[key] = value
    return sliced


def pose_matrix_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :4] = extrinsic
    return matrix


def quaternion_xyzw_from_matrix(rotation: np.ndarray) -> np.ndarray:
    return np.array(tf.SO3.from_matrix(rotation).wxyz, dtype=np.float32)


class IncrementalViserViewer:
    def __init__(self, port: int, point_size: float, conf_threshold: float, downsample_factor: int):
        self.port = port
        self.point_size = point_size
        self.conf_threshold = conf_threshold
        self.downsample_factor = downsample_factor
        self.server = viser.ViserServer(host="0.0.0.0", port=port)
        self.server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
        self.reset()

    def reset(self) -> None:
        for handle in getattr(self, "camera_handles", []):
            try:
                handle.remove()
            except Exception:
                pass
        if getattr(self, "point_cloud_handle", None) is not None:
            try:
                self.point_cloud_handle.remove()
            except Exception:
                pass
        if getattr(self, "trajectory_handle", None) is not None:
            try:
                self.trajectory_handle.remove()
            except Exception:
                pass
        self.camera_handles: list[Any] = []
        self.point_cloud_handle = None
        self.trajectory_handle = None
        self.all_points = np.zeros((0, 3), dtype=np.float32)
        self.all_colors = np.zeros((0, 3), dtype=np.uint8)
        self.positions: list[np.ndarray] = []
        self.frame_index = 0

    def _extract_sparse_points(
        self,
        frame_pred: dict[str, torch.Tensor],
    ) -> tuple[np.ndarray, np.ndarray]:
        image = tensor_frame_to_numpy(frame_pred["images"])
        colors = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        if "world_points" in frame_pred:
            world_points = frame_pred["world_points"][0, 0].detach().cpu().numpy()
        else:
            world_points = unproject_depth_map_to_point_map(
                frame_pred["depth"].detach().cpu().numpy(),
                frame_pred["extrinsic"].detach().cpu().numpy(),
                frame_pred["intrinsic"].detach().cpu().numpy(),
            )[0]

        if "world_points_conf" in frame_pred:
            conf = frame_pred["world_points_conf"][0, 0].detach().cpu().numpy()
        else:
            conf = frame_pred["depth_conf"][0, 0].detach().cpu().numpy()

        mask = conf.reshape(-1) > self.conf_threshold
        points = world_points.reshape(-1, 3)[mask]
        point_colors = colors.reshape(-1, 3)[mask]
        if self.downsample_factor > 1 and len(points) > 0:
            indices = np.arange(0, len(points), self.downsample_factor)
            points = points[indices]
            point_colors = point_colors[indices]
        return points.astype(np.float32), point_colors.astype(np.uint8)

    def _refresh_point_cloud(self) -> None:
        if self.point_cloud_handle is not None:
            self.point_cloud_handle.remove()
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            "/stream/map",
            points=self.all_points,
            colors=self.all_colors if len(self.all_colors) > 0 else np.zeros((0, 3), dtype=np.uint8),
            point_size=self.point_size,
            point_shape="circle",
            precision="float32",
        )

    def _refresh_trajectory(self) -> None:
        if self.trajectory_handle is not None:
            self.trajectory_handle.remove()
        if len(self.positions) < 2:
            return
        segments = np.stack(
            [np.stack([self.positions[i], self.positions[i + 1]], axis=0) for i in range(len(self.positions) - 1)],
            axis=0,
        ).astype(np.float32)
        self.trajectory_handle = self.server.scene.add_line_segments(
            "/stream/trajectory",
            points=segments,
            colors=np.array([255, 120, 0], dtype=np.uint8),
            line_width=2.0,
        )

    def add_frame(self, frame_pred: dict[str, torch.Tensor]) -> dict[str, Any]:
        extrinsic = frame_pred["extrinsic"][0, 0].detach().cpu().numpy()
        intrinsic = frame_pred["intrinsic"][0, 0].detach().cpu().numpy()
        image = (np.clip(tensor_frame_to_numpy(frame_pred["images"]), 0.0, 1.0) * 255).astype(np.uint8)
        rotation = extrinsic[:, :3]
        position = extrinsic[:, 3]
        wxyz = quaternion_xyzw_from_matrix(rotation)

        points, point_colors = self._extract_sparse_points(frame_pred)
        points_added = int(len(points))
        if points_added > 0:
            self.all_points = np.concatenate([self.all_points, points], axis=0)
            self.all_colors = np.concatenate([self.all_colors, point_colors], axis=0)
            self._refresh_point_cloud()

        self.positions.append(position.astype(np.float32))
        self._refresh_trajectory()

        frame_handle = self.server.scene.add_frame(
            f"/stream/frame_{self.frame_index}",
            wxyz=wxyz,
            position=position,
            axes_length=0.04,
            axes_radius=0.002,
            origin_radius=0.002,
        )
        focal = float(intrinsic[0, 0])
        fov = 2 * np.arctan2(float(intrinsic[0, 2]), focal)
        aspect = float(intrinsic[0, 2] / max(intrinsic[1, 2], 1e-6))
        frustum_handle = self.server.scene.add_camera_frustum(
            f"/stream/frame_{self.frame_index}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.03,
            wxyz=wxyz,
            position=position,
            image=image,
            color=(50, 180, 255),
        )
        self.camera_handles.extend([frame_handle, frustum_handle])
        self.frame_index += 1

        return {
            "points_added": points_added,
            "total_points": int(len(self.all_points)),
            "pose": {
                "position": position.astype(float).tolist(),
                "quaternion_wxyz": wxyz.astype(float).tolist(),
            },
        }

    def rebuild_sequence(self, sequence_pred: dict[str, torch.Tensor], images_cpu: torch.Tensor) -> dict[str, Any]:
        self.reset()
        sequence_pred = dict(sequence_pred)
        sequence_pred["images"] = images_cpu
        last_stats: dict[str, Any] | None = None
        total_frames = int(sequence_pred["extrinsic"].shape[1])
        for idx in range(total_frames):
            frame_pred = slice_frame_prediction(sequence_pred, idx)
            frame_pred["images"] = images_cpu[:, idx : idx + 1]
            last_stats = self.add_frame(frame_pred)
        if last_stats is None:
            raise RuntimeError("No frames available for viewer rebuild")
        return last_stats


class OnlineGCTRunner:
    def __init__(
        self,
        args: argparse.Namespace,
        model_path: Path,
        image_size: int,
        patch_size: int,
        max_frame_num: int,
        kv_cache_sliding_window: int,
        num_scale_frames: int,
        camera_num_iterations: int,
        use_sdpa: bool,
    ):
        self.args = args
        self.model_path = model_path
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_frame_num = max_frame_num
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.default_num_scale_frames = num_scale_frames
        self.camera_num_iterations = camera_num_iterations
        self.use_sdpa = use_sdpa
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: GCTStream | None = None
        self.dtype = torch.float32
        self.viewer: PointCloudViewer | None = None
        self.viewer_thread: threading.Thread | None = None
        self.viewer_lock = threading.Lock()
        self.browser_stream_state: dict[str, Any] | None = None

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return
        model = load_model(self.args, self.device)
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        if self.dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
            model.aggregator = model.aggregator.to(dtype=self.dtype)
        self.model = model

    def reset(self) -> None:
        if self.model is not None:
            self.model.clean_kv_cache()
        self.browser_stream_state = None
        self._stop_viewer()

    def _stop_viewer(self) -> None:
        with self.viewer_lock:
            if self.viewer is not None:
                try:
                    self.viewer.server.stop()
                except Exception as exc:
                    print(f"[e2e_io_streaming] viewer stop warning: {exc}")
                self.viewer = None
            self.viewer_thread = None

    def _launch_viewer(
        self,
        pred_dict: dict[str, Any],
        viewer_port: int,
        image_folder: str | None = None,
    ) -> None:
        self._stop_viewer()
        ready = threading.Event()
        error_holder: list[BaseException] = []

        def viewer_main() -> None:
            try:
                viewer_threshold = max(1.0, float(self.args.conf_threshold))
                viewer = PointCloudViewer(
                    pred_dict=pred_dict,
                    port=viewer_port,
                    vis_threshold=viewer_threshold,
                    downsample_factor=self.args.downsample_factor,
                    point_size=self.args.point_size,
                    image_folder=image_folder,
                )
                with self.viewer_lock:
                    self.viewer = viewer
                ready.set()
                viewer.run()
            except BaseException as exc:
                error_holder.append(exc)
                ready.set()
                raise

        thread = threading.Thread(target=viewer_main, name="lingbot-map-viewer", daemon=True)
        with self.viewer_lock:
            self.viewer_thread = thread
        thread.start()
        ready.wait(timeout=10.0)
        if error_holder:
            raise RuntimeError(f"Viewer failed to start: {error_holder[0]}") from error_holder[0]
        if not ready.is_set():
            raise RuntimeError("Viewer startup timed out")
        time.sleep(0.5)

    def _start_incremental_viewer(self, viewer_port: int) -> LivePointCloudViewer:
        self._stop_viewer()
        viewer = LivePointCloudViewer(
            port=viewer_port,
            point_size=self.args.point_size,
            vis_threshold=self.args.conf_threshold,
            downsample_factor=self.args.downsample_factor,
        )
        with self.viewer_lock:
            self.viewer = viewer
            self.viewer_thread = None
        time.sleep(0.2)
        return viewer

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=self.dtype)
        return nullcontext()

    def start_browser_stream(self, scale_frames: int, viewer_port: int, total_frames: int = 0) -> None:
        self._ensure_model_loaded()
        assert self.model is not None
        self.model.clean_kv_cache()
        self._stop_viewer()
        self.browser_stream_state = {
            "scale_frames": max(1, int(scale_frames)),
            "viewer_port": int(viewer_port),
            "total_frames": max(0, int(total_frames)),
            "sampled_frames": 0,
            "processed_frames": 0,
            "scale_buffer": [],
            "last_stream_stats": None,
            "viewer": None,
            "started_at": time.perf_counter(),
            "inference_time_total": 0.0,
            "inference_steps": 0,
        }

    def stream_browser_frame(self, rgb_frame: np.ndarray) -> dict[str, Any]:
        if self.browser_stream_state is None:
            raise RuntimeError("Browser stream has not been started")
        assert self.model is not None

        state = self.browser_stream_state
        frame_tensor = preprocess_frame_rgb(
            rgb_frame,
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        state["sampled_frames"] += 1

        with torch.no_grad(), self._autocast_context():
            if len(state["scale_buffer"]) < state["scale_frames"]:
                state["scale_buffer"].append(frame_tensor)
                if len(state["scale_buffer"]) < state["scale_frames"]:
                    return {
                        "type": "buffering",
                        "sampled_frames": state["sampled_frames"],
                        "processed_frames": state["processed_frames"],
                        "total_frames": state["total_frames"],
                    }

                scale_images = torch.stack(state["scale_buffer"], dim=0).to(self.device)
                t_infer = time.perf_counter()
                scale_output = self.model.forward(
                    scale_images,
                    num_frame_for_scale=len(state["scale_buffer"]),
                    num_frame_per_block=len(state["scale_buffer"]),
                    causal_inference=True,
                )
                infer_dt = time.perf_counter() - t_infer
                state["inference_time_total"] += infer_dt
                state["inference_steps"] += len(state["scale_buffer"])
                scale_processed, scale_images_cpu = postprocess_predictions(
                    dict(scale_output),
                    scale_images.unsqueeze(0),
                    free_cuda_cache=False,
                )
                viewer = self._start_incremental_viewer(viewer_port=state["viewer_port"])
                state["viewer"] = viewer
                for local_idx in range(int(scale_processed["extrinsic"].shape[1])):
                    frame_pred = slice_frame_prediction(scale_processed, local_idx)
                    frame_images = scale_images_cpu[:, local_idx : local_idx + 1]
                    vis_frame = prepare_for_visualization(frame_pred, frame_images)
                    state["last_stream_stats"] = viewer.append_prediction(vis_frame)
                state["processed_frames"] += len(state["scale_buffer"])
                del scale_output
            else:
                frame_input = frame_tensor.unsqueeze(0).to(self.device)
                t_infer = time.perf_counter()
                frame_output = self.model.forward(
                    frame_input,
                    num_frame_for_scale=state["scale_frames"],
                    num_frame_per_block=1,
                    causal_inference=True,
                )
                infer_dt = time.perf_counter() - t_infer
                state["inference_time_total"] += infer_dt
                state["inference_steps"] += 1
                frame_processed, frame_images_cpu = postprocess_predictions(
                    dict(frame_output),
                    frame_input.unsqueeze(0),
                    free_cuda_cache=False,
                )
                viewer = state["viewer"]
                if viewer is None:
                    viewer = self._start_incremental_viewer(viewer_port=state["viewer_port"])
                    state["viewer"] = viewer
                vis_frame = prepare_for_visualization(frame_processed, frame_images_cpu)
                state["last_stream_stats"] = viewer.append_prediction(vis_frame)
                state["processed_frames"] += 1
                del frame_output

        total_frames = state["total_frames"]
        processed_frames = state["processed_frames"]
        percent = 0 if total_frames <= 0 else min(99, int(100 * processed_frames / max(total_frames, 1)))
        elapsed = max(time.perf_counter() - state["started_at"], 1e-6)
        avg_inference_ms = (state["inference_time_total"] / max(state["inference_steps"], 1)) * 1000.0
        model_fps = processed_frames / max(state["inference_time_total"], 1e-6) if state["inference_time_total"] > 0 else 0.0
        stream_fps = state["sampled_frames"] / elapsed
        return {
            "type": "frame_result",
            "sampled_frames": state["sampled_frames"],
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "percent": percent,
            "avg_inference_ms": avg_inference_ms,
            "last_inference_ms": infer_dt * 1000.0 if 'infer_dt' in locals() else 0.0,
            "model_fps": model_fps,
            "stream_fps": stream_fps,
            "pose": None if state["last_stream_stats"] is None else state["last_stream_stats"]["pose"],
            "total_points": 0 if state["last_stream_stats"] is None else state["last_stream_stats"]["total_points"],
        }

    def finish_browser_stream(self) -> dict[str, Any]:
        if self.browser_stream_state is None:
            raise RuntimeError("Browser stream has not been started")
        state = self.browser_stream_state
        if self.model is not None:
            self.model.clean_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        final_stats = state["last_stream_stats"] or {
            "points_added": 0,
            "total_points": 0,
            "pose": None,
        }
        elapsed = max(time.perf_counter() - state["started_at"], 1e-6)
        result = {
            "type": "done",
            "mode": "browser_frame_streaming",
            "sampled_frames": state["sampled_frames"],
            "processed_frames": state["processed_frames"],
            "total_frames": state["total_frames"],
            "avg_inference_ms": (state["inference_time_total"] / max(state["inference_steps"], 1)) * 1000.0,
            "last_inference_ms": 0.0,
            "model_fps": state["processed_frames"] / max(state["inference_time_total"], 1e-6) if state["inference_time_total"] > 0 else 0.0,
            "stream_fps": state["sampled_frames"] / elapsed,
            "points_added": final_stats["points_added"],
            "total_points": final_stats["total_points"],
            "pose": final_stats["pose"],
        }
        self.browser_stream_state = None
        return result

    def _run_official_streaming(self, images: torch.Tensor, scale_frames: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        assert self.model is not None
        images = images.to(self.device)
        with torch.no_grad(), self._autocast_context():
            output = self.model.inference_streaming(
                images,
                num_scale_frames=scale_frames,
                keyframe_interval=1,
                output_device=torch.device("cpu"),
            )
        predictions, images_cpu = postprocess(output, output["images"])
        return predictions, images_cpu

    def _stream_root_video(
        self,
        video_path: str,
        fps: float,
        scale_frames: int,
        viewer_port: int,
        progress_cb=None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, Any]]:
        assert self.model is not None
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(1, round(src_fps / max(fps, 1.0)))
        total_sampled = max(1, math.ceil(total_src_frames / interval)) if total_src_frames > 0 else 0

        started_at = time.perf_counter()
        inference_time_total = 0.0
        inference_steps = 0

        def report(
            phase: str,
            message: str,
            percent: int,
            processed_frames: int,
            sampled_frames: int,
            last_inference_ms: float = 0.0,
        ) -> None:
            if progress_cb is not None:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                progress_cb(
                    phase,
                    message,
                    percent,
                    processed_frames=processed_frames,
                    sampled_frames=sampled_frames,
                    total_frames=total_sampled,
                    avg_inference_ms=(inference_time_total / max(inference_steps, 1)) * 1000.0,
                    last_inference_ms=last_inference_ms,
                    model_fps=processed_frames / max(inference_time_total, 1e-6) if inference_time_total > 0 else 0.0,
                    stream_fps=sampled_frames / elapsed,
                )

        all_images_cpu: list[torch.Tensor] = []
        all_pose_enc: list[torch.Tensor] = []
        all_depth: list[torch.Tensor] = []
        all_depth_conf: list[torch.Tensor] = []
        all_world_points: list[torch.Tensor] = []
        all_world_points_conf: list[torch.Tensor] = []
        scale_buffer: list[torch.Tensor] = []
        stream_viewer: IncrementalViserViewer | None = None
        last_stream_stats: dict[str, Any] | None = None

        def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to("cpu", non_blocking=True)

        sampled_frames = 0
        processed_frames = 0
        frame_idx = 0
        report("streaming", "Starting frame-by-frame inference", 5, processed_frames, sampled_frames)

        with torch.no_grad(), self._autocast_context():
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_idx % interval != 0:
                    frame_idx += 1
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_tensor = preprocess_frame_rgb(
                    frame_rgb,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                )
                all_images_cpu.append(frame_tensor.cpu())
                sampled_frames += 1

                if len(scale_buffer) < scale_frames:
                    scale_buffer.append(frame_tensor)
                    if len(scale_buffer) == scale_frames:
                        report(
                            "scale_frames",
                            f"Processing initial {len(scale_buffer)} scale frames",
                            10 if total_sampled == 0 else 10 + int(50 * (sampled_frames / total_sampled)),
                            processed_frames,
                            sampled_frames,
                            last_inference_ms=(infer_dt * 1000.0) / max(len(scale_buffer), 1),
                        )
                        scale_images = torch.stack(scale_buffer, dim=0).to(self.device)
                        t_infer = time.perf_counter()
                        scale_output = self.model.forward(
                            scale_images,
                            num_frame_for_scale=len(scale_buffer),
                            num_frame_per_block=len(scale_buffer),
                            causal_inference=True,
                        )
                        infer_dt = time.perf_counter() - t_infer
                        inference_time_total += infer_dt
                        inference_steps += len(scale_buffer)
                        all_pose_enc.append(to_cpu(scale_output["pose_enc"]))
                        if "depth" in scale_output:
                            all_depth.append(to_cpu(scale_output["depth"]))
                        if "depth_conf" in scale_output:
                            all_depth_conf.append(to_cpu(scale_output["depth_conf"]))
                        if "world_points" in scale_output:
                            all_world_points.append(to_cpu(scale_output["world_points"]))
                        if "world_points_conf" in scale_output:
                            all_world_points_conf.append(to_cpu(scale_output["world_points_conf"]))
                        scale_processed, scale_images_cpu = postprocess_predictions(
                            dict(scale_output),
                            scale_images.unsqueeze(0),
                            free_cuda_cache=False,
                        )
                        stream_viewer = self._start_incremental_viewer(viewer_port=viewer_port)
                        for local_idx in range(int(scale_processed["extrinsic"].shape[1])):
                            frame_pred = slice_frame_prediction(scale_processed, local_idx)
                            frame_images = scale_images_cpu[:, local_idx : local_idx + 1]
                            vis_frame = prepare_for_visualization(frame_pred, frame_images)
                            last_stream_stats = stream_viewer.append_prediction(vis_frame)
                        processed_frames += len(scale_buffer)
                        del scale_output
                else:
                    frame_input = frame_tensor.unsqueeze(0).to(self.device)
                    t_infer = time.perf_counter()
                    frame_output = self.model.forward(
                        frame_input,
                        num_frame_for_scale=scale_frames,
                        num_frame_per_block=1,
                        causal_inference=True,
                    )
                    infer_dt = time.perf_counter() - t_infer
                    inference_time_total += infer_dt
                    inference_steps += 1
                    all_pose_enc.append(to_cpu(frame_output["pose_enc"]))
                    if "depth" in frame_output:
                        all_depth.append(to_cpu(frame_output["depth"]))
                    if "depth_conf" in frame_output:
                        all_depth_conf.append(to_cpu(frame_output["depth_conf"]))
                    if "world_points" in frame_output:
                        all_world_points.append(to_cpu(frame_output["world_points"]))
                    if "world_points_conf" in frame_output:
                        all_world_points_conf.append(to_cpu(frame_output["world_points_conf"]))
                    frame_processed, frame_images_cpu = postprocess_predictions(
                        dict(frame_output),
                        frame_input.unsqueeze(0),
                        free_cuda_cache=False,
                    )
                    if stream_viewer is None:
                        stream_viewer = self._start_incremental_viewer(viewer_port=viewer_port)
                    vis_frame = prepare_for_visualization(frame_processed, frame_images_cpu)
                    last_stream_stats = stream_viewer.append_prediction(vis_frame)
                    processed_frames += 1
                    del frame_output

                progress = 10 if total_sampled == 0 else 10 + int(80 * (processed_frames / total_sampled))
                report(
                    "streaming",
                    f"Processed {processed_frames} / {total_sampled or '?'} streamed frames",
                    progress,
                    processed_frames,
                    sampled_frames,
                    last_inference_ms=infer_dt * 1000.0 if 'infer_dt' in locals() else 0.0,
                )
                frame_idx += 1

        cap.release()

        if not all_images_cpu:
            raise RuntimeError("No frames were sampled from the video")

        if processed_frames == 0 and scale_buffer:
            report("scale_frames", f"Processing short sequence of {len(scale_buffer)} scale frames", 60, processed_frames, sampled_frames)
            scale_images = torch.stack(scale_buffer, dim=0).to(self.device)
            t_infer = time.perf_counter()
            scale_output = self.model.forward(
                scale_images,
                num_frame_for_scale=len(scale_buffer),
                num_frame_per_block=len(scale_buffer),
                causal_inference=True,
            )
            infer_dt = time.perf_counter() - t_infer
            inference_time_total += infer_dt
            inference_steps += len(scale_buffer)
            all_pose_enc.append(to_cpu(scale_output["pose_enc"]))
            if "depth" in scale_output:
                all_depth.append(to_cpu(scale_output["depth"]))
            if "depth_conf" in scale_output:
                all_depth_conf.append(to_cpu(scale_output["depth_conf"]))
            if "world_points" in scale_output:
                all_world_points.append(to_cpu(scale_output["world_points"]))
            if "world_points_conf" in scale_output:
                all_world_points_conf.append(to_cpu(scale_output["world_points_conf"]))
            scale_processed, scale_images_cpu = postprocess_predictions(
                dict(scale_output),
                scale_images.unsqueeze(0),
                free_cuda_cache=False,
            )
            stream_viewer = self._start_incremental_viewer(viewer_port=viewer_port)
            for local_idx in range(int(scale_processed["extrinsic"].shape[1])):
                frame_pred = slice_frame_prediction(scale_processed, local_idx)
                frame_images = scale_images_cpu[:, local_idx : local_idx + 1]
                vis_frame = prepare_for_visualization(frame_pred, frame_images)
                last_stream_stats = stream_viewer.append_prediction(vis_frame)
            processed_frames = len(scale_buffer)
            del scale_output

        self.model.clean_kv_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        predictions: dict[str, torch.Tensor] = {
            "pose_enc": torch.cat(all_pose_enc, dim=1),
        }
        if all_depth:
            predictions["depth"] = torch.cat(all_depth, dim=1)
        if all_depth_conf:
            predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)
        if all_world_points:
            predictions["world_points"] = torch.cat(all_world_points, dim=1)
        if all_world_points_conf:
            predictions["world_points_conf"] = torch.cat(all_world_points_conf, dim=1)

        images_cpu = torch.stack(all_images_cpu, dim=0)
        report("postprocessing", "Converting streamed outputs for visualization", 92, processed_frames, sampled_frames)
        final_predictions, final_images_cpu = postprocess(predictions, images_cpu)
        if last_stream_stats is None:
            raise RuntimeError("Streaming viewer did not receive any frames")
        return final_predictions, final_images_cpu, last_stream_stats

    def run_uploaded_video(self, frame_paths: list[str], scale_frames: int, viewer_port: int) -> dict[str, Any]:
        if not frame_paths:
            raise RuntimeError("No frames extracted from uploaded video")
        self._ensure_model_loaded()
        assert self.model is not None
        self.model.clean_kv_cache()
        images = load_and_preprocess_images(
            frame_paths,
            mode="crop",
            image_size=self.image_size,
            patch_size=self.patch_size,
        )
        predictions, images_cpu = self._run_official_streaming(images, scale_frames=scale_frames)
        vis_predictions = prepare_for_visualization(predictions, images_cpu)
        self._launch_viewer(vis_predictions, viewer_port=viewer_port)
        total_frames = int(predictions["extrinsic"].shape[0] if predictions["extrinsic"].ndim == 3 else predictions["extrinsic"].shape[1])
        final_extrinsic = vis_predictions["extrinsic"][-1]
        final_position = final_extrinsic[:, 3].astype(float).tolist()
        total_points = int(np.count_nonzero(vis_predictions.get("depth_conf", np.array([])) > self.args.conf_threshold))
        return {
            "type": "upload_complete",
            "num_frames": total_frames,
            "points_added": total_points,
            "total_points": total_points,
            "pose": {
                "position": final_position,
                "quaternion_wxyz": quaternion_xyzw_from_matrix(final_extrinsic[:, :3]).astype(float).tolist(),
            },
        }

    def run_root_video(self, video_path: str, fps: float, scale_frames: int, viewer_port: int, progress_cb=None) -> dict[str, Any]:
        self._ensure_model_loaded()
        assert self.model is not None
        self.model.clean_kv_cache()
        predictions, images_cpu, stream_stats = self._stream_root_video(
            video_path=video_path,
            fps=fps,
            scale_frames=scale_frames,
            viewer_port=viewer_port,
            progress_cb=progress_cb,
        )
        if progress_cb is not None:
            total_frames = int(predictions["extrinsic"].shape[0] if predictions["extrinsic"].ndim == 3 else predictions["extrinsic"].shape[1])
            progress_cb("rendering", "Viewer updated live during streaming", 97, processed_frames=total_frames, sampled_frames=total_frames, total_frames=total_frames)
        total_frames = int(predictions["extrinsic"].shape[0] if predictions["extrinsic"].ndim == 3 else predictions["extrinsic"].shape[1])
        return {
            "type": "run_complete",
            "num_frames": total_frames,
            "mode": "frame_by_frame_streaming",
            "points_added": stream_stats["points_added"],
            "total_points": stream_stats["total_points"],
            "pose": stream_stats["pose"],
        }


@dataclass
class SessionConfig:
    session_id: str
    scale_frames: int
    fps: float
    total_frames: int
    source_name: str


class DemoServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.video_dir = Path(__file__).resolve().parent / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.runner = OnlineGCTRunner(
            args=args,
            model_path=Path(args.model_path),
            image_size=args.image_size,
            patch_size=args.patch_size,
            max_frame_num=args.max_frame_num,
            kv_cache_sliding_window=args.kv_cache_sliding_window,
            num_scale_frames=args.num_scale_frames,
            camera_num_iterations=args.camera_num_iterations,
            use_sdpa=args.use_sdpa,
        )
        self.active_session: SessionConfig | None = None
        self.lock = asyncio.Lock()
        self.progress: dict[str, Any] = {
            "phase": "idle",
            "message": "Server ready",
            "percent": 0,
            "sampled_frames": 0,
            "processed_frames": 0,
            "total_frames": 0,
            "viewer_port": args.viewer_port,
            "model_ready": False,
            "avg_inference_ms": 0.0,
            "last_inference_ms": 0.0,
            "model_fps": 0.0,
            "stream_fps": 0.0,
        }

    def set_progress(
        self,
        phase: str,
        message: str,
        percent: int,
        processed_frames: int | None = None,
        sampled_frames: int | None = None,
        total_frames: int | None = None,
        model_ready: bool | None = None,
        avg_inference_ms: float | None = None,
        last_inference_ms: float | None = None,
        model_fps: float | None = None,
        stream_fps: float | None = None,
    ) -> None:
        self.progress["phase"] = phase
        self.progress["message"] = message
        self.progress["percent"] = max(0, min(100, int(percent)))
        if processed_frames is not None:
            self.progress["processed_frames"] = int(processed_frames)
        if sampled_frames is not None:
            self.progress["sampled_frames"] = int(sampled_frames)
        if total_frames is not None:
            self.progress["total_frames"] = int(total_frames)
        if model_ready is not None:
            self.progress["model_ready"] = bool(model_ready)
        if avg_inference_ms is not None:
            self.progress["avg_inference_ms"] = float(avg_inference_ms)
        if last_inference_ms is not None:
            self.progress["last_inference_ms"] = float(last_inference_ms)
        if model_fps is not None:
            self.progress["model_fps"] = float(model_fps)
        if stream_fps is not None:
            self.progress["stream_fps"] = float(stream_fps)
        print(f"[e2e_io_streaming] {phase}: {message} ({self.progress['percent']}%)")

    def preload_model(self) -> None:
        self.set_progress("startup", "Loading model weights", 5, model_ready=False)
        self.runner._ensure_model_loaded()
        self.set_progress("idle", "Model loaded and ready", 100, model_ready=True)

    def viewer_url(self, host: str) -> str:
        return f"http://{host}:{self.args.viewer_port}"

    def list_root_videos(self) -> list[str]:
        return sorted([p.name for p in self.video_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"])

    async def run_existing_file(self, filename: str, fps: float, scale_frames: int, host: str) -> dict[str, Any]:
        async with self.lock:
            if self.active_session is not None:
                raise HTTPException(status_code=409, detail="Another session is already active")
            target = (self.video_dir / filename).resolve()
            if target.parent != self.video_dir or not target.is_file():
                raise HTTPException(status_code=404, detail=f"Video not found in e2e_io_streaming/videos: {filename}")
            session_id = f"file-{int(asyncio.get_running_loop().time() * 1000)}"
            self.active_session = SessionConfig(
                session_id=session_id,
                scale_frames=scale_frames,
                fps=fps,
                total_frames=0,
                source_name=target.name,
            )
            print(f"[e2e_io_streaming] run-file: started filename={target.name!r} fps={fps} scale_frames={scale_frames}")
            self.set_progress("preparing", f"Opening {target.name} from e2e_io_streaming/videos", 3, processed_frames=0, sampled_frames=0, total_frames=0)
            try:
                result = await asyncio.to_thread(
                    self.runner.run_root_video,
                    str(target),
                    fps,
                    scale_frames,
                    self.args.viewer_port,
                    self.set_progress,
                )
            finally:
                self.active_session = None
            self.set_progress("done", f"Completed {target.name}", 100, processed_frames=result["num_frames"], sampled_frames=result["num_frames"], total_frames=result["num_frames"])
            result["viewer_url"] = self.viewer_url(host)
            result["session_id"] = session_id
            result["sampled_frames"] = result["num_frames"]
            print(f"[e2e_io_streaming] run-file: completed session={session_id} viewer={result['viewer_url']}")
            return result

    async def reset(self) -> dict[str, Any]:
        async with self.lock:
            await asyncio.to_thread(self.runner.reset)
            self.active_session = None
            self.set_progress("idle", "Session reset", 0, processed_frames=0, sampled_frames=0, total_frames=0, model_ready=self.progress.get("model_ready", False))
            return {"type": "reset_complete"}

    async def begin_browser_stream(
        self,
        session_name: str,
        sample_fps: float,
        scale_frames: int,
        total_frames: int,
    ) -> str:
        async with self.lock:
            if self.active_session is not None:
                raise HTTPException(status_code=409, detail="Another session is already active")
            session_id = f"browser-{int(asyncio.get_running_loop().time() * 1000)}"
            self.active_session = SessionConfig(
                session_id=session_id,
                scale_frames=scale_frames,
                fps=sample_fps,
                total_frames=total_frames,
                source_name=session_name or "browser-stream",
            )
            await asyncio.to_thread(
                self.runner.start_browser_stream,
                scale_frames,
                self.args.viewer_port,
                total_frames,
            )
            self.set_progress(
                "browser_stream",
                f"Waiting for browser frames from {session_name or 'browser-stream'}",
                1,
                processed_frames=0,
                sampled_frames=0,
                total_frames=total_frames,
            )
            return session_id

    async def push_browser_frame(self, image_bytes: bytes) -> dict[str, Any]:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.asarray(image)
        result = await asyncio.to_thread(self.runner.stream_browser_frame, rgb)
        self.set_progress(
            "browser_stream",
            "Processing browser-owned frame stream",
            result.get("percent", self.progress.get("percent", 0)),
            processed_frames=result.get("processed_frames", 0),
            sampled_frames=result.get("sampled_frames", 0),
            total_frames=result.get("total_frames", self.progress.get("total_frames", 0)),
            avg_inference_ms=result.get("avg_inference_ms", self.progress.get("avg_inference_ms", 0.0)),
            last_inference_ms=result.get("last_inference_ms", self.progress.get("last_inference_ms", 0.0)),
            model_fps=result.get("model_fps", self.progress.get("model_fps", 0.0)),
            stream_fps=result.get("stream_fps", self.progress.get("stream_fps", 0.0)),
        )
        return result

    async def end_browser_stream(self) -> dict[str, Any]:
        result = await asyncio.to_thread(self.runner.finish_browser_stream)
        self.active_session = None
        total_frames = result.get("total_frames", 0)
        self.set_progress(
            "done",
            "Completed browser-owned frame stream",
            100,
            processed_frames=result.get("processed_frames", total_frames),
            sampled_frames=result.get("sampled_frames", total_frames),
            total_frames=total_frames,
            avg_inference_ms=result.get("avg_inference_ms", 0.0),
            last_inference_ms=result.get("last_inference_ms", 0.0),
            model_fps=result.get("model_fps", 0.0),
            stream_fps=result.get("stream_fps", 0.0),
        )
        return result

    def _extract_video_frames(self, video_path: Path, sample_fps: float, output_dir: Path) -> list[str]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open uploaded video: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(1, round(src_fps / max(sample_fps, 1.0)))
        saved_paths: list[str] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frame_path = output_dir / f"{len(saved_paths):06d}.png"
                cv2.imwrite(str(frame_path), frame)
                saved_paths.append(str(frame_path))
            idx += 1
            if total_frames > 0 and idx % 20 == 0:
                extract_percent = 30 + int(20 * min(idx / total_frames, 1.0))
                self.set_progress(
                    "extracting",
                    f"Extracted {len(saved_paths)} sampled frames",
                    extract_percent,
                    sampled_frames=len(saved_paths),
                )
        cap.release()
        if not saved_paths:
            raise RuntimeError(f"No frames were extracted (src_fps={src_fps}, total_frames={total_frames})")
        return saved_paths

    async def upload_video(self, upload: UploadFile, sample_fps: float, scale_frames: int, session_name: str, host: str) -> dict[str, Any]:
        async with self.lock:
            if self.active_session is not None:
                raise HTTPException(status_code=409, detail="Another session is already active")
            print(f"[e2e_io_streaming] upload-video: started filename={upload.filename!r} sample_fps={sample_fps} scale_frames={scale_frames}")
            self.set_progress("uploading", f"Receiving {upload.filename or 'video'}", 10, sampled_frames=0)
            with tempfile.TemporaryDirectory(prefix="lingbot_map_upload_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                video_path = tmpdir_path / (upload.filename or "upload.mp4")
                with video_path.open("wb") as fh:
                    fh.write(await upload.read())
                print(f"[e2e_io_streaming] upload-video: saved upload to {video_path}")
                self.set_progress("extracting", "Decoding uploaded video on backend", 25, sampled_frames=0)
                frames_dir = tmpdir_path / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)
                frame_paths = self._extract_video_frames(video_path, sample_fps=sample_fps, output_dir=frames_dir)
                print(f"[e2e_io_streaming] upload-video: extracted {len(frame_paths)} frames")
                session_id = f"upload-{int(asyncio.get_running_loop().time() * 1000)}"
                self.active_session = SessionConfig(
                    session_id=session_id,
                    scale_frames=scale_frames,
                    fps=sample_fps,
                    total_frames=len(frame_paths),
                    source_name=session_name or upload.filename or "upload",
                )
                try:
                    self.set_progress("preprocessing", f"Preprocessing {len(frame_paths)} frames", 55, sampled_frames=len(frame_paths))
                    self.set_progress("inference", f"Running model on {len(frame_paths)} frames", 70, sampled_frames=len(frame_paths))
                    result = await asyncio.to_thread(
                        self.runner.run_uploaded_video,
                        frame_paths,
                        scale_frames,
                        self.args.viewer_port,
                    )
                finally:
                    self.active_session = None
                self.set_progress("rendering", "Updating viewer", 90, sampled_frames=len(frame_paths))
                self.set_progress("done", f"Completed {len(frame_paths)} frames", 100, sampled_frames=len(frame_paths))
                result["viewer_url"] = self.viewer_url(host)
                result["session_id"] = session_id
                result["sampled_frames"] = len(frame_paths)
                print(f"[e2e_io_streaming] upload-video: completed session={session_id} viewer={result['viewer_url']}")
                return result


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="LingBot-Map E2E I/O Streaming Demo")
    app.state.demo_server = DemoServer(args)

    static_dir = Path(__file__).resolve().parent
    app.mount("/assets", StaticFiles(directory=static_dir), name="e2e_io_streaming_assets")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        html_path = static_dir / "index.html"
        return HTMLResponse(html_path.read_text())

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/videos")
    async def videos() -> JSONResponse:
        return JSONResponse({"videos": app.state.demo_server.list_root_videos()})

    @app.post("/reset")
    async def reset() -> JSONResponse:
        result = await app.state.demo_server.reset()
        return JSONResponse(result)

    @app.get("/status")
    async def status() -> JSONResponse:
        active = app.state.demo_server.active_session
        return JSONResponse(
            {
                "active_session": None if active is None else {
                    "session_id": active.session_id,
                    "scale_frames": active.scale_frames,
                    "fps": active.fps,
                    "total_frames": active.total_frames,
                    "source_name": active.source_name,
                },
                "viewer_port": args.viewer_port,
                "progress": app.state.demo_server.progress,
            }
        )

    @app.post("/upload-video")
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),
        sample_fps: float = Form(8.0),
        scale_frames: int = Form(8),
        session_name: str = Form("browser-upload"),
    ) -> JSONResponse:
        host = request.url.hostname or "localhost"
        result = await app.state.demo_server.upload_video(
            upload=file,
            sample_fps=sample_fps,
            scale_frames=scale_frames,
            session_name=session_name,
            host=host,
        )
        return JSONResponse(result)

    @app.post("/run-file")
    async def run_file(
        request: Request,
        filename: str = Form(...),
        sample_fps: float = Form(8.0),
        scale_frames: int = Form(8),
    ) -> JSONResponse:
        host = request.url.hostname or "localhost"
        result = await app.state.demo_server.run_existing_file(
            filename=filename,
            fps=sample_fps,
            scale_frames=scale_frames,
            host=host,
        )
        return JSONResponse(result)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        session_started = False
        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")
                    if msg_type == "start_stream":
                        sample_fps = float(payload.get("sample_fps", 8.0))
                        scale_frames = int(payload.get("scale_frames", 8))
                        total_frames = int(payload.get("total_frames", 0))
                        session_name = str(payload.get("session_name", "browser-stream"))
                        session_id = await app.state.demo_server.begin_browser_stream(
                            session_name=session_name,
                            sample_fps=sample_fps,
                            scale_frames=scale_frames,
                            total_frames=total_frames,
                        )
                        session_started = True
                        await websocket.send_text(json.dumps({
                            "type": "session_started",
                            "session_id": session_id,
                            "viewer_url": app.state.demo_server.viewer_url(websocket.url.hostname or "localhost"),
                        }))
                    elif msg_type == "end_stream":
                        result = await app.state.demo_server.end_browser_stream()
                        result["viewer_url"] = app.state.demo_server.viewer_url(websocket.url.hostname or "localhost")
                        await websocket.send_text(json.dumps(result))
                        break
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"Unsupported websocket message type: {msg_type}",
                        }))
                elif "bytes" in message and message["bytes"] is not None:
                    if not session_started:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Send start_stream before sending frame bytes.",
                        }))
                        continue
                    result = await app.state.demo_server.push_browser_frame(message["bytes"])
                    await websocket.send_text(json.dumps(result))
                elif message.get("type") == "websocket.disconnect":
                    break
        except Exception as exc:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(exc),
            }))
        finally:
            if session_started and app.state.demo_server.active_session is not None:
                await app.state.demo_server.reset()
            await websocket.close()

    @app.on_event("startup")
    async def preload_model() -> None:
        await asyncio.to_thread(app.state.demo_server.preload_model)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom streaming demo server for LingBot-Map")
    parser.add_argument("--model_path", type=str, default="checkpoints/lingbot-map.pt")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--viewer_port", type=int, default=8080)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--patch_size", type=int, default=14)
    parser.add_argument("--enable_3d_rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_frame_num", type=int, default=1024)
    parser.add_argument("--kv_cache_sliding_window", type=int, default=64)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--camera_num_iterations", type=int, default=1)
    parser.add_argument("--use_sdpa", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point_size", type=float, default=0.00001)
    parser.add_argument("--conf_threshold", type=float, default=1.5)
    parser.add_argument("--downsample_factor", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
