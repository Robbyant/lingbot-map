# E2E I/O Streaming Demo

This directory contains an end-to-end streaming demo for LingBot-Map.

It extends the repository's existing streaming inference model path into a live application:

- server-side file mode: read a video from `e2e_io_streaming/videos/`, sample frames, and run incremental inference
- browser-owned mode: decode a local browser video and send frames to the backend over WebSocket
- live viewer: update the 3D scene during inference instead of waiting until the full sequence finishes

## Assumptions

This README assumes the base repository is already set up:

- dependencies are installed
- the model checkpoint is available
- CUDA / PyTorch are already working if you plan to run on GPU

## Run

From the repository root:

```bash
conda activate lingbot-map
python e2e_io_streaming/demo.py --host 0.0.0.0 --port 8000 --viewer_port 8080
```

Then open:

```text
http://<host>:8000
```

The 3D viewer is served separately on port `8080`.

## Input Modes

### 1. `e2e_io_streaming/videos` MP4

Place videos in:

```text
e2e_io_streaming/videos/
```

Then in the web UI:

1. choose `e2e_io_streaming/videos MP4`
2. select a video from the dropdown, or type its filename
3. set `Sample FPS` and `Scale Frames`
4. click `Start Server File`

### 2. Browser-Owned Video

In the web UI:

1. choose `Browser-Owned Video`
2. select a local video file from your machine
3. set `Sample FPS` and `Scale Frames`
4. click `Start Browser Stream`

The browser decodes frames and sends them to the backend over WebSocket.

## Notes On Streaming

The original repository demo uses streaming model inference, but wraps it in offline-style video loading and offline visualization.

This app pushes streaming further:

- frames are processed incrementally
- the model runs in streaming mode
- the viewer updates during inference

The initial `Scale Frames` window is still processed together by design. After that, frames are processed one-by-one with KV cache.

## Metrics

The UI exposes live metrics for debugging and profiling:

- `Model FPS`: model-side processed frames per second
- `Stream FPS`: end-to-end observed frame rate
- `Avg Inference`: average model inference time
- `Last Inference`: most recent inference time

These help distinguish frontend/network bottlenecks from model bottlenecks.

## Files

- `demo.py`: FastAPI app and streaming backend
- `index.html`: web UI
- `live_point_cloud_viewer.py`: live viewer built from the repository's existing point cloud viewer semantics
- `videos/`: optional local test videos for server-file mode
