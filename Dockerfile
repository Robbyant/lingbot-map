# pytorch/pytorch images ship with Python, pip, and PyTorch pre-installed,
# so no NVIDIA registry auth or manual CUDA installation is needed.
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# System dependencies (Python/pip already present in base image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install lingbot-map with visualization extras
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e ".[vis]" && \
    pip install --no-cache-dir onnxruntime

# FlashInfer for efficient KV-cache attention (falls back to SDPA if unavailable)
RUN pip install --no-cache-dir flashinfer-python \
    -i https://flashinfer.ai/whl/cu128/torch2.9/ || \
    echo "WARNING: FlashInfer not installed — demo will use --use_sdpa fallback"

RUN mkdir -p /model /data/images

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
