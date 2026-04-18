FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# PyTorch 2.9.1 + CUDA 12.8
RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cu128

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
