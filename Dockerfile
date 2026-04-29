FROM spxiong/pytorch:2.11.0-py3.10.19-cuda13.0.2-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

ARG UBUNTU_MIRROR=https://mirrors.aliyun.com/ubuntu

RUN set -eux; \
    find /etc/apt -type f -name "*.list" -print0 | xargs -0 -r sed -i "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g"; \
    find /etc/apt -type f -name "*.list" -print0 | xargs -0 -r sed -i "s|https://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g"; \
    find /etc/apt -type f -name "*.list" -print0 | xargs -0 -r sed -i "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g"; \
    find /etc/apt -type f -name "*.list" -print0 | xargs -0 -r sed -i "s|https://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g"; \
    find /etc/apt -type f -name "*.list" -print0 | xargs -0 -r sed -i "/jammy-backports/d"; \
    for i in 1 2 3; do \
      apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 update && \
      apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 install -y --no-install-recommends --fix-missing \
        ca-certificates \
        git \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
      && break; \
      sleep 5; \
    done; \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE.txt ./
COPY lingbot_map ./lingbot_map
COPY demo.py gct_profile.py ./

RUN python -m pip install --upgrade pip
RUN python -m pip install --no-cache-dir flashinfer-python flashinfer-cubin
RUN python -m pip install --no-cache-dir flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu130 --no-deps || true
RUN python -m pip install -e ".[vis]"

EXPOSE 8080

CMD ["bash"]
