# Base: PyTorch 2.5.1 + CUDA 12.4 + cuDNN 9
# Requires host driver >= 525 (RTX 3090 / compute cap 8.6 confirmed)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first (layer cache-friendly)
COPY requirements.txt /tmp/requirements.txt

RUN pip install -U pip setuptools wheel && \
    pip install -r /tmp/requirements.txt && \
    # yacs is used by config.py but not listed in requirements.txt
    pip install yacs

WORKDIR /workspace

CMD ["/bin/bash"]
