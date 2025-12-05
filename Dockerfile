# MUSE: 10B Japanese LLM
# Docker image with CUDA, PyTorch, and all dependencies

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/workspace

# System packages
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Python setup
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip setuptools wheel

# Dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Workspace
WORKDIR /workspace
COPY . /workspace/
RUN pip install -e .

# Directories
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs

# Default command
CMD ["/bin/bash"]

# ==========================================
# Usage:
# ==========================================
#
# Build:
#   docker build -t muse-llm .
#
# Run (interactive):
#   docker run --gpus all -it muse-llm bash
#
# Run (training):
#   docker run --gpus all -it \
#     -v $(pwd)/data:/workspace/data \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     muse-llm make start-japanese
#
# Run (chat):
#   docker run --gpus all -it \
#     -v $(pwd)/checkpoints:/workspace/checkpoints \
#     muse-llm make chat
