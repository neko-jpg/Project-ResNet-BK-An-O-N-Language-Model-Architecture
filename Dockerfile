# Dockerfile for Mamba-Killer ResNet-BK
# Provides reproducible environment with all dependencies

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install pinned dependencies (GPU build) using shared requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install project in development mode
RUN pip install -e .

# Create directories for data and checkpoints
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/logs /workspace/results

# Set up Jupyter configuration
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports for Jupyter and TensorBoard
EXPOSE 8888 6006

# Set default command
CMD ["/bin/bash"]

# Build instructions:
# docker build -t mamba-killer-resnet-bk:latest .
#
# Run instructions:
# docker run --gpus all -it -p 8888:8888 -p 6006:6006 \
#   -v $(pwd)/data:/workspace/data \
#   -v $(pwd)/checkpoints:/workspace/checkpoints \
#   mamba-killer-resnet-bk:latest
#
# Run Jupyter:
# docker run --gpus all -it -p 8888:8888 -p 6006:6006 \
#   -v $(pwd)/data:/workspace/data \
#   -v $(pwd)/checkpoints:/workspace/checkpoints \
#   mamba-killer-resnet-bk:latest \
#   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
