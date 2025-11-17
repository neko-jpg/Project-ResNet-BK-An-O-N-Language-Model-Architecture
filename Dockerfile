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

# Install PyTorch with CUDA 11.8 support (pinned versions for reproducibility)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies (pinned versions)
RUN pip install \
    numpy==1.24.3 \
    scipy==1.11.3 \
    matplotlib==3.8.0 \
    seaborn==0.13.0 \
    pandas==2.1.1 \
    scikit-learn==1.3.1 \
    tqdm==4.66.1 \
    pyyaml==6.0.1 \
    tensorboard==2.15.1 \
    wandb==0.16.0

# Install Hugging Face ecosystem (pinned versions)
RUN pip install \
    transformers==4.35.0 \
    datasets==2.14.6 \
    tokenizers==0.15.0 \
    accelerate==0.24.1 \
    safetensors==0.4.0

# Install additional ML libraries
RUN pip install \
    einops==0.7.0 \
    triton==2.1.0 \
    ninja==1.11.1.1

# Install Jupyter for interactive development
RUN pip install \
    jupyter==1.0.0 \
    jupyterlab==4.0.7 \
    ipywidgets==8.1.1

# Install testing and development tools
RUN pip install \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    black==23.10.1 \
    flake8==6.1.0 \
    mypy==1.6.1

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
