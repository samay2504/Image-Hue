FROM nvidia/cuda:13.0.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY ../requirements.txt /app/requirements.txt

# Install PyTorch with CUDA 13.0 support
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY .. /app/

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/cache /app/logs /app/examples

# Expose ports for Streamlit and Gradio
EXPOSE 8501 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.ui.streamlit_app"]
