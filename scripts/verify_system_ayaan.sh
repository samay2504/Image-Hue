#!/bin/bash
# System verification script for Ubuntu 22.04 + RTX 5070 Ti setup
# Usage: bash scripts/verify_system_ayaan.sh

set -e

echo "============================================================"
echo "System Verification for Ayaan's GPU Setup"
echo "============================================================"
echo ""

# OS Information
echo "--- OS Information ---"
if command -v lsb_release &> /dev/null; then
    lsb_release -a
else
    echo "lsb_release not found, using uname..."
    uname -a
fi
echo ""

# Kernel
echo "--- Kernel ---"
uname -a
echo ""

# NVIDIA Driver & GPU
echo "--- NVIDIA Driver & GPU ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
    echo "NVIDIA Driver: OK ✓"
else
    echo "ERROR: nvidia-smi not found! ✗"
    echo ""
    echo "Please install NVIDIA drivers:"
    echo "  Ubuntu 22.04 instructions:"
    echo "  1. sudo ubuntu-drivers devices"
    echo "  2. sudo ubuntu-drivers autoinstall"
    echo "  3. sudo reboot"
    echo "  4. Verify with: nvidia-smi"
    echo ""
    echo "  Or manual install:"
    echo "  https://www.nvidia.com/download/index.aspx"
    echo ""
fi

# Docker
echo "--- Docker ---"
if command -v docker &> /dev/null; then
    docker --version
    echo ""
    if command -v nvidia-container-toolkit &> /dev/null || docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        echo "nvidia-container-toolkit: OK ✓"
    else
        echo "WARNING: nvidia-container-toolkit may not be installed"
        echo "Install with:"
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
    fi
else
    echo "Docker not installed (optional)"
fi
echo ""

# Python
echo "--- Python Environment ---"
python3 --version
echo ""
echo "Pip install command for PyTorch (cu128 nightly):"
echo "  pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
echo ""

# PyTorch installation
echo "--- PyTorch Installation ---"
if python3 -c "import torch" 2>/dev/null; then
    pip3 show torch 2>/dev/null || pip show torch 2>/dev/null || echo "torch installed but pip show failed"
    echo ""
    pip3 show torchvision 2>/dev/null || pip show torchvision 2>/dev/null || echo "torchvision installed but pip show failed"
    echo ""
    
    echo "--- PyTorch CUDA Configuration ---"
    python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Version (compiled): {torch.version.cuda}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Device Count: {torch.cuda.device_count()}')
    print(f'CUDA Device Name: {torch.cuda.get_device_name(0)}')
    print(f'CUDA Capability: {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA not available! ✗')
"
else
    echo "ERROR: PyTorch not installed! ✗"
    echo ""
    echo "Install with:"
    echo "  pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
fi
echo ""

# System Resources
echo "--- System Resources ---"
echo "CPU Cores (logical): $(nproc)"
echo "RAM Total: $(free -h | awk '/^Mem:/ {print $2}')"
echo "RAM Available: $(free -h | awk '/^Mem:/ {print $7}')"
echo ""

echo "============================================================"
echo "Verification Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "1. If nvidia-smi works and shows RTX 5070 Ti → GOOD"
echo "2. If CUDA Available: True → GOOD"
echo "3. Run: python3 scripts/verify_gpu_python.py"
echo ""
