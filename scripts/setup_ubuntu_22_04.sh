#!/bin/bash
#
# Setup script for Ubuntu 22.04 + RTX 5070 Ti + CUDA 12.8
#
# This script installs:
# - NVIDIA drivers (if not present)
# - PyTorch cu128 nightly build
# - Docker + nvidia-container-toolkit
# - Project dependencies
#
# Run with: bash scripts/setup_ubuntu_22_04.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Ubuntu 22.04 + RTX 5070 Ti Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Ubuntu 22.04
echo "[1/7] Checking OS version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "$ID" = "ubuntu" ] && [ "$VERSION_ID" = "22.04" ]; then
        echo -e "${GREEN}✓ Ubuntu 22.04 detected${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Not Ubuntu 22.04 (detected: $ID $VERSION_ID)${NC}"
        echo "This script is optimized for Ubuntu 22.04. Proceed with caution."
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo -e "${RED}✗ Cannot detect OS version${NC}"
    exit 1
fi

# Check for NVIDIA GPU
echo ""
echo "[2/7] Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA driver installed${NC}"
    nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${YELLOW}⚠ nvidia-smi not found${NC}"
    echo ""
    echo "Install NVIDIA drivers:"
    echo "  1. Check recommended driver: ubuntu-drivers devices"
    echo "  2. Install: sudo ubuntu-drivers autoinstall"
    echo "  3. Or manually: sudo apt install nvidia-driver-545  # or latest"
    echo "  4. Reboot: sudo reboot"
    echo ""
    echo "For RTX 5070 Ti (Blackwell architecture), you need driver >= 545"
    echo "See: https://www.nvidia.com/Download/index.aspx"
    echo ""
    read -p "Do you want to install NVIDIA drivers now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt update
        sudo ubuntu-drivers autoinstall
        echo -e "${GREEN}✓ NVIDIA drivers installed${NC}"
        echo -e "${YELLOW}⚠ Please reboot your system and run this script again${NC}"
        exit 0
    else
        echo -e "${RED}✗ Skipping NVIDIA driver installation${NC}"
        echo "You must install NVIDIA drivers before proceeding."
        exit 1
    fi
fi

# Install Python 3.10 and pip
echo ""
echo "[3/7] Installing Python 3.10 and pip..."
sudo apt update
sudo apt install -y python3.10 python3.10-dev python3-pip python3.10-venv
echo -e "${GREEN}✓ Python installed: $(python3 --version)${NC}"

# Create virtual environment (optional but recommended)
echo ""
echo "[4/7] Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "[5/7] Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ pip upgraded: $(pip --version)${NC}"

# Install PyTorch cu128 nightly
echo ""
echo "[6/7] Installing PyTorch cu128 nightly (for CUDA 12.8 + RTX 5070 Ti)..."
echo "This may take a few minutes..."
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Verify PyTorch CUDA
echo ""
echo "Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'cuDNN version: {torch.backends.cudnn.version()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch CUDA support verified${NC}"
else
    echo -e "${RED}✗ PyTorch CUDA not available${NC}"
    echo "This could mean:"
    echo "  1. NVIDIA drivers not properly installed"
    echo "  2. CUDA toolkit not compatible"
    echo "  3. PyTorch installation failed"
    exit 1
fi

# Install project dependencies
echo ""
echo "[7/7] Installing project dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Project dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ requirements.txt not found, skipping${NC}"
fi

# Install Docker and nvidia-container-toolkit (optional)
echo ""
echo "=========================================="
echo "Optional: Docker + nvidia-container-toolkit"
echo "=========================================="
read -p "Install Docker with GPU support? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Installing Docker..."
    
    # Remove old versions
    sudo apt remove -y docker docker-engine docker.io containerd runc || true
    
    # Install Docker
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg lsb-release
    
    # Add Docker's official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    
    # Set up repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✓ Docker installed${NC}"
    
    # Install nvidia-container-toolkit
    echo ""
    echo "Installing nvidia-container-toolkit..."
    
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    
    # Configure Docker to use nvidia runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    echo -e "${GREEN}✓ nvidia-container-toolkit installed${NC}"
    
    # Test Docker GPU support
    echo ""
    echo "Testing Docker GPU support..."
    if sudo docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi; then
        echo -e "${GREEN}✓ Docker GPU support verified${NC}"
    else
        echo -e "${RED}✗ Docker GPU test failed${NC}"
    fi
    
    echo ""
    echo -e "${YELLOW}⚠ Note: You need to log out and log back in for docker group changes to take effect${NC}"
fi

# Summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Your system is ready for GPU training:"
echo "  ✓ Ubuntu 22.04"
echo "  ✓ NVIDIA drivers"
echo "  ✓ Python 3.10"
echo "  ✓ PyTorch cu128 nightly"
echo "  ✓ Project dependencies"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "  ✓ Docker with GPU support"
fi
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Verify system: bash scripts/verify_system_ayaan.sh"
echo "  3. Test GPU: python scripts/verify_gpu_python.py"
echo "  4. Quick train test: python scripts/quick_gpu_train_test.py"
echo "  5. Profile DataLoader: python scripts/profile_dataloader.py /path/to/data"
echo "  6. Start training: python -m src.train --config configs/train_tiny.yaml --train_dir /path/to/data"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
echo ""
echo "PyTorch Information:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}')"
echo ""
echo -e "${GREEN}Happy training! 🚀${NC}"
