#!/bin/bash
# System verification script

set -e

echo "🔍 Verifying system setup for Colorful Image Colorization..."
echo ""

# Check Python version
echo "1️⃣ Checking Python version..."
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+' || python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "   ✓ Python $python_version"

if [[ $(echo "$python_version < 3.10" | bc) -eq 1 ]]; then
    echo "   ❌ Python 3.10+ required"
else
    echo "   ✅ Python version OK"
fi

# Check CUDA availability
echo ""
echo "2️⃣ Checking CUDA/GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    
    # Check PyTorch CUDA
    if python -c "import torch" 2>/dev/null; then
        cuda_available=$(python -c "import torch; print(torch.cuda.is_available())")
        if [ "$cuda_available" == "True" ]; then
            gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
            echo "   ✅ CUDA available: $gpu_name"
            echo ""
            echo "   📦 Recommended PyTorch installation command:"
            echo "   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
        else
            echo "   ⚠️ PyTorch not detecting CUDA"
            echo "   Install PyTorch with CUDA support:"
            echo "   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
        fi
    else
        echo "   ⚠️ PyTorch not installed"
        echo "   Install with: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130"
    fi
else
    echo "   ❌ No NVIDIA GPU detected"
    echo "   CPU-only mode will be used (slower)"
fi

# Check memory
echo ""
echo "3️⃣ Checking system memory..."
if command -v free &> /dev/null; then
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    echo "   ✓ Total RAM: ${total_mem}GB"
    if [ "$total_mem" -lt 8 ]; then
        echo "   ⚠️ Low memory detected. Recommended: 16GB+"
    else
        echo "   ✅ Memory OK"
    fi
fi

# Check Docker
echo ""
echo "4️⃣ Checking Docker..."
if command -v docker &> /dev/null; then
    docker_version=$(docker --version | grep -oP '\d+\.\d+\.\d+')
    echo "   ✓ Docker $docker_version"
    
    if docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "   ✅ Docker + NVIDIA Container Toolkit OK"
    else
        echo "   ⚠️ Docker GPU support not configured"
        echo "   Install nvidia-container-toolkit for GPU support in Docker"
    fi
else
    echo "   ❌ Docker not installed"
fi

# Check Redis
echo ""
echo "5️⃣ Checking Redis..."
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "   ✅ Redis is running"
    else
        echo "   ⚠️ Redis installed but not running"
        echo "   Start with: redis-server"
    fi
else
    echo "   ⚠️ Redis not installed (optional for caching)"
    echo "   Install with: apt-get install redis-server (Ubuntu/Debian)"
fi

echo ""
echo "✅ System verification complete!"
