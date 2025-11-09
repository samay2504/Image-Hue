#!/bin/bash
# Docker build and run script

set -e

echo "🐳 Building and running Colorful Image Colorization with Docker..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose not found. Please install docker-compose first."
    exit 1
fi

# Check NVIDIA Docker support (optional)
if command -v nvidia-smi &> /dev/null; then
    if ! docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
        echo "⚠️  NVIDIA Container Toolkit not configured. GPU support disabled."
        echo "Install with: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    else
        echo "✅ NVIDIA Docker support detected"
    fi
fi

# Build and start services
echo "🔨 Building Docker images..."
cd docker
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "✅ Services started!"
echo ""
echo "Access the applications:"
echo "  📊 Streamlit UI:  http://localhost:8501"
echo "  🎨 Gradio UI:     http://localhost:7860"
echo "  🗄️  Redis:         localhost:6379"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
