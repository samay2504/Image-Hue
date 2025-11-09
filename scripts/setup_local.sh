#!/bin/bash
# Setup script for local development (Linux/macOS/WSL2)

set -e

echo "🎨 Setting up Colorful Image Colorization project..."

# Check Python version
python_version=$(python --version 2>&1 | grep -oP '\d+\.\d+' || python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.10" | bc) -eq 1 ]]; then
    echo "❌ Python 3.10 or higher required. Current: $python_version"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.0 support
echo "🔥 Installing PyTorch with CUDA 13.0..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install other requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install dev requirements
echo "🛠️ Installing dev requirements..."
pip install -r requirements-dev.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints cache logs examples data runs

# Download example images (optional)
echo "🖼️ Setting up example images..."
# Add your example image download logic here if needed

echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate  (Linux/macOS/WSL2)"
echo "  .venv\\Scripts\\activate   (Windows PowerShell)"
echo ""
echo "To verify your GPU setup, run:"
echo "  python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"
