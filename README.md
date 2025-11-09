# 🎨 Colorful Image Colorization

Production-ready implementation of **"Colorful Image Colorization"** (Zhang, Isola, Efros - ECCV 2016) with interactive demos, memory-safe training, and comprehensive tooling.

[![CI](https://github.com/yourusername/colorization/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/colorization/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Features

- **📊 Paper-Accurate Implementation**: Classification-based colorization with 313-bin quantized ab space, class rebalancing, and annealed-mean decoding (T=0.38)
- **🎬 Interactive UIs**: Beautiful Streamlit and Gradio frontends with blend animations, temperature sliders, and method comparison
- **🚀 Production-Ready**: Docker support, Redis caching, memory safeguards (FP16, gradient checkpointing, tiling)
- **💾 Memory-Safe**: Automatic OOM handling for RTX 3060 6GB and similar GPUs
- **🔄 Multiple Methods**: Classification (paper), L2 regression baseline, OpenCV color transfer
- **🧪 Fully Tested**: Unit tests, integration tests, CI/CD with GitHub Actions
- **📦 Cross-Platform**: Works on Linux, Windows (WSL2), and macOS with local venv or Docker

## 🚀 Quick Start

### Option 1: Local Installation (Recommended for Development)

#### Windows (PowerShell)
```powershell
# Clone repository
git clone https://github.com/yourusername/colorization.git
cd colorization

# Run setup script
.\scripts\setup_local.ps1

# Activate environment
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Linux / macOS / WSL2
```bash
# Clone repository
git clone https://github.com/yourusername/colorization.git
cd colorization

# Run setup script
chmod +x scripts/*.sh
./scripts/setup_local.sh

# Activate environment
source .venv/bin/activate

# Install PyTorch with CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Option 2: Docker (Recommended for Production)

```bash
# Build and run with Docker Compose
cd docker
docker-compose up --build

# Access UIs
# Streamlit: http://localhost:8501
# Gradio: http://localhost:7860
```

**Prerequisites for Docker GPU support:**
- Docker Desktop (Windows) or Docker Engine (Linux)
- NVIDIA Container Toolkit ([installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- NVIDIA GPU drivers

## 🎨 Running the UIs

### Streamlit (Recommended)
```bash
# Windows
.\scripts\run_streamlit.ps1

# Linux/macOS
./scripts/run_streamlit.sh

# Or directly
python -m streamlit run src/ui/streamlit_app.py --server.port 8501
```

### Gradio
```bash
# Windows
.\scripts\run_gradio.ps1

# Linux/macOS
./scripts/run_gradio.sh

# Or directly
python src/ui/gradio_app.py --port 7860
```

Access the UIs:
- **Streamlit**: http://localhost:8501
- **Gradio**: http://localhost:7860

## 🏋️ Training

### Quick Training (Small Dataset)

```bash
# Prepare your dataset
# Structure: data/train/<images>, data/val/<images>

# Compute color statistics (optional but recommended)
python -m src.data.dataset data/train --output data/color_stats.npz

# Train with quick config
python -m src.train \
    --config configs/quicktrain.yaml \
    --train_dir data/train \
    --val_dir data/val
```

### Full Training (Paper Settings)

```bash
# For ImageNet-scale training
python -m src.train \
    --config configs/fulltrain.yaml \
    --train_dir /path/to/imagenet/train \
    --val_dir /path/to/imagenet/val
```

**Training Configuration:**

Edit `configs/quicktrain.yaml` or `configs/fulltrain.yaml`:

```yaml
model:
  model_type: "mobile"  # Options: paper, mobile, l2
  num_classes: 313
  base_channels: 32

num_epochs: 50
batch_size: 16
learning_rate: 1e-4
use_amp: true  # FP16 mixed precision
```

**Memory Management:**
- Model automatically scales batch size if OOM occurs
- Use `model_type: mobile` for 6GB GPUs
- Use `model_type: paper` for 12GB+ GPUs
- FP16 mixed precision reduces memory by ~40%

## 🖼️ Inference

### Command-Line

```bash
# Single image
python -m src.infer \
    input.jpg \
    --output colorized.jpg \
    --model checkpoints/best_model.pth \
    --method classification \
    --temperature 0.38

# Batch processing
python -m src.infer \
    input_folder/ \
    --output output_folder/ \
    --model checkpoints/best_model.pth \
    --method classification
```

### Python API

```python
from src.infer import ColorizationInference
from PIL import Image

# Initialize engine
engine = ColorizationInference(
    model_path="checkpoints/best_model.pth",
    device="cuda",  # or "cpu"
    use_cache=True,
    redis_url="redis://localhost:6379"
)

# Colorize image
img = Image.open("grayscale.jpg")
result = engine.colorize_image(
    img,
    method="classification",
    temperature=0.38
)

# Save result
result_img = Image.fromarray((result * 255).astype('uint8'))
result_img.save("colorized.jpg")

# Create blend animation
frames = engine.create_blend_animation(
    img,
    num_frames=30
)
```

## 📊 Methods

### 1. Classification-Based (Paper Method) ⭐ Recommended

```python
result = engine.colorize_image(img, method="classification", temperature=0.38)
```

- **Description**: Treats colorization as per-pixel classification over 313 quantized ab bins
- **Features**: Class rebalancing, soft-encoding, annealed-mean decoding
- **Temperature**: 
  - Lower (0.1-0.3): More vibrant, saturated colors
  - Default (0.38): Balanced (paper recommendation)
  - Higher (0.5-1.0): More conservative, desaturated

### 2. L2 Regression Baseline

```python
result = engine.colorize_image(img, method="l2")
```

- Direct regression to ab values
- Simpler but tends to produce desaturated results
- Useful for comparison

### 3. OpenCV Color Transfer

```python
result = engine.colorize_image(img, method="opencv")
```

- Simple baseline using traditional CV techniques
- No neural network required
- Limited quality

## 🎬 UI Features

### Streamlit UI

- 📤 **Image Upload**: Drag & drop or browse
- 🎯 **Method Selector**: Switch between classification, L2, OpenCV
- 🌡️ **Temperature Slider**: Adjust color vibrancy (0.01-1.0)
- 🎬 **Blend Animation**: Smooth grayscale→color transition
- 📊 **Side-by-Side Comparison**: Visual before/after
- 💾 **Download Results**: Save colorized images
- 📈 **System Monitor**: GPU memory usage, cache stats

### Gradio UI

- All Streamlit features
- 🖼️ **Animation Gallery**: View all blend frames
- 🎛️ **Real-time Blend Slider**: Interactive color mixing
- 🔗 **Shareable Links**: Public demo URLs (with --share)

## 🐳 Docker Usage

### Build and Run

```bash
# Build image
docker build -t colorize-app:latest -f docker/Dockerfile .

# Run Streamlit
docker run --gpus all -p 8501:8501 colorize-app:latest

# Run Gradio
docker run --gpus all -p 7860:7860 colorize-app:latest \
    python src/ui/gradio_app.py --port 7860

# Or use Docker Compose (recommended)
docker-compose up
```

### Docker Compose Services

```yaml
# docker-compose.yml includes:
- redis: Caching service (port 6379)
- app-streamlit: Streamlit UI (port 8501)
- app-gradio: Gradio UI (port 7860)
```

## 🧪 Testing

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ --cov=src --cov-report=html

# Run specific test file
pytest src/tests/test_ops.py -v

# Run integration tests only
pytest src/tests/test_integration.py -v
```

## 📦 Project Structure

```
colorization/
├── src/
│   ├── models/
│   │   ├── model.py          # PaperNet, MobileLiteVariant, L2RegressionNet
│   │   └── ops.py            # Quantization, encoding, color conversion
│   ├── data/
│   │   ├── dataset.py        # Dataset loaders
│   │   └── transforms.py     # Data augmentation
│   ├── ui/
│   │   ├── streamlit_app.py  # Streamlit interface
│   │   └── gradio_app.py     # Gradio interface
│   ├── cache/
│   │   └── redis_client.py   # Redis caching with disk fallback
│   ├── utils/
│   │   ├── memory.py         # Memory management, tiling
│   │   └── logger.py         # TensorBoard and file logging
│   ├── tests/                # Unit and integration tests
│   ├── train.py              # Training script
│   └── infer.py              # Inference engine
├── configs/
│   ├── quicktrain.yaml       # Quick training config
│   └── fulltrain.yaml        # Full paper training config
├── docker/
│   ├── Dockerfile            # CUDA 13.0 Docker image
│   └── docker-compose.yml    # Multi-service setup
├── scripts/
│   ├── setup_local.sh        # Linux/macOS setup
│   ├── setup_local.ps1       # Windows setup
│   ├── run_streamlit.sh      # Launch Streamlit
│   ├── run_gradio.sh         # Launch Gradio
│   └── verify_system.sh      # System verification
├── .github/workflows/
│   └── ci.yml                # GitHub Actions CI
├── requirements.txt          # Python dependencies
├── requirements-dev.txt      # Development dependencies
└── README.md                 # This file
```

## 🔧 Configuration

### Model Configuration

```yaml
# configs/quicktrain.yaml
model:
  model_type: "mobile"    # paper | mobile | l2
  num_classes: 313        # Number of ab bins
  base_channels: 32       # Channel multiplier (mobile/l2 only)
```

### Training Hyperparameters (from Paper)

- **Optimizer**: Adam (β₁=0.9, β₂=0.99)
- **Weight Decay**: 1e-3
- **Learning Rate Schedule**:
  - Initial: 3e-5
  - → 1e-5 at 200k iterations
  - → 3e-6 at 375k iterations
- **Batch Size**: 32 (adjust for your GPU)
- **Image Size**: 256×256
- **Quantization**: Grid size 10 → 313 in-gamut bins
- **Soft-encoding**: K=5 neighbors, σ=5
- **Class Rebalancing**: σ=5, λ=0.5
- **Annealed-mean Temperature**: T=0.38

## 💾 Memory Requirements

### Training

| Model Type | GPU Memory | Batch Size | Training Speed |
|------------|-----------|------------|----------------|
| Mobile (32ch) | 4-6 GB | 16 | Fast |
| Mobile (64ch) | 8-10 GB | 16 | Fast |
| Paper (full) | 10-14 GB | 16 | Moderate |
| Paper (full) | 14-20 GB | 32 | Moderate |

**Memory Saving Techniques:**
- ✅ FP16 mixed precision (`use_amp: true`)
- ✅ Gradient checkpointing (automatic for large models)
- ✅ Auto batch size reduction on OOM
- ✅ Mobile variant (fewer parameters)

### Inference

| Image Size | GPU Memory | Tile Size |
|-----------|-----------|-----------|
| 256×256 | <1 GB | Not needed |
| 512×512 | 1-2 GB | Not needed |
| 1024×1024 | 3-4 GB | 512 recommended |
| 2048×2048 | 8+ GB | 512 required |

Use tiling for large images:
```python
result = engine.colorize_image(img, tile_size=512)
```

## 🔍 Troubleshooting

### CUDA Out of Memory

```python
# Solution 1: Use mobile variant
model:
  model_type: "mobile"
  base_channels: 16  # Reduce from 32

# Solution 2: Reduce batch size
batch_size: 8  # Down from 16

# Solution 3: Enable gradient checkpointing
# (automatically enabled for paper model)

# Solution 4: Use CPU
device: "cpu"
```

### PyTorch Not Detecting GPU

```bash
# Verify CUDA installation
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Redis Connection Issues

```bash
# Start Redis locally
# Linux/macOS
redis-server

# Windows (with WSL2)
sudo service redis-server start

# Or disable caching
# In Python:
engine = ColorizationInference(use_cache=False)
```

### Import Errors

```bash
# Ensure you're in the project root directory
cd /path/to/colorization

# Install in development mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
$env:PYTHONPATH += ";$(pwd)"  # Windows PowerShell
```

## 📚 Paper Implementation Details

### Quantization (Section 3.2)

- **ab space**: Grid size 10 → covers [-110, 110] range
- **In-gamut filtering**: Only 313 of 441 bins represent valid RGB colors
- **Soft-encoding**: Gaussian kernel (σ=5) on K=5 nearest neighbors

### Class Rebalancing (Section 3.3, Equation 2)

```
1. Compute empirical distribution p from training set
2. Smooth with Gaussian: p_smooth = GaussianFilter(p, σ=5)
3. Mix with uniform: p̃ = (1-λ)·p_smooth + λ·(1/Q)  [λ=0.5]
4. Compute weights: w = p̃^(-1)
5. Normalize: w = w / E[w]  (mean=1)
```

### Annealed-Mean Decoding (Section 3.4, Equation 5)

```
1. Apply temperature: Z̃ = Z / T
2. Compute softmax: P(·) = softmax(Z̃)
3. Expected value: âb = Σ P(q) · ab(q)
```

Lower T → more diverse/vibrant colors  
Higher T → more conservative/muted colors

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Guidelines:**
- Follow PEP 8 style
- Add tests for new features
- Update documentation
- Run `black` and `flake8` before committing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation is based on:

> **Colorful Image Colorization**  
> Richard Zhang, Phillip Isola, Alexei A. Efros  
> European Conference on Computer Vision (ECCV), 2016  
> [Paper](https://arxiv.org/abs/1603.08511) | [Project Page](http://richzhang.github.io/colorization/)

Original PyTorch implementation: [richzhang/colorization](https://github.com/richzhang/colorization)

## 📧 Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@example.com

## 🎯 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2016colorful,
  title={Colorful image colorization},
  author={Zhang, Richard and Isola, Phillip and Efros, Alexei A},
  booktitle={European conference on computer vision},
  pages={649--666},
  year={2016},
  organization={Springer}
}
```

---

Made with ❤️ for reproducible deep learning research
