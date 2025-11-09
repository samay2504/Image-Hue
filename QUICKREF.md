# 🎨 Colorful Image Colorization - Quick Reference

## One-Line Quick Starts

### 🚀 Fastest Demo (No Training Needed)
```bash
# Windows
git clone <repo> && cd colorization && .\scripts\setup_local.ps1 && .\scripts\run_streamlit.ps1

# Linux/macOS
git clone <repo> && cd colorization && ./scripts/setup_local.sh && ./scripts/run_streamlit.sh
```

### 🐳 Docker Quick Start
```bash
cd colorization/docker && docker-compose up
# Access: http://localhost:8501 (Streamlit) or http://localhost:7860 (Gradio)
```

---

## 📦 Installation Commands

### PyTorch with CUDA 13.0 (Critical!)
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

---

## 🎨 Common Use Cases

### 1. Colorize Single Image
```bash
python -m src.infer input.jpg --output colored.jpg --method classification --temperature 0.38
```

### 2. Batch Colorization
```bash
python -m src.infer input_folder/ --output output_folder/ --method classification
```

### 3. Python API
```python
from src.infer import ColorizationInference
engine = ColorizationInference(model_path="checkpoints/best.pth")
result = engine.colorize_image("input.jpg", method="classification", temperature=0.38)
```

### 4. Launch UIs
```bash
# Streamlit
python -m streamlit run src/ui/streamlit_app.py

# Gradio
python src/ui/gradio_app.py --port 7860
```

---

## 🏋️ Training Commands

### Quick Training (Test Pipeline)
```bash
python -m src.train --config configs/quicktrain.yaml --train_dir data/train --val_dir data/val
```

### Full Training (Paper Settings)
```bash
python -m src.train --config configs/fulltrain.yaml --train_dir /path/to/imagenet/train --val_dir /path/to/imagenet/val
```

### Resume from Checkpoint
```bash
python -m src.train --config configs/quicktrain.yaml --train_dir data/train --resume checkpoints/epoch_0010.pth
```

---

## 🔧 Configuration Cheat Sheet

### Model Types
- `paper`: Full PaperNet (10-14GB GPU, most accurate)
- `mobile`: Lightweight variant (4-6GB GPU, fast)
- `l2`: L2 regression baseline (comparison)

### Temperature Values
- `0.1-0.3`: Vibrant, saturated colors
- `0.38`: Balanced (paper recommendation) ⭐
- `0.5-1.0`: Conservative, desaturated

### Memory Optimization
```yaml
# configs/quicktrain.yaml
use_amp: true              # FP16 (saves ~40% memory)
batch_size: 8              # Reduce if OOM
model:
  model_type: "mobile"     # Use mobile variant
  base_channels: 16        # Reduce from 32 if needed
```

---

## 🐛 Troubleshooting

### OOM Error
```bash
# Solution 1: Use mobile model
model_type: "mobile"

# Solution 2: Reduce batch size
batch_size: 4

# Solution 3: Enable tiling for inference
result = engine.colorize_image(img, tile_size=256)
```

### GPU Not Detected
```bash
# Reinstall PyTorch
pip3 uninstall torch torchvision
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### Import Errors
```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/macOS
$env:PYTHONPATH += ";$(pwd)"              # Windows
```

---

## 📊 Model Configurations

### For 6GB GPU (RTX 3060)
```yaml
model:
  model_type: "mobile"
  base_channels: 32
batch_size: 16
use_amp: true
```

### For 12GB+ GPU
```yaml
model:
  model_type: "paper"
batch_size: 32
use_amp: true
```

### For CPU Only
```yaml
model:
  model_type: "mobile"
  base_channels: 16
batch_size: 4
use_amp: false
```

---

## 🧪 Testing Commands

```bash
# Run all tests
pytest src/tests/ -v

# Run with coverage
pytest src/tests/ --cov=src --cov-report=html

# Run specific test
pytest src/tests/test_ops.py::TestColorSpaceConversion::test_roundtrip_conversion -v

# Skip slow tests
pytest src/tests/ -m "not slow"
```

---

## 🐳 Docker Commands

```bash
# Build
docker build -t colorize:latest -f docker/Dockerfile .

# Run Streamlit
docker run --gpus all -p 8501:8501 colorize:latest

# Run with Docker Compose
cd docker && docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## 📂 Directory Quick Reference

```
src/models/ops.py         # Color space, quantization, encoding
src/models/model.py       # Model architectures
src/train.py              # Training script
src/infer.py              # Inference engine
src/ui/streamlit_app.py   # Streamlit UI
src/ui/gradio_app.py      # Gradio UI
configs/quicktrain.yaml   # Quick training config
configs/fulltrain.yaml    # Full paper training config
```

---

## 🎯 Paper Implementation Reference

| Component | Paper Value | Config Key |
|-----------|-------------|------------|
| ab bins (Q) | 313 | `num_classes: 313` |
| Grid size | 10 | (hardcoded in ops.py) |
| Soft-encoding σ | 5 | `SIGMA_SOFT = 5.0` |
| K neighbors | 5 | `K_NEIGHBORS = 5` |
| Temperature (T) | 0.38 | `temperature: 0.38` |
| Rebalancing λ | 0.5 | `LAMBDA_REBALANCE = 0.5` |
| Rebalancing σ | 5 | `SIGMA_REBALANCE = 5.0` |
| Learning rate | 3e-5 → 1e-5 → 3e-6 | `learning_rate: 3e-5` |
| Weight decay | 1e-3 | `weight_decay: 1e-3` |
| Adam β₁ | 0.9 | (hardcoded) |
| Adam β₂ | 0.99 | (hardcoded) |

---

## 💡 Pro Tips

1. **First time setup**: Run `./scripts/verify_system.sh` to check everything
2. **Out of memory**: Use `model_type: mobile` and `batch_size: 8`
3. **Slow training**: Enable `use_amp: true` for 2x speedup
4. **Large images**: Use `tile_size=512` in inference
5. **Cache results**: Redis caching is automatic with Docker Compose
6. **Quick test**: Use synthetic data from `data/README.md` script

---

## 📞 Help

- **Issues**: Open GitHub issue
- **Documentation**: See full README.md
- **Paper**: [arXiv:1603.08511](https://arxiv.org/abs/1603.08511)

---

**Made with ❤️ for reproducible deep learning research**
