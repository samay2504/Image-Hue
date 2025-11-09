# Model Checkpoints

Place trained model checkpoints here.

## Checkpoint Format

Checkpoints are saved as `.pth` files containing:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `scheduler_state_dict`: LR scheduler state (if used)
- `scaler_state_dict`: Mixed precision scaler (if used)
- `epoch`: Current epoch number
- `global_step`: Global training step
- `best_val_loss`: Best validation loss
- `config`: Full training configuration

## Loading Checkpoints

### In Python

```python
from src.infer import ColorizationInference

engine = ColorizationInference(
    model_path='checkpoints/best_model.pth',
    device='cuda'
)
```

### Resume Training

```bash
python -m src.train \
    --config configs/quicktrain.yaml \
    --train_dir data/train \
    --val_dir data/val \
    --resume checkpoints/epoch_0010.pth
```

## Pre-trained Models

For pre-trained models on ImageNet, you can:
1. Train your own (recommended for best results on your data)
2. Use the official implementation: https://github.com/richzhang/colorization

Convert official PyTorch checkpoints:
```python
import torch

# Load official checkpoint
official = torch.load('official_checkpoint.pth')

# Extract model weights (adapt as needed)
our_checkpoint = {
    'model_state_dict': official['model'],
    'config': {'model': {'model_type': 'paper', 'num_classes': 313}}
}

torch.save(our_checkpoint, 'checkpoints/converted_model.pth')
```

## Checkpoint Naming Convention

- `best_model.pth`: Best model based on validation loss
- `final_model.pth`: Final model after training completes
- `epoch_XXXX.pth`: Checkpoint at specific epoch
- `quicktrain_YYYYMMDD.pth`: Quick training checkpoint with date
