# Training Data

Place your training and validation datasets here.

## Recommended Structure

```
data/
├── train/
│   ├── class1/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class2/
│   │   └── ...
│   └── ... (or flat structure with all images)
│
├── val/
│   └── ... (same structure as train)
│
└── color_stats.npz (computed color distribution)
```

## Preparing Data

### 1. Download ImageNet (Full Training)

```bash
# Download from official source
# https://image-net.org/download.php

# Extract to data/train and data/val
```

### 2. Use Custom Dataset

Any folder of color images works. The code will:
- Convert to Lab space
- Extract L channel as input
- Use ab channels as target

### 3. Compute Color Statistics

```bash
python -m src.data.dataset data/train --output data/color_stats.npz
```

This computes the empirical color distribution for class rebalancing.

## Quick Test Dataset

For quick testing, create a small dataset:

```bash
# Create 100 sample images (random for testing)
python -c "
from PIL import Image
import numpy as np
from pathlib import Path

Path('data/train').mkdir(parents=True, exist_ok=True)
Path('data/val').mkdir(parents=True, exist_ok=True)

for i in range(80):
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(f'data/train/img{i:03d}.jpg')

for i in range(20):
    img = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
    Image.fromarray(img).save(f'data/val/img{i:03d}.jpg')

print('Created 100 sample images')
"
```
