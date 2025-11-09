# OpenCV Colorization Method

## Overview

The OpenCV colorization method is a **model-free baseline** that can colorize grayscale images without requiring any trained deep learning model. This makes it perfect for:

- **Testing the UI** before training any models
- **Fallback option** when models are unavailable
- **Quick demonstrations** without GPU requirements
- **Baseline comparisons** for evaluating trained models

## How It Works (Enhanced Version)

The OpenCV method uses **intelligent multi-colormap blending** based on image characteristics:

1. **Contrast Enhancement**: Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
2. **Brightness Analysis**: Analyzes image to identify bright, dark, and mid-tone regions
3. **Multi-Colormap Application**: 
   - **Bright areas** → Cool tones (blues/cyans from COLORMAP_WINTER)
   - **Dark areas** → Warm tones (reds/oranges from COLORMAP_AUTUMN)
   - **Mid tones** → Natural tones (browns/grays from COLORMAP_BONE)
4. **Adaptive Blending**: Automatically adjusts color intensity based on image contrast
5. **Normalization**: Ensures output is in proper [0, 1] range

### Why Different Colors Now?

**Previous Version** (orange only):
- Used single COLORMAP_AUTUMN for entire image
- Result: Everything looked orange/warm

**Enhanced Version** (diverse colors):
- Uses 3 different colormaps intelligently
- Bright regions (sky, highlights) → cool/blue tones
- Dark regions (shadows, foreground) → warm/orange tones
- Creates more **natural-looking and diverse** colorization

### Algorithm Steps

```python
# 1. Enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_enhanced = clahe.apply(gray)

# 2. Analyze brightness
mean_brightness = np.mean(gray_enhanced)
std_brightness = np.std(gray_enhanced)

# 3. Apply multiple colormaps
colored_warm = cv2.applyColorMap(gray_enhanced, cv2.COLORMAP_AUTUMN)  # Reds/oranges
colored_cool = cv2.applyColorMap(gray_enhanced, cv2.COLORMAP_WINTER)  # Blues/cyans
colored_natural = cv2.applyColorMap(gray_enhanced, cv2.COLORMAP_BONE)  # Browns/grays

# 4. Create brightness-based masks
bright_mask = (gray_enhanced > mean_brightness + std * 0.5)
dark_mask = (gray_enhanced < mean_brightness - std * 0.5)
mid_mask = 1.0 - bright_mask - dark_mask

# 5. Blend colormaps based on brightness
result = bright_mask * colored_cool + dark_mask * colored_warm + mid_mask * colored_natural

# 6. Adaptive blending with grayscale
color_ratio = 0.5 + (contrast_measure * 0.3)  # More color in high-contrast images
final = color_ratio * result + (1 - color_ratio) * grayscale
```

## Usage

### Command Line

```bash
# Colorize a single image with OpenCV method
python -m src.infer input.jpg --output output.png --method opencv

# Colorize a folder
python -m src.infer input_folder/ --output output_folder/ --method opencv
```

### Python API

```python
from src.infer import ColorizationInference

# Initialize without any model
engine = ColorizationInference(
    model_path=None,  # No model needed!
    device='cpu',
    use_cache=False
)

# Colorize using OpenCV method
result = engine.colorize_image(
    'input.jpg',
    method='opencv'
)

# Save result
from PIL import Image
Image.fromarray((result * 255).astype('uint8')).save('output.png')
```

### Streamlit UI

1. Start the Streamlit app:
```bash
streamlit run src/ui/streamlit_app.py
```

2. In the sidebar:
   - **Leave "Model Checkpoint Path" empty** (no model needed)
   - Select **"🔧 OpenCV Color Transfer"** from the method dropdown
   - Click "Initialize/Reload Model"

3. Upload or select an image and click "🎨 Colorize!"

The OpenCV method will work immediately without any trained model.

## Testing

Run the test script to verify OpenCV colorization works:

```bash
python test_opencv_colorization.py
```

Expected output:
```
============================================================
Testing OpenCV Colorization (No Model Required)
============================================================

✓ Created test image: (256, 256, 3)
→ Initializing inference engine (no model)...
✓ Engine initialized
→ Testing OpenCV colorization...
✓ Colorization successful!
  Output is properly colorized (channels differ)

============================================================
✅ ALL TESTS PASSED!
============================================================

OpenCV colorization works WITHOUT any trained model.
```

## Comparison with Deep Learning Methods

| Feature | OpenCV Method | Classification Method | L2 Regression |
|---------|---------------|----------------------|---------------|
| **Requires trained model** | ❌ No | ✅ Yes | ✅ Yes |
| **GPU required** | ❌ No | Recommended | Recommended |
| **Training time** | 0 (instant) | ~12-48 hours | ~8-24 hours |
| **Quality** | Basic/Baseline | High | Medium |
| **Speed** | Very Fast | Fast | Fast |
| **Color accuracy** | Limited | Excellent | Good |
| **Use case** | Testing, Demos | Production | Baseline |

## Limitations

1. **Fixed color scheme**: Uses warm tones (autumn colors) - not adaptive to image content
2. **No semantic understanding**: Doesn't know what objects should be (e.g., sky = blue)
3. **Limited color variety**: All images get similar warm tones
4. **No learning**: Can't improve with more data

## Advantages

1. **Zero setup**: No model training required
2. **Fast**: Runs on CPU with minimal computation
3. **Consistent**: Always produces output without failures
4. **No dependencies**: Only requires OpenCV (already included)
5. **Memory efficient**: No large model weights to load

## When to Use

### ✅ Good for:
- **Initial testing** of the UI and pipeline
- **Demonstrations** before training models
- **Fallback** when trained models unavailable
- **Baseline comparisons** in research
- **Low-resource environments** without GPU

### ❌ Not recommended for:
- **Production colorization** (use trained models)
- **High-quality** results needed
- **Semantic accuracy** required (e.g., sky should be blue)
- **Diverse color palettes** needed

## Configuration

The OpenCV method parameters are hardcoded in `src/infer.py`:

```python
# Modify these in _colorize_opencv() method:
CLAHE_CLIP_LIMIT = 2.0          # Contrast enhancement strength
CLAHE_TILE_SIZE = (8, 8)        # Grid size for local contrast
COLORMAP = cv2.COLORMAP_AUTUMN  # Color scheme (try COLORMAP_SUMMER, etc.)
BLEND_RATIO_COLOR = 0.7         # 70% colored, 30% grayscale
```

### Alternative Colormaps

Try different OpenCV colormaps for varied effects:

- `COLORMAP_AUTUMN` - Warm reds/yellows (default)
- `COLORMAP_SUMMER` - Green/yellow tones
- `COLORMAP_WINTER` - Cool blues
- `COLORMAP_JET` - Rainbow spectrum
- `COLORMAP_VIRIDIS` - Purple to yellow
- `COLORMAP_PLASMA` - Dark purple to bright yellow

## Next Steps

After testing with the OpenCV method:

1. **Train a classification model**:
   ```bash
   python train.py --config configs/train_mobile.yaml
   ```

2. **Compare results** between OpenCV and your trained model

3. **Use trained model** in UI by providing checkpoint path:
   ```
   Model Checkpoint Path: checkpoints/best_model.pth
   Method: 📊 Paper Classification (Recommended)
   ```

## Troubleshooting

### Issue: "No model checkpoint provided" warning

**Solution**: This is expected! OpenCV method doesn't need a model. The warning is informational only.

### Issue: Colors look too saturated

**Solution**: Adjust the blend ratio in code (decrease `BLEND_RATIO_COLOR` from 0.7 to 0.5)

### Issue: Want different color tones

**Solution**: Change `COLORMAP_AUTUMN` to another colormap (see alternatives above)

## Summary

The OpenCV colorization method is a production-ready, model-free baseline that:

- ✅ Works immediately without training
- ✅ Runs on any hardware (CPU sufficient)
- ✅ Perfect for testing the UI
- ✅ Can be used as a fallback option
- ⚠️ Provides basic quality (not production-grade for final applications)

**For best results in production, train and use the deep learning models!**
