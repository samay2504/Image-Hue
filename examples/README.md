# Example Images

Place example images here for testing and demonstration.

Recommended examples:
- Grayscale portraits
- Landscape photos
- Urban scenes
- Historical photos

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

You can download sample grayscale images from:
- [Unsplash](https://unsplash.com/)
- [Pexels](https://www.pexels.com/)
- Convert color images to grayscale for testing

## Adding Examples

```bash
# Download an image
wget https://example.com/image.jpg -O examples/sample1.jpg

# Or use Python
from PIL import Image
img = Image.open('color_image.jpg').convert('L').convert('RGB')
img.save('examples/grayscale_sample.jpg')
```
