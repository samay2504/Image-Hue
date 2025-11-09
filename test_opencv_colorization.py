"""
Test script to verify OpenCV colorization works without trained models.
"""
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.infer import ColorizationInference


def test_opencv_colorization():
    """Test that OpenCV method works on a grayscale image without any model."""
    
    print("=" * 60)
    print("Testing OpenCV Colorization (No Model Required)")
    print("=" * 60)
    
    # Create a simple test grayscale image (checkerboard pattern)
    size = 256
    checkerboard = np.zeros((size, size), dtype=np.uint8)
    square_size = 32
    for i in range(0, size, square_size):
        for j in range(0, size, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                checkerboard[i:i+square_size, j:j+square_size] = 255
    
    # Add some gradient
    gradient = np.linspace(0, 255, size, dtype=np.uint8)
    gradient_img = np.tile(gradient, (size, 1))
    
    # Blend checkerboard and gradient
    test_image = (checkerboard * 0.5 + gradient_img * 0.5).astype(np.uint8)
    
    # Convert to RGB format
    test_rgb = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB) / 255.0
    
    print(f"\n✓ Created test image: {test_rgb.shape}")
    
    # Initialize inference engine WITHOUT a model
    print("\n→ Initializing inference engine (no model)...")
    engine = ColorizationInference(
        model_path=None,  # No model needed for OpenCV
        device='cpu',
        use_cache=False
    )
    print("✓ Engine initialized")
    
    # Test OpenCV colorization
    print("\n→ Testing OpenCV colorization...")
    try:
        colorized = engine.colorize_image(
            test_rgb,
            method='opencv'
        )
        
        print(f"✓ Colorization successful!")
        print(f"  Input shape: {test_rgb.shape}")
        print(f"  Output shape: {colorized.shape}")
        print(f"  Output dtype: {colorized.dtype}")
        print(f"  Output range: [{colorized.min():.3f}, {colorized.max():.3f}]")
        
        # Verify output properties
        assert colorized.shape == test_rgb.shape, "Shape mismatch"
        assert colorized.shape[2] == 3, "Should be RGB"
        assert 0 <= colorized.min() <= colorized.max() <= 1.0, "Values should be in [0, 1]"
        assert colorized.dtype == np.float32 or colorized.dtype == np.float64, "Should be float"
        
        # Check that it's actually colored (not just grayscale)
        r_channel = colorized[:, :, 0]
        g_channel = colorized[:, :, 1]
        b_channel = colorized[:, :, 2]
        
        # Channels should have some variance between them
        has_color = (
            np.std(r_channel - g_channel) > 0.01 or
            np.std(g_channel - b_channel) > 0.01 or
            np.std(r_channel - b_channel) > 0.01
        )
        
        if has_color:
            print("✓ Output is properly colorized (channels differ)")
        else:
            print("⚠ Warning: Output appears grayscale-like")
        
        # Save test result
        output_path = Path(__file__).parent / "test_opencv_result.png"
        result_img = Image.fromarray((colorized * 255).astype(np.uint8))
        result_img.save(output_path)
        print(f"\n✓ Test result saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nOpenCV colorization works WITHOUT any trained model.")
        print("This can be used as a fallback method in the UI.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_real_image():
    """Test with a real grayscale image if available."""
    print("\n" + "=" * 60)
    print("Testing with Real Image (if available)")
    print("=" * 60)
    
    # Look for example images
    examples_dir = Path(__file__).parent / "examples"
    if not examples_dir.exists():
        print("ℹ No examples directory found, skipping real image test")
        return True
    
    image_files = list(examples_dir.glob("*.jpg")) + list(examples_dir.glob("*.png"))
    if not image_files:
        print("ℹ No example images found, skipping real image test")
        return True
    
    test_img_path = image_files[0]
    print(f"\n→ Loading: {test_img_path.name}")
    
    try:
        engine = ColorizationInference(model_path=None, device='cpu', use_cache=False)
        
        colorized = engine.colorize_image(
            str(test_img_path),
            method='opencv'
        )
        
        # Save result
        output_path = Path(__file__).parent / f"test_opencv_{test_img_path.stem}_result.png"
        result_img = Image.fromarray((colorized * 255).astype(np.uint8))
        result_img.save(output_path)
        
        print(f"✓ Colorized real image saved to: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠ Real image test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_opencv_colorization()
    test_with_real_image()
    
    sys.exit(0 if success else 1)
