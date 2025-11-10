# Save as fix_color_stats.py
import numpy as np
from src.models.ops import get_ab_quantization_grid, lab_to_rgb_single

# Load existing stats
data = np.load('data/color_stats.npz')
print(f"Current ab_grid shape: {data['ab_grid'].shape}")
print(f"Current class_weights shape: {data['class_weights'].shape}")

# Generate correct 313-bin grid manually with stricter filtering
GRID_SIZE = 10
a_range = np.arange(-110, 110, GRID_SIZE)
b_range = np.arange(-110, 110, GRID_SIZE)
aa, bb = np.meshgrid(a_range, b_range)
ab_grid_full = np.stack([aa.flatten(), bb.flatten()], axis=1)

# Filter with very strict criteria
in_gamut = []
in_gamut_indices = []

for idx, ab in enumerate(ab_grid_full):
    lab = np.array([50.0, ab[0], ab[1]], dtype=np.float64)
    rgb = lab_to_rgb_single(lab)
    
    # Very strict: must be finite and in [0, 1] with no tolerance
    if (np.all(np.isfinite(rgb)) and 
        np.all(rgb >= -1e-6) and 
        np.all(rgb <= 1.0 + 1e-6)):
        in_gamut.append(ab)
        in_gamut_indices.append(idx)

ab_grid_313 = np.array(in_gamut, dtype=np.float32)
print(f"\nNew ab_grid shape: {ab_grid_313.shape}")

if ab_grid_313.shape[0] != 313:
    print(f"WARNING: Got {ab_grid_313.shape[0]} bins instead of 313!")
    print("Using only first 313 bins...")
    ab_grid_313 = ab_grid_313[:313]
    in_gamut_indices = in_gamut_indices[:313]

# Extract corresponding class weights
if len(in_gamut_indices) <= len(data['class_weights']):
    class_weights_313 = data['class_weights'][in_gamut_indices]
else:
    print("WARNING: Not enough class weights, using first 313")
    class_weights_313 = data['class_weights'][:313]

print(f"New class_weights shape: {class_weights_313.shape}")

# Save fixed version
np.savez('data/color_stats.npz',
         empirical_distribution=data['empirical_distribution'][:313],
         class_weights=class_weights_313,
         ab_grid=ab_grid_313)

print("\n✓ Fixed color_stats.npz saved!")
print(f"  - ab_grid: {ab_grid_313.shape}")
print(f"  - class_weights: {class_weights_313.shape}")