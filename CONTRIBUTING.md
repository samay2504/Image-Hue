# Contributing to Colorful Image Colorization

Thank you for considering contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/colorization.git
   cd colorization
   ```
3. **Set up development environment**:
   ```bash
   ./scripts/setup_local.sh  # Linux/macOS
   # or
   .\scripts\setup_local.ps1  # Windows
   ```

## 📝 Development Workflow

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**

3. **Run tests**:
   ```bash
   pytest src/tests/ -v
   ```

4. **Format code**:
   ```bash
   black src/
   isort src/
   ```

5. **Check linting**:
   ```bash
   flake8 src/
   ```

6. **Commit changes**:
   ```bash
   git add .
   git commit -m "Add: Your descriptive commit message"
   ```

7. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## 🎯 Code Style

- Follow PEP 8
- Use type hints where appropriate
- Maximum line length: 120 characters
- Use Black for formatting
- Use isort for import sorting

### Example:

```python
from typing import Tuple

import numpy as np
import torch


def colorize_image(
    image: np.ndarray,
    model: torch.nn.Module,
    temperature: float = 0.38
) -> np.ndarray:
    """
    Colorize a grayscale image.
    
    Args:
        image: Input grayscale image
        model: Colorization model
        temperature: Annealed-mean temperature
        
    Returns:
        Colorized RGB image
    """
    # Implementation
    pass
```

## 🧪 Testing

- Write tests for new features
- Maintain test coverage above 80%
- Tests should be fast (<1s per test)
- Mark slow tests with `@pytest.mark.slow`
- Mark GPU tests with `@pytest.mark.gpu`

### Test Structure:

```python
import pytest
import numpy as np

from src.models.ops import rgb_to_lab


class TestColorConversion:
    def test_rgb_to_lab_shape(self):
        """Test output shape."""
        rgb = np.random.rand(64, 64, 3)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (64, 64, 3)
    
    def test_rgb_to_lab_ranges(self):
        """Test output ranges."""
        rgb = np.random.rand(64, 64, 3)
        lab = rgb_to_lab(rgb)
        assert np.all(lab[:, :, 0] >= 0) and np.all(lab[:, :, 0] <= 100)
```

## 📚 Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Update README.md if adding new features
- Add inline comments for complex logic

## 🐛 Bug Reports

When filing a bug report, include:

1. **Description**: Clear description of the bug
2. **Steps to reproduce**: Minimal code to reproduce
3. **Expected behavior**: What should happen
4. **Actual behavior**: What actually happens
5. **Environment**:
   - OS (Windows/Linux/macOS)
   - Python version
   - PyTorch version
   - GPU (if applicable)

## ✨ Feature Requests

When requesting a feature:

1. **Use case**: Why is this feature needed?
2. **Proposed solution**: How should it work?
3. **Alternatives**: Other solutions considered
4. **Additional context**: Any other relevant info

## 🔍 Pull Request Process

1. **Ensure tests pass**: All tests must pass
2. **Update documentation**: Update relevant docs
3. **Add changelog entry**: In PR description
4. **One feature per PR**: Keep PRs focused
5. **Descriptive commit messages**: Use conventional commits

### Commit Message Format:

```
Type: Brief description (max 50 chars)

Detailed explanation if needed (max 72 chars per line).

- Bullet points for changes
- Multiple changes if needed

Fixes #123
```

**Types:**
- `Add`: New feature
- `Fix`: Bug fix
- `Refactor`: Code refactoring
- `Docs`: Documentation changes
- `Test`: Test additions/changes
- `Chore`: Maintenance tasks

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

## 📞 Questions?

- Open a GitHub issue with the `question` label
- Reach out to maintainers

Thank you for contributing! 🎨
