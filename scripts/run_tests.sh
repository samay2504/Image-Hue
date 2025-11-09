#!/bin/bash
# Test runner script for Linux/macOS

echo "🧪 Running Colorization Tests..."
echo ""

# Activate conda environment
CONDA_ENV="d:/Projects2.0/Sem 7 Assigns/Computer Vision/Project/.conda"

if [ -d "$CONDA_ENV" ]; then
    echo "✓ Activating conda environment: $CONDA_ENV"
    source conda activate "$CONDA_ENV"
else
    echo "⚠ Conda environment not found. Using system Python."
fi

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "================================================================================"
echo "Running Unit Tests"
echo "================================================================================"
echo ""

# Run tests with coverage
pytest src/tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html -m "not slow"

echo ""
echo "================================================================================"
echo "Test Summary"
echo "================================================================================"
echo ""
echo "Coverage report generated in htmlcov/index.html"
echo ""
echo "To run GPU tests: pytest src/tests/ -v -m gpu"
echo "To run all tests: pytest src/tests/ -v"
echo ""
