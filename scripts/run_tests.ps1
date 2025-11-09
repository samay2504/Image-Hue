# Test runner script for Windows PowerShell
# Activates conda environment and runs tests

Write-Host "🧪 Running Colorization Tests..." -ForegroundColor Cyan
Write-Host ""

# Activate conda environment
$condaEnv = "d:\Projects2.0\Sem 7 Assigns\Computer Vision\Project\.conda"

if (Test-Path $condaEnv) {
    Write-Host "✓ Activating conda environment: $condaEnv" -ForegroundColor Green
    & conda activate $condaEnv
} else {
    Write-Host "⚠ Conda environment not found. Using system Python." -ForegroundColor Yellow
}

# Check CUDA availability
Write-Host ""
Write-Host "Checking CUDA availability..." -ForegroundColor Cyan
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Blue
Write-Host "Running Unit Tests" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Blue
Write-Host ""

# Run tests with coverage
pytest src/tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html -m "not slow"

Write-Host ""
Write-Host "=" * 80 -ForegroundColor Blue
Write-Host "Test Summary" -ForegroundColor Cyan
Write-Host "=" * 80 -ForegroundColor Blue
Write-Host ""
Write-Host "Coverage report generated in htmlcov/index.html" -ForegroundColor Green
Write-Host ""
Write-Host "To run GPU tests: pytest src/tests/ -v -m gpu" -ForegroundColor Yellow
Write-Host "To run all tests: pytest src/tests/ -v" -ForegroundColor Yellow
Write-Host ""
