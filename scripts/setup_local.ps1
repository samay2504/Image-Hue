# Setup script for local development (Windows PowerShell)

Write-Host "🎨 Setting up Colorful Image Colorization project..." -ForegroundColor Cyan

# Check Python version
$pythonVersion = python --version 2>&1 | Select-String -Pattern '\d+\.\d+' | ForEach-Object { $_.Matches.Value }
Write-Host "Python version: $pythonVersion"

if ([version]$pythonVersion -lt [version]"3.10") {
    Write-Host "❌ Python 3.10 or higher required. Current: $pythonVersion" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "⬆️ Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 13.0 support
Write-Host "🔥 Installing PyTorch with CUDA 13.0..." -ForegroundColor Yellow
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install other requirements
Write-Host "📚 Installing requirements..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install dev requirements
Write-Host "🛠️ Installing dev requirements..." -ForegroundColor Yellow
pip install -r requirements-dev.txt

# Create necessary directories
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
$dirs = @("checkpoints", "cache", "logs", "examples", "data", "runs")
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To verify your GPU setup, run:" -ForegroundColor Cyan
Write-Host "  python -c 'import torch; print(f`"CUDA available: {torch.cuda.is_available()}`")'"
