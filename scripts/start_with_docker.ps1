# Docker build and run script (Windows PowerShell)

Write-Host "🐳 Building and running Colorful Image Colorization with Docker..." -ForegroundColor Cyan

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker not found. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

# Check Docker Compose
if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ docker-compose not found. Please install docker-compose first." -ForegroundColor Red
    exit 1
}

# Check NVIDIA Docker support (optional)
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    try {
        docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi 2>&1 | Out-Null
        Write-Host "✅ NVIDIA Docker support detected" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  NVIDIA Container Toolkit not configured. GPU support disabled." -ForegroundColor Yellow
        Write-Host "Install from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    }
}

# Build and start services
Write-Host "🔨 Building Docker images..." -ForegroundColor Yellow
Set-Location docker
docker-compose build

Write-Host "🚀 Starting services..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "✅ Services started!" -ForegroundColor Green
Write-Host ""
Write-Host "Access the applications:" -ForegroundColor Cyan
Write-Host "  📊 Streamlit UI:  http://localhost:8501"
Write-Host "  🎨 Gradio UI:     http://localhost:7860"
Write-Host "  🗄️  Redis:         localhost:6379"
Write-Host ""
Write-Host "View logs:" -ForegroundColor Cyan
Write-Host "  docker-compose logs -f"
Write-Host ""
Write-Host "Stop services:" -ForegroundColor Cyan
Write-Host "  docker-compose down"
