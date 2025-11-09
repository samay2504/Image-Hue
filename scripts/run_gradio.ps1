# Run Gradio app (Windows PowerShell)

Write-Host "🚀 Starting Gradio app..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".venv") {
    .\.venv\Scripts\Activate.ps1
}

# Run Gradio
python src/ui/gradio_app.py --port 7860
