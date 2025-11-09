# Run Streamlit app (Windows PowerShell)

Write-Host "🚀 Starting Streamlit app..." -ForegroundColor Cyan

# Activate virtual environment if it exists
if (Test-Path ".venv") {
    .\.venv\Scripts\Activate.ps1
}

# Run Streamlit
python -m streamlit run src/ui/streamlit_app.py `
    --server.address 0.0.0.0 `
    --server.port 8501 `
    --server.headless true
