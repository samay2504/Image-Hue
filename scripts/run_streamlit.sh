#!/bin/bash
# Run Streamlit app

set -e

echo "🚀 Starting Streamlit app..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Streamlit
python -m streamlit run src/ui/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.headless true

