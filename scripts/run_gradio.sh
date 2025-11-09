#!/bin/bash
# Run Gradio app

set -e

echo "🚀 Starting Gradio app..."

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Gradio
python src/ui/gradio_app.py --port 7860
