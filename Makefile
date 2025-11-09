# Colorization Project Makefile

.PHONY: help setup install test lint format clean docker-build docker-up docker-down train infer streamlit gradio

help:
	@echo "Available commands:"
	@echo "  make setup          - Set up development environment"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code with black"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start Docker services"
	@echo "  make docker-down    - Stop Docker services"
	@echo "  make streamlit      - Run Streamlit UI"
	@echo "  make gradio         - Run Gradio UI"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
	. .venv/bin/activate && pip install -r requirements.txt
	. .venv/bin/activate && pip install -r requirements-dev.txt

install:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

test:
	pytest src/tests/ -v

test-coverage:
	pytest src/tests/ -v --cov=src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 src/
	black --check src/
	isort --check-only src/

format:
	black src/
	isort src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -rf cache/*.pkl

docker-build:
	cd docker && docker-compose build

docker-up:
	cd docker && docker-compose up -d

docker-down:
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

streamlit:
	python -m streamlit run src/ui/streamlit_app.py --server.port 8501

gradio:
	python src/ui/gradio_app.py --port 7860

train-quick:
	python -m src.train --config configs/quicktrain.yaml --train_dir data/train --val_dir data/val

train-full:
	python -m src.train --config configs/fulltrain.yaml --train_dir data/train --val_dir data/val

infer:
	python -m src.infer examples/sample.jpg --output output.jpg --method classification

verify:
	./scripts/verify_system.sh
