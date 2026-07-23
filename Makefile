.PHONY: install install-dev lint format test test-all smoke train evaluate quantize serve export-onnx clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,serve,onnx]"

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest -m "not slow"

test-all:
	pytest --cov=emocapnet --cov-report=term-missing

smoke:
	emocapnet smoke --out runs/smoke --epochs 1

train:
	emocapnet train --config configs/default.yaml

evaluate:
	emocapnet evaluate --config configs/default.yaml --controllability

quantize:
	emocapnet quantize --config configs/default.yaml

serve:
	emocapnet serve --config configs/default.yaml --int8

export-onnx:
	emocapnet export-onnx --config configs/default.yaml --out onnx

clean:
	rm -rf runs .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
