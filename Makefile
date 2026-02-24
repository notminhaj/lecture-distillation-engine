.PHONY: install lint test run clean

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

test:
	pytest tests/ -v --cov=distillation --cov-report=term-missing

# Quick smoke-test on a short clip
run-demo:
	python scripts/run_pipeline.py --input examples/sample.mp4 --domain khutba

clean:
	rm -rf outputs/ cache/ .pytest_cache/ dist/ *.egg-info/
	find . -name "__pycache__" -type d -exec rm -rf {} +
