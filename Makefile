.PHONY: help install install-dev test lint format clean build upload

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install .

install-dev:  ## Install the package in development mode with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest

lint:  ## Run linting
	flake8 aviro tests
	mypy aviro

format:  ## Format code
	black aviro tests

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean  ## Build the package
	python -m build

upload: build  ## Upload to PyPI
	python -m twine upload dist/*

upload-test: build  ## Upload to Test PyPI
	python -m twine upload --repository testpypi dist/*
