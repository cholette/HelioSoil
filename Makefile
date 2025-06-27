# Makefile for the HelioSoil project

# Use PowerShell on Windows, otherwise default to sh
ifeq ($(OS),Windows_NT)
	SHELL := pwsh.exe
	CLEAN_CMD = Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -Path build, dist, *.egg-info, .pytest_cache; Get-ChildItem -Path . -Include __pycache__ -Recurse | Remove-Item -Recurse -Force
else
	SHELL := /bin/sh
	CLEAN_CMD = rm -rf build dist *.egg-info .pytest_cache; find . -type d -name "__pycache__" -exec rm -rf {} +
endif

# Phony targets don't represent actual files
.PHONY: help install test format hooks clean

help:
	@echo "Available commands:"
	@echo "  install   - Install the package in editable mode with all dev dependencies."
	@echo "  test      - Run all tests with pytest."
	@echo "  format    - Format code with black and check style with flake8."
	@echo "  clean     - Remove build artifacts and pycache files."

install:
	@echo "--> Installing package in editable mode with dev dependencies..."
	pip install -e .[dev]
	@echo "--> Installing pre-commit hooks..."
	pre-commit install

test:
	@echo "--> Running tests..."
	pytest

format:
	@echo "--> Formatting code..."
	black .
	flake8 .

clean:
	@echo "--> Cleaning up build artifacts and pycache..."
	$(CLEAN_CMD)
