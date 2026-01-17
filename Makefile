# Makefile for DFS Professional Implementation
# Industry-Grade Build and Test Automation

.PHONY: help install test test-verbose test-coverage test-benchmark lint format type-check clean docs run

# Default target
help:
	@echo "Professional DFS Implementation - Build System"
	@echo "==============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install development dependencies"
	@echo "  test          Run unit tests with coverage"
	@echo "  test-verbose  Run tests with verbose output"
	@echo "  test-coverage Run tests with detailed coverage report"
	@echo "  test-benchmark Run performance benchmarks"
	@echo "  lint          Run code quality checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run static type checking with mypy"
	@echo "  clean         Clean build artifacts"
	@echo "  docs          Generate documentation"
	@echo "  run           Run the DFS implementation"
	@echo "  all           Run all quality checks and tests"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

# Testing
test:
	pytest test_dfs_professional.py --cov=DFS_Lab_Manual --cov-report=term-missing --cov-fail-under=96.7

test-verbose:
	pytest test_dfs_professional.py -v --cov=DFS_Lab_Manual --cov-report=term-missing

test-coverage:
	pytest test_dfs_professional.py --cov=DFS_Lab_Manual --cov-report=html --cov-report=xml --cov-report=term-missing

test-benchmark:
	pytest test_dfs_professional.py --benchmark-only --benchmark-sort=mean

test-integration:
	pytest test_dfs_professional.py -m integration -v

test-unit:
	pytest test_dfs_professional.py -m unit -v

# Code Quality
lint:
	flake8 DFS_Lab_Manual.py test_dfs_professional.py --max-line-length=88 --extend-ignore=E203,W503
	bandit -r DFS_Lab_Manual.py

format:
	black DFS_Lab_Manual.py test_dfs_professional.py
	isort DFS_Lab_Manual.py test_dfs_professional.py

type-check:
	mypy DFS_Lab_Manual.py test_dfs_professional.py --ignore-missing-imports

# Documentation
docs:
	sphinx-build -b html docs/ docs/_build/html
	@echo "Documentation available at docs/_build/html/index.html"

# Utility
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +

run:
	python DFS_Lab_Manual.py

# Quality Gates
quality-check: lint type-check
	@echo "Quality checks passed!"

# Full test suite
all: format lint type-check test-coverage test-benchmark
	@echo "All checks completed successfully!"

# Development setup
dev-setup: install
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify installation."

# Continuous integration target
ci: install lint type-check test-coverage
	@echo "CI pipeline completed successfully!"

# Performance profiling
profile:
	python -m cProfile -o profile_output.prof DFS_Lab_Manual.py
	@echo "Profile saved to profile_output.prof"
	@echo "Use 'python -m pstats profile_output.prof' to analyze"

# Security audit
security:
	bandit -r DFS_Lab_Manual.py -f json -o security_report.json
	@echo "Security report saved to security_report.json"

# Dependency check
deps-check:
	pip-audit --requirement requirements.txt
	@echo "Dependency audit completed"
