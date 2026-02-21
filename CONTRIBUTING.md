# Contributing

Thank you for your interest in contributing to Time Series ML Demonstration! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/<your-username>/time.git
   cd time
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements-dev.txt
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
5. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Standards

- **Formatting:** [Black](https://github.com/psf/black) with 120-character line length
- **Import sorting:** [isort](https://pycqa.github.io/isort/) with Black-compatible profile
- **Linting:** [Flake8](https://flake8.pycqa.org/) with project `.flake8` config
- **Type hints:** Encouraged but not required; [mypy](http://mypy-lang.org/) runs in relaxed mode
- **Style:** Follow PEP 8 conventions

Pre-commit hooks enforce formatting and linting automatically on each commit.

## Making Changes

1. Write clear, focused commits with descriptive messages
2. Add tests for new functionality in the `tests/` directory
3. Ensure all tests pass: `pytest`
4. Run the full linting suite: `pre-commit run --all-files`

## Pull Request Process

1. Update documentation if your changes affect the public API or pipeline behavior
2. Ensure CI checks pass on your PR
3. Provide a clear description of your changes and the motivation behind them
4. Link any related issues

## Reporting Issues

- **Bugs:** Use the [bug report template](https://github.com/stbiadmin/time/issues/new?template=bug_report.md)
- **Feature requests:** Use the [feature request template](https://github.com/stbiadmin/time/issues/new?template=feature_request.md)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.
