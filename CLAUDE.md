# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands

- Install dependencies: `uv pip install -r pyproject.toml`
- Run pipeline: `python run_pipeline.py`
- Run all tests: `pytest tests/`
- Run single test: `pytest tests/path/to/test_file.py::test_function_name -v`
- Run tests with asyncio: `pytest tests/ --asyncio-mode=auto`
- Linting: `ruff check .`
- Formatting: `ruff format .`

## Code Style Guidelines

- **Formatting**: Line length 88, double quotes for strings
- **Imports**: Group imports: standard library, third-party, local; sort alphabetically
- **Types**: Use type hints for all function parameters and return values
- **Naming**: snake_case for variables/functions, PascalCase for classes
- **Async/Await**: Use asyncio for I/O bound operations
- **Error Handling**: Use specific exceptions, log errors with logger
- **Constants**: UPPER_CASE for constants
- **Redis Keys**: Follow established patterns for Redis keys (e.g., `paper:{arxiv_id}`)
- **Testing**: Use pytest fixtures, test async code with pytest_asyncio