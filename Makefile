PHONY: install test lint format

install:
	pip install -e .[dev,ui]

test:
	pytest -q

lint:
	ruff check .

format:
	black .
