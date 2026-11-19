.PHONY: install test lint format notebooks clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check Code tests

format:
	$(PYTHON) -m ruff format Code tests

notebooks:
	$(PYTHON) -m jupyter notebook --notebook-dir=Code

clean:
	$(PYTHON) -m pytest --cache-clear
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
