.PHONY: help install test lint format pipeline dashboard docker-up docker-down clean

PYTHON ?= python
PIP ?= pip

help:
	@echo "Available commands:"
	@echo "  make install     Install production & dev dependencies"
	@echo "  make test        Run pytest test suite"
	@echo "  make lint        Check code quality with ruff"
	@echo "  make format      Auto-format code with ruff"
	@echo "  make pipeline    Run end-to-end data pipeline (ingest -> train -> predict -> evaluate -> plots)"
	@echo "  make dashboard   Launch interactive Streamlit dashboard"
	@echo "  make docker-up   Start PostgreSQL/TimescaleDB container"
	@echo "  make docker-down Stop container services"
	@echo "  make clean       Remove caches and temporary artifacts"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -v

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

pipeline:
	$(PYTHON) scripts/run_pipeline.py --mode full

dashboard:
	$(PYTHON) -m streamlit run src/dashboard/app.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
