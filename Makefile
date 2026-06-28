# FleetCast — developer & pipeline entry points.
# Local stages run via `uv run`; `up`/`down` drive the Docker Compose stack.

.DEFAULT_GOAL := help
SHELL := /bin/bash

# Offline by default so `make test` is deterministic and network-free.
# Use `make test-all` to additionally run the @network integration tests.
export FLEETCAST_OFFLINE ?= 1

.PHONY: help install lock data smoke features backtest train eval demo \
        test test-all lint format up down build logs clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install pinned deps (uv sync)
	uv sync

lock: ## Refresh the uv lockfile
	uv lock

# ----------------------------------------------------------------------------
# Pipeline stages (each maps to a `fleetcast` subcommand)
# ----------------------------------------------------------------------------
data: ## Download raw TLC/weather data + DuckDB zone-hour smoke test
	uv run fleetcast data

smoke: ## DuckDB zone-hour smoke test only (data must already be downloaded)
	uv run fleetcast smoke

features: ## Build the dense zone-hour panel + causal features (DuckDB SQL)
	uv run fleetcast features

backtest: ## Run the rolling-origin backtest (baselines)
	uv run fleetcast backtest

train: ## Train the global LightGBM model through the backtest folds
	uv run fleetcast train

eval: ## Print the eval table and enforce the WAPE/coverage gates
	uv run fleetcast eval

demo: ## Launch the Streamlit choropleth dashboard locally
	uv run fleetcast demo

# ----------------------------------------------------------------------------
# Quality
# ----------------------------------------------------------------------------
test: ## Run the offline test suite (network tests skipped)
	uv run pytest

test-all: ## Run the full suite including @network integration tests
	FLEETCAST_OFFLINE=0 uv run pytest

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

# ----------------------------------------------------------------------------
# Docker Compose stack
# ----------------------------------------------------------------------------
build: ## Build the Docker image
	docker compose build

up: ## Bring up the stack (pipeline runs, then the Streamlit app serves)
	docker compose up --build

down: ## Stop the stack and remove containers
	docker compose down

logs: ## Tail Compose logs
	docker compose logs -f

clean: ## Remove generated processed/forecast artifacts (keeps raw downloads)
	rm -rf data/processed/* data/forecasts/* .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
