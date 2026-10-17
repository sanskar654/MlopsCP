# =============================================================================
# Vulnerability Prioritization MLOps Platform - Operations Makefile
# =============================================================================

.PHONY: help setup train test lint up down restart logs clean traffic dvc-repro

PYTHON = python
DOCKER_COMPOSE = docker compose

help:
	@echo "Available commands:"
	@echo "  make setup       Install dependencies"
	@echo "  make train       Run full end-to-end training pipeline on real data"
	@echo "  make test        Execute all unit and integration tests"
	@echo "  make lint        Check formatting and code quality (strict)"
	@echo "  make up          Start full containerized production stack"
	@echo "  make down        Stop all containers and services"
	@echo "  make restart     Restart all services"
	@echo "  make logs        Tail service container logs"
	@echo "  make traffic     Run API benchmark & load simulation"
	@echo "  make dvc-repro   Reproduce DVC pipeline stages"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-dev.txt

train:
	$(PYTHON) scripts/run_pipeline.py

test:
	$(PYTHON) -m pytest tests/unit tests/integration -v --cov=src

lint:
	black --check src tests scripts
	flake8 src tests scripts --max-line-length=120 --ignore=E501,W503

up:
	$(DOCKER_COMPOSE) up -d --build

down:
	$(DOCKER_COMPOSE) down

restart:
	$(DOCKER_COMPOSE) restart

logs:
	$(DOCKER_COMPOSE) logs -f

traffic:
	$(PYTHON) scripts/simulate_traffic.py --requests 300 --concurrency 10

dvc-repro:
	dvc repro
