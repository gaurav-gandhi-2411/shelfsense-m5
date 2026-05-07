.PHONY: setup ui ui-down data train submit test lint format report clean help

# Default target
.DEFAULT_GOAL := help

setup: ## Build Docker image and prime persistent directories
	mkdir -p mlruns dagster_home data/raw data/processed/features
	docker compose build train
	@echo "Setup complete. Run make ui to start MLflow + Dagster."

ui: ## Start MLflow (:5000) and Dagster (:3000) UIs
	docker compose up -d
	@echo "MLflow:   http://localhost:5000"
	@echo "Dagster:  http://localhost:3000"

ui-down: ## Stop and remove UI containers
	docker compose down

data: ## Download M5 data and build feature parquets (NotImplementedError until Stage 4)
	docker compose run --rm train data download
	docker compose run --rm train features build

train: ## Run production model training (NotImplementedError until Stage 4)
	docker compose run --rm train train tweedie-mh

submit: ## Generate ensemble submission and push to Kaggle (NotImplementedError until Stage 4)
	docker compose run --rm train ensemble --candidates tvp_13,store_dept --method optuna
	docker compose run --rm train submit --variant best --kaggle

test: ## Run pytest with coverage report (host venv)
	pytest --cov=shelfsense --cov-report=term-missing tests/

lint: ## Check formatting and types (host venv)
	ruff check shelfsense/ tests/
	black --check shelfsense/ tests/
	mypy shelfsense/

format: ## Apply ruff + black formatting fixes in-place
	ruff check --fix shelfsense/ tests/
	black shelfsense/ tests/

report: ## Regenerate leaderboard and portfolio charts (NotImplementedError until Stage 6)
	docker compose run --rm train report --regenerate-charts

clean: ## Remove build artefacts and cache dirs (preserves data/ and mlruns/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

help: ## List all available targets
	@grep -E '^[a-zA-Z_-]+:.*##' Makefile | sed 's/:.*##/ —/' | sort
