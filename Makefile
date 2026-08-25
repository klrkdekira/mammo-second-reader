.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

PY ?= uv run python
UV_CACHE_DIR ?= /tmp/mammo-second-reader-uv-cache
ENSEMBLE_CONFIG ?= configs/ensemble.toml

export MPLBACKEND := Agg
export UV_CACHE_DIR

MODEL_ARGS = $(if $(strip $(SEED)),--seed "$(SEED)") $(if $(strip $(RUN_NAME)),--run-name "$(RUN_NAME)")

.PHONY: help setup test lint format format-check typecheck check web splits \
	cache-224 cache-448 preprocess qa-preprocessing patch-data patch-qa fixture \
	train evaluate ensemble statistics figures freeze evidence leakage-audit

help: ## Show the supported workflow.
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [VARIABLE=value]\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install the locked project environment.
	uv sync --locked

test: ## Run the test suite.
	uv run pytest -q

lint: ## Run Ruff diagnostics.
	uv run ruff check .

format: ## Format Python files with Ruff.
	uv run ruff format .

format-check: ## Check formatting without changing files.
	uv run ruff format --check .

typecheck: ## Type-check the source tree.
	uv run mypy src

check: ## Run tests, lint, formatting checks and type checks.
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

web: ## Launch the research web application.
	$(PY) -m src.web.app

splits: ## Build the canonical CBIS-DDSM train/validation/test manifests.
	$(PY) -m src.data.splits

cache-224: ## Build aligned 224-pixel image and ROI caches.
	$(PY) -m src.data.dicom_to_png
	$(PY) -m src.data.cache_roi_masks

cache-448: ## Build aligned 448-pixel image and ROI caches.
	$(PY) -m src.data.dicom_to_png --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448
	$(PY) -m src.data.cache_roi_masks --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448

preprocess: ## Build both cache resolutions from the frozen canonical manifests.
	$(MAKE) cache-224
	$(MAKE) cache-448

qa-preprocessing: ## Audit segmentation, artefacts and ROI crop coverage.
	$(PY) -m src.data.qa_preprocessing

patch-data: ## Build deterministic Stage 0 patch data.
	$(PY) -m src.data.patch_manifest --config configs/patch_learning/stage0.toml

patch-qa: ## Build the locked Stage 0 manual-review package.
	$(PY) -m src.data.patch_qa --config configs/patch_learning/stage0.toml

fixture: ## Build the bounded web fine-tuning fixture.
	$(PY) -m src.data.make_finetune_archive

train: ## Train one model: make train CONFIG=path.toml [SEED=n RUN_NAME=name].
	@if [ -z "$(strip $(CONFIG))" ]; then echo "CONFIG is required (for example, CONFIG=configs/vgg16_transfer.toml)"; exit 2; fi
	$(PY) -m src.training.train --config "$(CONFIG)" $(MODEL_ARGS)

evaluate: ## Evaluate one model: make evaluate CONFIG=path.toml [SEED=n RUN_NAME=name].
	@if [ -z "$(strip $(CONFIG))" ]; then echo "CONFIG is required (for example, CONFIG=configs/vgg16_transfer.toml)"; exit 2; fi
	$(PY) -m src.evaluation.evaluate --config "$(CONFIG)" $(MODEL_ARGS)

ensemble: ## Evaluate the configured probability ensemble.
	$(PY) -m src.training.ensemble --config "$(ENSEMBLE_CONFIG)"

statistics: ## Generate patient-level intervals and paired comparisons.
	$(PY) -m src.evaluation.statistics

figures: ## Regenerate report figures from canonical evidence.
	$(PY) -m src.reporting.make_figures

freeze: ## Validate and freeze the canonical evidence bundle.
	$(PY) -m src.evaluation.freeze

evidence: ## Generate statistics and figures, then freeze the evidence.
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze

leakage-audit: ## Recompute the disclosed post-hoc leakage sensitivity analysis.
	$(PY) -m src.evaluation.leakage_sensitivity
