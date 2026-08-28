.DEFAULT_GOAL := help
.DELETE_ON_ERROR:

-include .reporting.mk

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

PY ?= uv run python
UV_CACHE_DIR ?= /tmp/mammo-second-reader-uv-cache
ENSEMBLE_CONFIG ?= configs/ensemble.toml
PATCH_CONFIG ?= configs/patch_learning/vgg16_patch.toml
PATCH_DATA ?= results/patch_learning/data
PATCH_CHECKPOINT ?= models/patch_learning/vgg16_patch.pt
# Patch-transfer runs write their own evidence files. The frozen milestone
# metrics and predictions must not gain in-progress patch runs (§3 and §9).
PATCH_METRICS ?= results/patch_learning/metrics.json
PATCH_PREDICTIONS ?= results/patch_learning/predictions

# The reference arm runs first, so a failure there stops the comparison
# before the candidate consumes GPU hours.
TRANSFER_RUNS := \
	configs/patch_learning/vgg16_imagenet_448_quarantined.toml:42:vgg16_imagenet_448_quarantined \
	configs/patch_learning/vgg16_patch_imagenet_448.toml:42:vgg16_patch_imagenet_448
REPORT_TEMPLATE ?=
REPORT_SOURCE ?=
REPORT_UPDATE ?=
REPORT_FORCE ?=
PYTHON_PATHS := src tests
ARCHIVE_ROOT ?= ../mammo-second-reader-superseded/pipeline-$(shell date -u +%Y%m%dT%H%M%SZ)

CENTRAL_RUNS := \
	configs/vgg16_scratch.toml:42:vgg16_scratch \
	configs/vgg16_scratch.toml:7:vgg16_scratch_seed7 \
	configs/vgg16_scratch.toml:2026:vgg16_scratch_seed2026 \
	configs/vgg16_transfer.toml:42:vgg16_imagenet \
	configs/vgg16_transfer.toml:7:vgg16_imagenet_seed7 \
	configs/vgg16_transfer.toml:2026:vgg16_imagenet_seed2026 \
	configs/vgg16_highres_448.toml:42:vgg16_imagenet_448 \
	configs/vgg16_highres_448.toml:7:vgg16_imagenet_448_seed7 \
	configs/vgg16_highres_448.toml:2026:vgg16_imagenet_448_seed2026

EXPANDED_RUNS := \
	configs/vgg19_transfer.toml:42:vgg19_imagenet \
	configs/resnet50_transfer.toml:42:resnet50_imagenet \
	configs/efficientnet_b4.toml:42:efficientnet_b4_imagenet

HISTORICAL_RUNS := \
	configs/baseline.toml:42:baseline \
	configs/regularised_base.toml:42:regularised_base \
	configs/regularised_heavy_aug.toml:42:regularised_heavy_aug \
	configs/regularised_label_smooth.toml:42:regularised_label_smooth \
	configs/regularised_mixup.toml:42:regularised_mixup \
	configs/regularised_combined.toml:42:regularised_combined \
	configs/regularised_extensions/regularised_base_120.toml:42:regularised_base_120 \
	configs/regularised_extensions/regularised_label_smooth_120.toml:42:regularised_label_smooth_120 \
	configs/regularised_extensions/regularised_mixup_120.toml:42:regularised_mixup_120

REPORT_RUNS := $(CENTRAL_RUNS) $(EXPANDED_RUNS) $(HISTORICAL_RUNS)

export MPLBACKEND := Agg
export UV_CACHE_DIR

MODEL_ARGS = $(if $(strip $(SEED)),--seed "$(SEED)") $(if $(strip $(RUN_NAME)),--run-name "$(RUN_NAME)")

.PHONY: all help setup test lint format format-check typecheck check web splits \
	cache-224 cache-448 preprocess qa-preprocessing patch-data patch-qa patch-verify patch-train patch-transfer fixture \
	train evaluate experiments evaluate-experiments ensemble statistics figures freeze evidence \
	verify-evidence report-draft report-pack report-check submission-check leakage-audit \
	archive-evidence clean-evidence clean-cache clean-dev clean pipeline

all: pipeline ## Run the full pipeline.

help: ## List targets.
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make <target> [VARIABLE=value]\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install dependencies.
	uv sync --locked

test: ## Run tests.
	uv run pytest -q

lint: ## Run Ruff.
	uv run ruff check $(PYTHON_PATHS)

format: ## Format Python files.
	uv run ruff format $(PYTHON_PATHS)

format-check: ## Check Python formatting.
	uv run ruff format --check $(PYTHON_PATHS)

typecheck: ## Run mypy.
	uv run mypy src

check: ## Run code checks.
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) typecheck

web: ## Start the web app.
	$(PY) -m src.web.app

splits: ## Build data splits.
	$(PY) -m src.data.splits

cache-224: ## Build 224-pixel caches.
	$(PY) -m src.data.dicom_to_png
	$(PY) -m src.data.cache_roi_masks

cache-448: ## Build 448-pixel caches.
	$(PY) -m src.data.dicom_to_png --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448
	$(PY) -m src.data.cache_roi_masks --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448

preprocess: ## Build image caches.
	$(MAKE) cache-224
	$(MAKE) cache-448

qa-preprocessing: ## Check preprocessing outputs.
	$(PY) -m src.data.qa_preprocessing

patch-data: ## Build Stage 0 patch data.
	$(PY) -m src.data.patch_manifest --config configs/patch_learning/stage0.toml

patch-qa: ## Build Stage 0 review files.
	$(PY) -m src.data.patch_qa --config configs/patch_learning/stage0.toml

patch-verify: ## Verify the frozen Stage 0 patch tree.
	$(PY) -m src.data.patch_verify --data-root "$(PATCH_DATA)"

patch-train: ## Train the Stage 0 five-class patch classifier.
	@test -f "$(PATCH_DATA)/train.csv" || { \
		echo "Stage 0 patch data not found at $(PATCH_DATA)."; \
		echo "The 55,619-patch tree is ~11 GB and is not in version control."; \
		echo "Copy it to this host, then run 'make patch-verify'."; \
		exit 2; }
	@mkdir -p results/logs
	$(PY) -m src.training.train_patch --config "$(PATCH_CONFIG)" $(MODEL_ARGS) \
		2>&1 | tee "results/logs/$(if $(strip $(RUN_NAME)),$(RUN_NAME),vgg16_patch).patch-train.log"

patch-transfer: ## Train and evaluate both arms of the patch-transfer comparison.
	@test -f "$(PATCH_CHECKPOINT)" || { \
		echo "Patch checkpoint not found at $(PATCH_CHECKPOINT)."; \
		echo "Run 'make patch-train' first: the candidate arm initialises from it."; \
		echo "Checked up front so the reference arm does not train for nothing."; \
		exit 2; }
	@mkdir -p results/logs "$(PATCH_PREDICTIONS)"
	@for spec in $(TRANSFER_RUNS); do \
		config="$${spec%%:*}"; \
		remainder="$${spec#*:}"; \
		seed="$${remainder%%:*}"; \
		run_name="$${remainder#*:}"; \
		$(PY) -m src.training.train --config "$$config" \
			--seed "$$seed" --run-name "$$run_name" \
			2>&1 | tee "results/logs/$$run_name.train.log"; \
		$(PY) -m src.evaluation.evaluate --config "$$config" \
			--seed "$$seed" --run-name "$$run_name" \
			--metrics-path "$(PATCH_METRICS)" \
			--predictions-dir "$(PATCH_PREDICTIONS)" \
			2>&1 | tee "results/logs/$$run_name.evaluate.log"; \
	done

fixture: ## Build the fine-tuning fixture.
	$(PY) -m src.data.make_finetune_archive

train: ## Train one model.
	@if [ -z "$(strip $(CONFIG))" ]; then echo "CONFIG is required (for example, CONFIG=configs/vgg16_transfer.toml)"; exit 2; fi
	$(PY) -m src.training.train --config "$(CONFIG)" $(MODEL_ARGS)

evaluate: ## Evaluate one model.
	@if [ -z "$(strip $(CONFIG))" ]; then echo "CONFIG is required (for example, CONFIG=configs/vgg16_transfer.toml)"; exit 2; fi
	$(PY) -m src.evaluation.evaluate --config "$(CONFIG)" $(MODEL_ARGS)

experiments: ## Run all model experiments.
	@mkdir -p results/logs
	@for spec in $(REPORT_RUNS); do \
		config="$${spec%%:*}"; \
		remainder="$${spec#*:}"; \
		seed="$${remainder%%:*}"; \
		run_name="$${remainder#*:}"; \
		$(PY) -m src.training.train --config "$$config" \
			--seed "$$seed" --run-name "$$run_name" \
			2>&1 | tee "results/logs/$$run_name.train.log"; \
		$(PY) -m src.evaluation.evaluate --config "$$config" \
			--seed "$$seed" --run-name "$$run_name" \
			2>&1 | tee "results/logs/$$run_name.evaluate.log"; \
	done
	$(PY) -m src.training.ensemble --config "$(ENSEMBLE_CONFIG)" \
		2>&1 | tee results/logs/ensemble.evaluate.log

evaluate-experiments: ## Evaluate existing checkpoints.
	@mkdir -p results/logs
	@for spec in $(REPORT_RUNS); do \
		config="$${spec%%:*}"; \
		remainder="$${spec#*:}"; \
		seed="$${remainder%%:*}"; \
		run_name="$${remainder#*:}"; \
		$(PY) -m src.evaluation.evaluate --config "$$config" \
			--seed "$$seed" --run-name "$$run_name" \
			2>&1 | tee "results/logs/$$run_name.evaluate.log"; \
	done
	$(PY) -m src.training.ensemble --config "$(ENSEMBLE_CONFIG)" \
		2>&1 | tee results/logs/ensemble.evaluate.log

ensemble: ## Evaluate the ensemble.
	$(PY) -m src.training.ensemble --config "$(ENSEMBLE_CONFIG)"

statistics: ## Write statistics.
	$(PY) -m src.evaluation.statistics

figures: ## Write figures.
	$(PY) -m src.reporting.make_figures

freeze: ## Freeze results.
	$(PY) -m src.evaluation.freeze

evidence: ## Build and freeze results.
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze

verify-evidence: ## Verify saved results.
	$(PY) -m src.evaluation.verify_bundle

report-draft: ## Create the report if absent.
	@test -n "$(strip $(REPORT_TEMPLATE))" || { echo "REPORT_TEMPLATE is required"; exit 2; }
	@test -n "$(strip $(REPORT_SOURCE))" || { echo "REPORT_SOURCE is required"; exit 2; }
	$(PY) -m src.reporting.report_draft --template "$(REPORT_TEMPLATE)" --output "$(REPORT_SOURCE)" $(if $(strip $(REPORT_FORCE)),--force)

report-pack: report-draft ## Build report update notes.
	@test -n "$(strip $(REPORT_UPDATE))" || { echo "REPORT_UPDATE is required"; exit 2; }
	$(PY) -m src.reporting.report_pack --report-source "$(REPORT_SOURCE)" --output "$(REPORT_UPDATE)"

report-check: report-draft ## Check the report.
	@test -n "$(strip $(REPORT_UPDATE))" || { echo "REPORT_UPDATE is required"; exit 2; }
	$(PY) -m src.reporting.report_pack --report-source "$(REPORT_SOURCE)" --output "$(REPORT_UPDATE)" --fail-on-stale

submission-check: ## Run submission checks.
	$(MAKE) check
	$(MAKE) verify-evidence
	@if [ -n "$(strip $(REPORT_SOURCE))" ]; then $(MAKE) report-check; fi

leakage-audit: ## Run the split audit.
	$(PY) -m src.evaluation.leakage_sensitivity

archive-evidence: ## Archive current results.
	@test ! -e "$(ARCHIVE_ROOT)" || { echo "Archive already exists: $(ARCHIVE_ROOT)"; exit 2; }
	@mkdir -p "$(ARCHIVE_ROOT)/results"
	@if [ -d models ]; then mv models "$(ARCHIVE_ROOT)/models"; fi
	@mkdir -p models
	@for name in metrics.json statistics.json evidence-freeze.json; do \
		if [ -f "results/$$name" ]; then mv "results/$$name" "$(ARCHIVE_ROOT)/results/"; fi; \
	done
	@for directory in predictions figures logs qa_preprocessing; do \
		if [ -d "results/$$directory" ]; then \
			mv "results/$$directory" "$(ARCHIVE_ROOT)/results/$$directory"; \
		fi; \
		if [ "$$directory" != qa_preprocessing ]; then mkdir -p "results/$$directory"; fi; \
	done
	@echo "Archived active internal evidence to $(ARCHIVE_ROOT)"

clean-evidence: ## Delete generated evidence and models without making a backup.
	rm -rf models
	mkdir -p models
	rm -f results/metrics.json results/statistics.json results/evidence-freeze.json
	rm -rf results/predictions results/figures results/logs results/qa_preprocessing
	mkdir -p results/predictions results/figures results/logs

clean-cache: ## Remove image caches.
	@if [ -d data/cbis-ddsm/cbis_ddsm ]; then \
		find data/cbis-ddsm/cbis_ddsm -type f -name '*.npy' -delete; \
	fi
	rm -rf data/cbis-ddsm/cache_448

clean-dev: ## Remove development caches.
	find src tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache

clean: ## Delete generated evidence, models, and caches without making a backup.
	$(MAKE) clean-evidence
	$(MAKE) clean-cache
	$(MAKE) clean-dev

pipeline: ## Run the pipeline from a clean state.
	$(MAKE) setup
	$(MAKE) check
	$(MAKE) clean
	$(MAKE) preprocess
	$(MAKE) qa-preprocessing
	$(MAKE) experiments
	$(MAKE) evidence
	$(MAKE) verify-evidence
	@if [ -n "$(strip $(REPORT_SOURCE))" ]; then $(MAKE) report-pack; fi
