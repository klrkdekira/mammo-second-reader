PY ?= uv run python3

# Figures are written to disk, never displayed. Without this matplotlib picks
# TkAgg and dies on a headless host with "couldn't connect to display".
export MPLBACKEND := Agg

.PHONY: all sync webapp splits cache cache-roi clean-cache data train train-baseline train-regularised-base train-regularised-heavy-aug train-regularised-label-smooth train-regularised-mixup train-regularised-combined train-regularisation train-vgg16-scratch train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer train-transfer train-vgg16-seed-study evaluate-vgg16-seed-study clean-results evaluate evaluate-existing evaluate-ensemble statistics figures freeze-evidence reproduce-existing results clean-models clean pipeline

# Default target
all: clean sync pipeline

sync:
	uv sync

webapp:
	$(PY) -m src.web.app

# Data preprocessing
splits:
	$(PY) -m src.data.splits

cache:
	$(PY) -m src.data.dicom_to_png

cache-roi:
	$(PY) -m src.data.cache_roi_masks

clean-cache:
	find ./data/cbis-ddsm/ -type f -name "*.npy" -delete

data: splits cache cache-roi

# Training
train-baseline:
	$(PY) -m src.training.train --config configs/baseline.toml

train-regularised-base:
	$(PY) -m src.training.train --config configs/regularised_base.toml

train-regularised-heavy-aug:
	$(PY) -m src.training.train --config configs/regularised_heavy_aug.toml

train-regularised-label-smooth:
	$(PY) -m src.training.train --config configs/regularised_label_smooth.toml

train-regularised-mixup:
	$(PY) -m src.training.train --config configs/regularised_mixup.toml

train-regularised-combined:
	$(PY) -m src.training.train --config configs/regularised_combined.toml

# Regularisation ablations
train-regularisation: train-regularised-base train-regularised-heavy-aug train-regularised-label-smooth train-regularised-mixup train-regularised-combined

train-vgg16-scratch:
	$(PY) -m src.training.train --config configs/vgg16_scratch.toml

train-vgg16-transfer:
	$(PY) -m src.training.train --config configs/vgg16_transfer.toml

train-vgg19-transfer:
	$(PY) -m src.training.train --config configs/vgg19_transfer.toml

train-resnet50-transfer:
	$(PY) -m src.training.train --config configs/resnet50_transfer.toml

train-efficientnet_b4-transfer:
	$(PY) -m src.training.train --config configs/efficientnet_b4.toml

# Transfer learning
train-transfer: train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer

# Seed 42 is the existing run. These two repeats complete a three-seed VGG-16 study.
train-vgg16-seed-study:
	$(PY) -m src.training.train --config configs/vgg16_scratch.toml --seed 7 --run-name vgg16_scratch_seed7
	$(PY) -m src.training.train --config configs/vgg16_transfer.toml --seed 7 --run-name vgg16_imagenet_seed7
	$(PY) -m src.training.train --config configs/vgg16_scratch.toml --seed 2026 --run-name vgg16_scratch_seed2026
	$(PY) -m src.training.train --config configs/vgg16_transfer.toml --seed 2026 --run-name vgg16_imagenet_seed2026

evaluate-vgg16-seed-study:
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_scratch.toml --seed 7 --run-name vgg16_scratch_seed7
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_transfer.toml --seed 7 --run-name vgg16_imagenet_seed7
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_scratch.toml --seed 2026 --run-name vgg16_scratch_seed2026
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_transfer.toml --seed 2026 --run-name vgg16_imagenet_seed2026

train: train-baseline train-regularisation train-vgg16-scratch train-transfer

# Evaluation & figures
clean-results:
	rm -f results/metrics.json results/statistics.json results/evidence-freeze.json
	rm -rf results/predictions
	rm -f results/figures/*.png

evaluate: evaluate-ensemble
	$(PY) -m src.evaluation.evaluate --config configs/baseline.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_base.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_heavy_aug.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_label_smooth.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_mixup.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_combined.toml
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_scratch.toml
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/vgg19_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/resnet50_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/efficientnet_b4.toml

# Ensemble model evaluation
evaluate-ensemble:
	$(PY) -m src.training.ensemble --config configs/ensemble.toml

statistics:
	$(PY) -m src.evaluation.statistics

figures:
	$(PY) -m src.reporting.make_figures

# Recheck existing models, rebuild the figures, and freeze the results.
evaluate-existing: evaluate

freeze-evidence:
	$(PY) -m src.evaluation.freeze

reproduce-existing:
	$(MAKE) evaluate-existing
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze-evidence

results:
	$(MAKE) clean-results
	$(MAKE) evaluate
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze-evidence

# Cleanup
clean-models:
	rm -f models/*.pt models/*.json

clean: clean-cache clean-models clean-results
	rm -rf data/cbis-ddsm/training
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

pipeline: data train results
