PY ?= uv run python3

.PHONY: all sync webapp splits cache cache-roi clean-cache data train train-baseline train-regularised-base train-regularised-heavy-aug train-regularised-label-smooth train-regularised-mixup train-regularised-combined train-regularisation train-vgg16-scratch train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer train-transfer clean-results evaluate evaluate-ensemble figures results clean-models clean pipeline

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

train: train-baseline train-regularisation train-vgg16-scratch train-transfer

# Evaluation & figures
clean-results:
	rm -rf results/metrics.json
	rm -rf results/figures/*.png

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

figures:
	$(PY) -m src.reporting.make_figures

results: clean-results evaluate figures

# Cleanup
clean-models:
	rm -f models/*.pt models/*.json

clean: clean-cache clean-models clean-results
	rm -rf data/cbis-ddsm/training
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

pipeline: data train results
