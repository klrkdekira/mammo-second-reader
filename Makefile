PY ?= uv run python3

.PHONY: all sync webapp splits cache cache-roi clean-cache data train train-baseline train-regularised-base train-regularised-heavy-aug train-regularised-label-smooth train-regularised-mixup train-regularised-combined train-regularisation train-vgg16-scratch train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer train-transfer clean-results evaluate figures results clean-models clean pipeline

# Default target: clean everything and run the entire pipeline from scratch
all: clean sync pipeline

sync:
	uv sync

webapp:
	$(PY) -m src.web.app

# Build train/validation/test splits
splits:
	$(PY) -m src.data.splits

# Preprocess DICOM images to PNG
cache:
	$(PY) -m src.data.dicom_to_png

# Pre-crop ROI lesion masks
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

# All regularisation ablations
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

# All transfer learning
train-transfer: train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer

train: train-baseline train-regularisation train-vgg16-scratch train-transfer

# Clean up evaluation results
clean-results:
	rm -rf results/metrics.json
	rm -rf results/figures/*.png

evaluate:
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

figures:
	$(PY) -m src.reporting.make_figures

results: clean-results evaluate figures

# Clean up trained model checkpoints and history
clean-models:
	rm -f models/*.pt models/*.json

# Perform a full clean of cached data, trained models, evaluation results, and python caches
clean: clean-cache clean-models clean-results
	rm -rf data/cbis-ddsm/training
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

pipeline: data train results
