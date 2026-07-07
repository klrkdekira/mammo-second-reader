PY ?= uv run python3

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

data: splits cache

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

train-resnet18-transfer:
	$(PY) -m src.training.train --config configs/resnet18_transfer.toml

train-densenet121-transfer:
	$(PY) -m src.training.train --config configs/densenet121_transfer.toml

train-efficientnet_b0-transfer:
	$(PY) -m src.training.train --config configs/efficientnet_b0_transfer.toml

train-mobilenet_v3-transfer:
	$(PY) -m src.training.train --config configs/mobilenet_v3_transfer.toml

train-convnext_tiny-transfer:
	$(PY) -m src.training.train --config configs/convnext_tiny_transfer.toml

# All transfer learning
train-transfer: train-vgg16-transfer train-vgg19-transfer train-resnet18-transfer train-resnet50-transfer train-densenet121-transfer train-efficientnet_b0-transfer train-efficientnet_b4-transfer train-mobilenet_v3-transfer train-convnext_tiny-transfer

train: train-baseline train-regularisation train-vgg16-scratch train-transfer

# Evaluation
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
	$(PY) -m src.evaluation.evaluate --config configs/resnet18_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/resnet50_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/densenet121_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/efficientnet_b0_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/efficientnet_b4.toml
	$(PY) -m src.evaluation.evaluate --config configs/mobilenet_v3_transfer.toml
	$(PY) -m src.evaluation.evaluate --config configs/convnext_tiny_transfer.toml

figures:
	$(PY) -m src.reporting.make_figures

results: clean-results evaluate figures

# Clean up the training data cache.
clean:
	rm -rf data/cbis-ddsm/training

pipeline: data train results
