PY ?= uv run python3

sync:
	uv sync

webapp:
	$(PY) -m src.web.app

# Build train/validation/test splits
splits:
	$(PY) -m src.data.splits

# Preprocess DICOM images to PNG and cache them for faster training.
cache:
	$(PY) -m src.data.dicom_to_png

clean-cache:
	find ./data/cbis-ddsm/ -type f -name "*.npy" -delete

data: splits cache

# Training
train-baseline:
	$(PY) -m src.training.train --config configs/baseline.toml

train-regularised:
	$(PY) -m src.training.train --config configs/regularised.toml

train-vgg16-scratch:
	$(PY) -m src.training.train --config configs/vgg16_scratch.toml

train-vgg16-transfer:
	$(PY) -m src.training.train --config configs/vgg16_transfer.toml

train: train-baseline train-regularised train-vgg16-scratch train-vgg16-transfer

# Evaluation
clean-results:
	rm -rf results/metrics.json
	rm -rf results/figures/*.png

evaluate:
	$(PY) -m src.evaluation.evaluate --config configs/baseline.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised.toml
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_scratch.toml
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_transfer.toml

figures:
	$(PY) -m src.reporting.make_figures

results: clean-results evaluate figures

# Clean up the training data cache.
clean:
	rm -rf data/cbis-ddsm/training

pipeline: data train results
