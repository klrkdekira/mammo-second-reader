PY ?= uv run python3

# Figures are written to disk, never displayed. Without this matplotlib picks
# TkAgg and dies on a headless host with "couldn't connect to display".
export MPLBACKEND := Agg

.PHONY: all sync webapp splits cache cache-roi cache-highres patch-data patch-qa finetune-fixture inbreast-manifest inbreast-cache evaluate-inbreast-cold reproduce-inbreast-cold clean-cache data train train-baseline train-regularised-base train-regularised-heavy-aug train-regularised-label-smooth train-regularised-mixup train-regularised-combined train-regularisation train-regularised-extensions train-vgg16-scratch train-vgg16-transfer train-vgg19-transfer train-resnet50-transfer train-efficientnet_b4-transfer train-transfer train-vgg16-seed-study train-vgg16-highres train-vgg16-highres-seed-study evaluate-regularised-extensions evaluate-vgg16-seed-study evaluate-vgg16-highres evaluate-vgg16-highres-seed-study clean-results evaluate evaluate-existing evaluate-ensemble statistics figures freeze-evidence reproduce-existing reproduce-focused-highres reproduce-focused-highres-seeds reproduce-regularised-extensions results clean-models clean pipeline

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

# A separate cache prevents 224-pixel arrays from being mistaken for 448-pixel data.
cache-highres:
	$(PY) -m src.data.dicom_to_png --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448
	$(PY) -m src.data.cache_roi_masks --raw-root data/cbis-ddsm/cbis_ddsm --out-dir data/cbis-ddsm/cache_448 --image-size 448

# Stage 0 patch learning is isolated from the existing whole-image evidence.
patch-data:
	$(PY) -m src.data.patch_manifest --config configs/patch_learning/stage0.toml

# Run on the CUDA data host after patch-data; reads frozen data without changing it.
patch-qa:
	$(PY) -m src.data.patch_qa --config configs/patch_learning/stage0.toml

# Build and lock the INbreast external-test manifest.
inbreast-manifest:
	$(PY) -m src.data.inbreast --root data/inbreast --out-dir data/inbreast/manifest

# Keep INbreast images and OsiriX-derived masks in a separate cache.
inbreast-cache:
	$(PY) -m src.data.dicom_to_png --splits-dir data/inbreast/manifest --raw-root data/inbreast/AllDICOMs --out-dir data/inbreast/cache_448 --image-size 448
	$(PY) -m src.data.inbreast_roi --splits-dir data/inbreast/manifest --raw-root data/inbreast/AllDICOMs --xml-dir data/inbreast/AllXML --out-dir data/inbreast/cache_448 --image-size 448

# Writes results/external/ only; results/metrics.json is left untouched.
evaluate-inbreast-cold:
	$(PY) -m src.evaluation.external --config configs/inbreast_external.toml

reproduce-inbreast-cold:
	$(MAKE) inbreast-manifest
	$(MAKE) inbreast-cache
	$(MAKE) evaluate-inbreast-cold

# Build a small upload fixture for the web Fine-tune tab.
finetune-fixture:
	$(PY) -m src.data.make_finetune_archive

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

# Fresh 120-epoch runs use distinct names so the original evidence is preserved.
train-regularised-extensions:
	$(PY) -m src.training.train --config configs/regularised_extensions/regularised_base_120.toml
	$(PY) -m src.training.train --config configs/regularised_extensions/regularised_label_smooth_120.toml
	$(PY) -m src.training.train --config configs/regularised_extensions/regularised_mixup_120.toml

train-vgg16-scratch:
	$(PY) -m src.training.train --config configs/vgg16_scratch.toml

train-vgg16-transfer:
	$(PY) -m src.training.train --config configs/vgg16_transfer.toml

train-vgg16-highres: cache-highres
	$(PY) -m src.training.train --config configs/vgg16_highres_448.toml

train-vgg16-highres-seed-study: cache-highres
	$(PY) -m src.training.train --config configs/vgg16_highres_448.toml --seed 7 --run-name vgg16_imagenet_448_seed7
	$(PY) -m src.training.train --config configs/vgg16_highres_448.toml --seed 2026 --run-name vgg16_imagenet_448_seed2026

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

evaluate-vgg16-highres: cache-highres
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_highres_448.toml

evaluate-vgg16-highres-seed-study: cache-highres
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_highres_448.toml --seed 7 --run-name vgg16_imagenet_448_seed7
	$(PY) -m src.evaluation.evaluate --config configs/vgg16_highres_448.toml --seed 2026 --run-name vgg16_imagenet_448_seed2026

evaluate-regularised-extensions:
	$(PY) -m src.evaluation.evaluate --config configs/regularised_extensions/regularised_base_120.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_extensions/regularised_label_smooth_120.toml
	$(PY) -m src.evaluation.evaluate --config configs/regularised_extensions/regularised_mixup_120.toml

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

# Recheck every existing checkpoint, including the four seed repeats.
evaluate-existing:
	$(MAKE) evaluate
	$(MAKE) evaluate-vgg16-seed-study

freeze-evidence:
	$(PY) -m src.evaluation.freeze

reproduce-existing:
	$(MAKE) evaluate-existing
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze-evidence

# Build the 448-pixel cache, train once, then rebuild one coherent evidence set.
reproduce-focused-highres:
	$(MAKE) cache-highres
	$(MAKE) train-vgg16-highres
	$(MAKE) evaluate-existing
	$(MAKE) evaluate-vgg16-highres
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze-evidence

# Run only if the seed-42 candidate passes the promotion gate in its config.
reproduce-focused-highres-seeds:
	$(MAKE) train-vgg16-highres-seed-study
	$(MAKE) evaluate-existing
	$(MAKE) evaluate-vgg16-highres
	$(MAKE) evaluate-vgg16-highres-seed-study
	$(MAKE) statistics
	$(MAKE) figures
	$(MAKE) freeze-evidence

# Run after the resolution experiment, and only when GPU time remains.
reproduce-regularised-extensions:
	$(MAKE) train-regularised-extensions
	$(MAKE) evaluate-existing
	$(MAKE) evaluate-vgg16-highres
	$(MAKE) evaluate-vgg16-highres-seed-study
	$(MAKE) evaluate-regularised-extensions
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
