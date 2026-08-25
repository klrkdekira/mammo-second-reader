# Mammo Second Reader

An auditable mammography-classification research pipeline developed for the
CM3070 Computer Science Final Project. It covers deterministic preprocessing,
patient-disjoint experiment manifests, model training, validation-derived
operating thresholds, case-level predictions, patient-level uncertainty and an
interactive research demonstration.

> **Research use only.** This software is not a medical device, has not been
> clinically validated and must not be used to diagnose, triage or make treatment
> decisions.

## Environment

Python 3.12 and [uv](https://docs.astral.sh/uv/) are required. Linux CUDA hosts
use the locked PyTorch CUDA 12.8 source configured in `pyproject.toml`.

```bash
uv sync --locked
make check
```

`make check` runs the complete test, Ruff formatting/lint and mypy gates. Use
`make help` for the intentionally small supported command surface.

## Data and canonical split

Raw CBIS-DDSM data is not distributed by this repository. Place the official
case-description CSVs under `data/cbis-ddsm/` and the DICOM tree under
`data/cbis-ddsm/cbis_ddsm/`. The repository tracks the small authoritative
manifests and overlap ledger under `manifests/cbis-ddsm/`; large images and
caches remain ignored.

The canonical manifests contain 2,147 train, 247 validation and 645 test images.
Their patient sets are pairwise disjoint. Test assignment takes precedence when
official mass and calcification metadata assign the same patient to different
partitions. Every training and evaluation entry point refuses overlapping
manifests. Registered counts, hashes, rerun names and analysis rules are in
[`CORRECTED_RERUN_PROTOCOL.md`](CORRECTED_RERUN_PROTOCOL.md).

To reproduce the split builder into a temporary comparison directory before
touching the frozen manifests:

```bash
uv run python -m src.data.splits --splits-dir /tmp/cbis-ddsm-split-check
```

Build the aligned caches after verifying the comparison hashes:

```bash
make cache-224
make cache-448
make qa-preprocessing
```

## Training and evaluation

One parameterised target handles every single-model run; there are no legacy or
alternate “clean” targets.

```bash
make train CONFIG=configs/vgg16_transfer.toml
make evaluate CONFIG=configs/vgg16_transfer.toml

make train CONFIG=configs/vgg16_transfer.toml \
  SEED=7 RUN_NAME=vgg16_imagenet_seed7
make evaluate CONFIG=configs/vgg16_transfer.toml \
  SEED=7 RUN_NAME=vgg16_imagenet_seed7
```

Training writes a best-AUC checkpoint, history and validation-derived threshold
sidecar to `models/`. Evaluation reloads those artefacts and writes an upserted
record to `results/metrics.json` plus validation/test prediction CSVs. Each record
hashes its config, manifests, checkpoint, predictions and relevant code, and
captures the Git commit and runtime environment.

After the registered runs are complete:

```bash
make ensemble
make evidence
```

This produces patient-level bootstrap statistics, report figures and a validated
`results/evidence-freeze.json`. The corrected internal rerun protocol is
retrospective because the test set informed earlier work. The retained
`results/leakage_sensitivity/` record documents the discovered historical split
contamination. `results/external/` is frozen evidence for the earlier model and
must not be presented as external validation of a corrected model.

## Research interface

```bash
make web
```

The interface supports research inference, explanation, batch evaluation and a
bounded fine-tuning demonstration. Predictions can be wrong; Grad-CAM is a model
attention visualisation, not a lesion diagnosis or causal explanation. Dataset
shift, retrospective selection, limited external evidence and subgroup sample
sizes constrain generalisation.

## Repository map

- `configs/` — experiment definitions
- `manifests/cbis-ddsm/` — canonical split CSVs and exclusion ledger
- `src/data/` — ingestion, preprocessing and manifest validation
- `src/training/` and `src/evaluation/` — model fitting and evidence generation
- `src/web/` — interactive research application
- `results/` — small report evidence; checkpoints and medical images are ignored
- `tests/` — regression, audit and end-to-end tests

## Citation

If you use this software, cite it using [`CITATION.cff`](CITATION.cff).
