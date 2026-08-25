"""Post-hoc leakage sensitivity analysis.

The official CBIS-DDSM mass and calcification partitions were combined without
first reconciling their participant assignments, so a small number of patients
appear in both the development and test manifests used by the frozen 22-run
evidence base. Stage 0 of the patch-learning programme discovered this and
froze an exclusion ledger.

This module quantifies the effect on the milestone evidence without retraining
anything. It recomputes the headline discrimination metrics and the two central
paired comparisons on the subset of test images whose patients never appeared
in training or validation, reusing the same bootstrap functions that produced
`results/statistics.json`.

This analysis is POST-HOC. No selection rule was registered before the leakage
was discovered, and the affected test images were part of every result the
project has already reported. It bounds an error; it does not replace the
frozen evidence, which is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import pandas as pd

from src.evaluation.statistics import model_intervals, paired_comparison

DEFAULT_LEDGER = Path("manifests/cbis-ddsm/excluded-test-overlap-patients.csv")
DEFAULT_PREDICTIONS = Path("results/predictions")
DEFAULT_OUT = Path("results/leakage_sensitivity/metrics-clean-subset.json")

# Runs carrying a claim in the report: the headline resolution model and its
# seed repeats, the matched transfer pair, and the ensemble.
REPORTED_RUNS = (
    "vgg16_imagenet_448",
    "vgg16_imagenet_448_seed7",
    "vgg16_imagenet_448_seed2026",
    "vgg16_imagenet",
    "vgg16_scratch",
    "ensemble",
)

# The two comparisons the report treats as statistically supported.
CENTRAL_PAIRS = (
    ("vgg16_imagenet", "vgg16_scratch", "ImageNet transfer minus scratch"),
    (
        "vgg16_imagenet_448",
        "vgg16_imagenet",
        "448 pixels minus 224 pixels",
    ),
)


def contaminated_patients(ledger_path: Path) -> set[str]:
    """Read the Stage 0 exclusion ledger."""
    ledger = pd.read_csv(ledger_path)
    return set(ledger["patient_id"].astype(str))


def load_predictions(run: str, predictions_dir: Path) -> pd.DataFrame:
    path = predictions_dir / f"{run}.test.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    return pd.read_csv(path)


def split_frame(
    frame: pd.DataFrame, excluded: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the full frame and the leakage-free subset."""
    mask = ~frame["patient_id"].astype(str).isin(excluded)
    return frame, frame.loc[mask].reset_index(drop=True)


def _auc_estimate(result: dict) -> float:
    return float(result["metrics"]["auc"]["estimate"])


def _pair_auc(result: dict) -> dict:
    return result["first_minus_second"]["auc"]


def _auc_block(result: dict) -> dict[str, float | int]:
    metrics = result["metrics"]["auc"]
    return {
        "n_cases": result["n_cases"],
        "n_patients": result["n_patients"],
        "auc": metrics["estimate"],
        "ci_lower": metrics["ci_lower"],
        "ci_upper": metrics["ci_upper"],
    }


def run_analysis(
    *,
    ledger_path: Path = DEFAULT_LEDGER,
    predictions_dir: Path = DEFAULT_PREDICTIONS,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, object]:
    excluded = contaminated_patients(ledger_path)

    models: dict[str, object] = {}
    frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for run in REPORTED_RUNS:
        full, clean = split_frame(load_predictions(run, predictions_dir), excluded)
        frames[run] = (full, clean)
        full_result = model_intervals(full, n_resamples=n_resamples, seed=seed)
        clean_result = model_intervals(clean, n_resamples=n_resamples, seed=seed)
        models[run] = {
            "full": _auc_block(full_result),
            "clean": _auc_block(clean_result),
            "auc_shift": _auc_estimate(clean_result) - _auc_estimate(full_result),
        }

    comparisons: dict[str, object] = {}
    for first, second, label in CENTRAL_PAIRS:
        full_pair = paired_comparison(
            frames[first][0], frames[second][0], n_resamples=n_resamples, seed=seed
        )
        clean_pair = paired_comparison(
            frames[first][1], frames[second][1], n_resamples=n_resamples, seed=seed
        )
        key = f"{first}__minus__{second}"
        comparisons[key] = {
            "label": label,
            "full": _pair_auc(full_pair),
            "clean": _pair_auc(clean_pair),
        }

    reference = frames[REPORTED_RUNS[0]]
    return {
        "version": 1,
        "analysis": "post_hoc_leakage_sensitivity",
        "status": (
            "POST-HOC. No selection rule was registered before the leakage was "
            "discovered. This bounds the effect of the split defect on the "
            "frozen evidence; it does not replace it."
        ),
        "leakage": {
            "cause": (
                "The official CBIS-DDSM mass and calcification partitions were "
                "combined without reconciling participant assignments, so some "
                "patients occur in both the development and test manifests."
            ),
            "ledger": str(ledger_path),
            "n_contaminated_patients": len(excluded),
            "n_test_cases_full": len(reference[0]),
            "n_test_cases_clean": len(reference[1]),
            "n_test_cases_removed": len(reference[0]) - len(reference[1]),
        },
        "bootstrap": {"n_resamples": n_resamples, "seed": seed},
        "models": models,
        "central_comparisons": comparisons,
    }


def main(out_path: Path = DEFAULT_OUT, *, n_resamples: int = 2000) -> dict[str, object]:
    payload = run_analysis(n_resamples=n_resamples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    payload = main(args.out, n_resamples=args.resamples)
    leakage = cast(dict, payload["leakage"])
    print(
        f"Removed {leakage['n_test_cases_removed']} of "
        f"{leakage['n_test_cases_full']} test cases "
        f"({leakage['n_contaminated_patients']} patients)."
    )
    for run, entry in cast(dict, payload["models"]).items():
        print(
            f"{run:<30} full {entry['full']['auc']:.4f} -> "
            f"clean {entry['clean']['auc']:.4f} "
            f"({entry['auc_shift']:+.4f})"
        )
    for entry in cast(dict, payload["central_comparisons"]).values():
        full, clean = entry["full"], entry["clean"]
        print(
            f"{entry['label']:<34} full {full['estimate']:+.4f} "
            f"[{full['ci_lower']:+.4f}, {full['ci_upper']:+.4f}] -> "
            f"clean {clean['estimate']:+.4f} "
            f"[{clean['ci_lower']:+.4f}, {clean['ci_upper']:+.4f}]"
        )


if __name__ == "__main__":
    cli()
