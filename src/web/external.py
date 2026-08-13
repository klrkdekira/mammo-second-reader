"""Read and run the external evaluation from the web app.

The run action requires explicit acknowledgement and passes readiness checks
before starting. Gradio-specific rendering lives in ``external_page``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_CONFIG = Path("configs/inbreast_external.toml")
DEFAULT_RESULT = Path("results/external/metrics-inbreast.json")

SUBSET_LABELS = {
    "full": "Full set (primary)",
    "lesion_present": "Lesion-present (secondary, drops BI-RADS 1 normals)",
}

# Metrics promoted to the headline table, in display order. Names match the
# keys produced by src.evaluation.statistics.model_intervals.
HEADLINE_METRICS = (
    ("auc", "AUC"),
    ("average_precision", "Average precision"),
    ("sensitivity", "Sensitivity @ Youden"),
    ("specificity", "Specificity @ Youden"),
    ("sensitivity_at_fixed_specificity", "Sensitivity @ fixed threshold"),
    ("specificity_at_fixed_specificity", "Specificity @ fixed threshold"),
    ("calcification_sensitivity_at_fixed_specificity", "Calcification sensitivity"),
    ("dense_breast_sensitivity_at_fixed_specificity", "Dense-breast (D4) sensitivity"),
    ("brier_score", "Brier score"),
    ("negative_log_likelihood", "Negative log-likelihood"),
)


@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class Readiness:
    """What is and is not present for a cold run, and whether one already ran."""

    checks: list[Check]
    can_run: bool
    result_path: Path
    has_result: bool

    @property
    def blocking(self) -> list[Check]:
        return [check for check in self.checks if not check.ok]


def _describe_count(path: Path, frame: pd.DataFrame | None) -> str:
    if frame is None:
        return f"missing: {path}"
    return f"{len(frame)} rows"


def _read_manifest(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError):
        return None


def readiness(
    config_path: Path = DEFAULT_CONFIG, result_path: Path = DEFAULT_RESULT
) -> Readiness:
    """Check every input the cold run needs, without running anything.

    Never raises: an unreadable input becomes a failed check with its reason, so
    the page can explain why the button is disabled instead of showing a stack
    trace.
    """
    from src.evaluation.external import (
        load_external_config,
        load_locked_operating_point,
    )

    checks: list[Check] = []
    config = None
    try:
        config = load_external_config(config_path)
        checks.append(Check("External config", True, str(config_path)))
    except (OSError, ValueError) as exc:
        checks.append(Check("External config", False, str(exc)))

    if config is None:
        return Readiness(
            checks=checks,
            can_run=False,
            result_path=result_path,
            has_result=result_path.is_file(),
        )

    checkpoint = Path("models") / f"{config.internal_run}.pt"
    sidecar = Path("models") / f"{config.internal_run}.threshold.json"
    metrics = Path("results/metrics.json")

    checks.append(
        Check(
            "Frozen checkpoint",
            checkpoint.is_file(),
            str(checkpoint) if checkpoint.is_file() else f"missing: {checkpoint}",
        )
    )
    try:
        locked = load_locked_operating_point(
            metrics_path=metrics,
            run_name=config.internal_run,
            checkpoint_path=checkpoint,
            threshold_sidecar=sidecar,
        )
        checks.append(
            Check(
                "Locked operating point",
                True,
                f"threshold and temperature agree with the frozen record "
                f"(checkpoint {locked.checkpoint_sha256[:12]}…)",
            )
        )
    except (OSError, ValueError, KeyError) as exc:
        checks.append(Check("Locked operating point", False, str(exc)))

    primary = _read_manifest(config.manifest)
    checks.append(
        Check(
            "Primary manifest",
            primary is not None,
            _describe_count(config.manifest, primary),
        )
    )
    if config.lesion_present_manifest is not None:
        subset = _read_manifest(config.lesion_present_manifest)
        checks.append(
            Check(
                "Lesion-present manifest",
                subset is not None,
                _describe_count(config.lesion_present_manifest, subset),
            )
        )
    if config.manifest_lock is not None:
        lock_present = config.manifest_lock.is_file()
        checks.append(
            Check(
                "Manifest lock",
                lock_present,
                str(config.manifest_lock)
                if lock_present
                else f"missing: {config.manifest_lock}",
            )
        )

    if primary is not None:
        root = Path(config.image_root)
        ids = primary["image_id"].astype(str).tolist()
        cached = sum(1 for image_id in ids if (root / f"{image_id}.npy").is_file())
        checks.append(
            Check(
                "Cached images",
                cached == len(ids) and bool(ids),
                f"{cached}/{len(ids)} cached in {root}",
            )
        )

    return Readiness(
        checks=checks,
        can_run=all(check.ok for check in checks),
        result_path=result_path,
        has_result=result_path.is_file(),
    )


def readiness_markdown(state: Readiness) -> str:
    """Render the readiness checks as a Markdown checklist."""
    lines = [
        f"{'[OK]' if c.ok else '[MISSING]'} **{c.label}:** {c.detail}"
        for c in state.checks
    ]
    if state.has_result:
        lines.append(
            f"**Existing result:** {state.result_path} "
            "(the cold test has already been run)"
        )
    else:
        lines.append(f"**Existing result:** none at {state.result_path}")
    return "\n\n".join(lines)


def load_result(result_path: Path = DEFAULT_RESULT) -> dict:
    """Read a completed cold-evaluation result file."""
    path = Path(result_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No cold external result at {path}. Run the evaluation on the host "
            "that holds the INbreast data, then reload this page."
        )
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def available_subsets(result: dict) -> list[str]:
    subsets = result.get("subsets")
    return sorted(subsets) if isinstance(subsets, dict) else []


def locked_markdown(result: dict) -> str:
    """Summarise the transferred operating point for display."""
    locked = result.get("locked_operating_point", {})
    protocol = result.get("protocol", {})
    checkpoint = str(locked.get("checkpoint_sha256", ""))
    model_line = (
        f"**Model** `{locked.get('source_run', '?')}` · checkpoint `{checkpoint[:16]}…`"
    )
    threshold_line = (
        f"**Youden threshold** {_fmt(locked.get('youden_threshold'))} "
        f"· **fixed-specificity threshold** "
        f"{_fmt(locked.get('fixed_specificity_threshold'))} "
        f"(target {_fmt(locked.get('fixed_specificity_target'), 2)})"
    )
    calibration_line = (
        f"**Calibration temperature** {_fmt(locked.get('temperature'))} "
        f"· refitted on external data: "
        f"**{locked.get('refitted_on_external', 'unknown')}**"
    )
    label_line = f"**Label construct:** {protocol.get('label_construct', 'unknown')}"
    return f"{model_line}\n\n{threshold_line}\n\n{calibration_line}\n\n{label_line}"


def _fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, int | float):
        return f"{value:.{digits}f}"
    return "N/A"


def _interval(metrics: dict, name: str) -> tuple[str, str]:
    entry = metrics.get(name)
    if not isinstance(entry, dict):
        return "N/A", "N/A"
    estimate = _fmt(entry.get("estimate"))
    lower, upper = entry.get("ci_lower"), entry.get("ci_upper")
    if isinstance(lower, int | float) and isinstance(upper, int | float):
        return estimate, f"{lower:.4f} to {upper:.4f}"
    return estimate, "N/A"


def headline_table(result: dict, subset: str) -> pd.DataFrame:
    """Point estimates with patient-level 95% intervals for one subset."""
    record = result.get("subsets", {}).get(subset, {})
    metrics = record.get("intervals", {}).get("metrics", {})
    rows = []
    for key, label in HEADLINE_METRICS:
        estimate, interval = _interval(metrics, key)
        rows.append({"Metric": label, "Estimate": estimate, "95% CI": interval})
    return pd.DataFrame(rows)


def summary_markdown(result: dict, subset: str) -> str:
    """One-paragraph description of what was scored and at what operating point."""
    record = result.get("subsets", {}).get(subset, {})
    if not record:
        return "No result for this subset."
    fixed = record.get("fixed_specificity", {})
    panel = record.get("test", {})
    calibration = record.get("calibration", {})
    achieved = fixed.get("achieved_specificity")
    target = fixed.get("target")
    drift = ""
    if isinstance(achieved, int | float) and isinstance(target, int | float):
        drift = (
            f" The threshold was chosen for {target:.0%} specificity internally and "
            f"achieves **{achieved:.1%}** here; that gap is the result, not an error."
        )
    return (
        f"**{SUBSET_LABELS.get(subset, subset)}:** "
        f"{record.get('n_cases', '?')} images from {record.get('n_patients', '?')} "
        f"patients, {record.get('n_malignant', '?')} malignant "
        f"({_fmt(record.get('prevalence'), 3)} prevalence). "
        f"AUC **{_fmt(panel.get('auc'))}**. "
        f"Expected calibration error {_fmt(calibration.get('ece_before'))} before and "
        f"{_fmt(calibration.get('ece_after'))} after the transferred temperature."
        + drift
    )


def strata_table(result: dict, subset: str, key: str) -> pd.DataFrame:
    """Density or lesion-type strata for one subset."""
    record = result.get("subsets", {}).get(subset, {})
    rows = record.get(key)
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame([{"note": f"no {key} recorded for this subset"}])
    frame = pd.DataFrame(rows)
    # Small strata are reported as skipped rather than silently absent, so keep
    # the reason column visible when any row was skipped.
    if "skipped_reason" in frame.columns and frame["skipped_reason"].isna().all():
        frame = frame.drop(columns="skipped_reason")
    return frame.round(4)


def roc_points(result: dict, subset: str) -> tuple[list[float], list[float]]:
    record = result.get("subsets", {}).get(subset, {})
    roc = record.get("roc", {})
    return list(roc.get("fpr", [])), list(roc.get("tpr", []))


def run_cold_evaluation(
    config_path: Path = DEFAULT_CONFIG,
    *,
    acknowledged: bool,
    result_path: Path = DEFAULT_RESULT,
) -> dict:
    """Run the cold external evaluation, refusing without an explicit acknowledgement.

    The acknowledgement is not decoration. Overwriting an existing result after
    reading it is the one action that would invalidate the pre-registration, so
    it has to be a deliberate choice recorded in the caller.
    """
    if not acknowledged:
        raise ValueError(
            "Cold external evaluation consumes a one-shot pre-registered test. "
            "Tick the acknowledgement to proceed."
        )
    state = readiness(config_path, result_path)
    if not state.can_run:
        missing = "; ".join(f"{c.label}: {c.detail}" for c in state.blocking)
        raise ValueError(f"Cold run inputs are not ready: {missing}")

    from src.evaluation.external import main as run_external

    LOGGER.warning(
        "Running the cold external evaluation; this spends the pre-registration."
    )
    return run_external(Path(config_path), output_path=Path(result_path))
