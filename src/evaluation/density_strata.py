"""Density-stratified evaluation.

Mammography sensitivity is known to fall sharply on dense breasts. CBIS
ships ordinal BI-RADS density labels 1 to 4 that map directly onto the
categories used by Payne 2025 (Volpara a to d, calibrated to BI-RADS 5th
edition) and Woo 2025 (BI-RADS A to D), making the three results
comparable category-by-category.
"""

from typing import TypedDict

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate


class DensityMetricRow(TypedDict):
    density: int
    n: int
    auc: float | None
    acc: float | None
    sens: float | None
    spec: float | None
    ppv: float | None
    skipped_reason: str | None


def metrics_by_density(
    test_df: pd.DataFrame, y_prob: np.ndarray, threshold: float, min_n: int = 10
) -> pd.DataFrame:
    densities = np.asarray(test_df["birads_density"].to_numpy())
    labels = np.asarray(test_df["label"].to_numpy())
    rows: list[DensityMetricRow] = []
    for d in (1, 2, 3, 4):
        mask = densities == d
        n = int(mask.sum())
        if n < min_n:
            rows.append(
                {
                    "density": d,
                    "n": n,
                    "auc": None,
                    "acc": None,
                    "sens": None,
                    "spec": None,
                    "ppv": None,
                    "skipped_reason": f"n<{min_n}",
                }
            )
            continue
        panel = evaluate(labels[mask], y_prob[mask], threshold=threshold)
        rows.append(
            {
                "density": d,
                "n": n,
                "auc": panel.auc,
                "acc": panel.accuracy,
                "sens": panel.sensitivity,
                "spec": panel.specificity,
                "ppv": panel.ppv,
                "skipped_reason": None,
            }
        )
    return pd.DataFrame(rows)


PAYNE_2025_SENSITIVITY = {1: 0.750, 2: 0.735, 3: 0.598, 4: 0.513}
PAYNE_2025_INTERVAL_PER_1000 = {1: 1.8, 2: 2.6, 3: 4.8, 4: 7.9}
