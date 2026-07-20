"""Lesion-type-stratified evaluation.

Masses and calcifications are near-distinct vision tasks: mass malignancy
shows in centimetre-scale margin and shape features, calcification
malignancy in sub-millimetre speck morphology that whole-image
downsampling can erase. Stratifying the test panel by the CBIS
abnormality type shows whether one lesion family is dragging down the
pooled metrics, which is the evidence needed to justify (or reject)
training separate mass and calc models.
"""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate

LESION_TYPES = ("mass", "calcification")


def metrics_by_lesion_type(test_df: pd.DataFrame, y_prob: np.ndarray,
                           threshold: float, min_n: int = 10) -> pd.DataFrame:
    types = test_df["lesion_type"].astype(str).str.strip().str.lower().values
    rows = []
    for lesion in LESION_TYPES:
        mask = types == lesion
        n = int(mask.sum())
        if n < min_n:
            rows.append({"lesion_type": lesion, "n": n, "auc": None,
                         "acc": None, "sens": None, "spec": None,
                         "ppv": None, "skipped_reason": f"n<{min_n}"})
            continue
        panel = evaluate(test_df.loc[mask, "label"].values,
                         y_prob[mask], threshold=threshold)
        rows.append({"lesion_type": lesion, "n": n, "auc": panel.auc,
                     "acc": panel.accuracy, "sens": panel.sensitivity,
                     "spec": panel.specificity, "ppv": panel.ppv,
                     "skipped_reason": None})
    return pd.DataFrame(rows)
