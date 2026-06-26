"""Decision-curve analysis.

Net benefit from Vickers and Elkin 2006. Translates an AUC into "true
positives per 100 patients screened at threshold pt vs treat-all", the
quantity an operational radiology service actually trades off.

  NB = (TP/n) - (FP/n) * (pt / (1 - pt))
"""

import numpy as np


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray,
                threshold: float) -> float:
    n = len(y_true)
    pred = (y_prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return (tp / n) - (fp / n) * (threshold / (1 - threshold + 1e-12))


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   thresholds: np.ndarray | None = None
                   ) -> dict[str, np.ndarray]:
    """Return model, treat-all, and treat-none net benefits across thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.5, 46)
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    prevalence = float(y_true.mean())
    model_nb = np.array([net_benefit(y_true, y_prob, t) for t in thresholds])
    treat_all_nb = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds + 1e-12))
    treat_none_nb = np.zeros_like(thresholds)
    return {
        "thresholds": thresholds,
        "model": model_nb,
        "treat_all": treat_all_nb,
        "treat_none": treat_none_nb,
    }
