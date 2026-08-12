"""Tests for the shared model audit."""

import numpy as np
import pandas as pd

from src.evaluation.audit import build_audit


def _manifest():
    return pd.DataFrame(
        {
            "label": [0, 1] * 10,
            "birads_density": [1] * 10 + [2] * 10,
            "lesion_type": ["mass"] * 10 + ["calcification"] * 10,
        }
    )


def test_build_audit_covers_probability_and_subgroup_results():
    frame = _manifest()
    logits = np.linspace(-2.0, 2.0, len(frame))

    audit = build_audit(
        frame,
        logits,
        frame,
        logits,
        operating_threshold=0.5,
    )

    assert audit.record["calibration"]["version"] == 2
    assert "raw" in audit.record["probability_metrics"]
    assert "precision_recall" in audit.record
    assert audit.record["fixed_specificity"]["threshold_source"] == "validation"
    assert len(audit.record["fixed_specificity"]["density_strata"]) == 4
    assert len(audit.record["fixed_specificity"]["lesion_strata"]) == 2
    assert len(audit.record["density_strata"]) == 4
    assert len(audit.record["lesion_strata"]) == 2
