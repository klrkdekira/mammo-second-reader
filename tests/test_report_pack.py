from src.reporting.report_pack import build_report_pack, stale_findings

RUNS = (
    "vgg16_scratch",
    "vgg16_scratch_seed7",
    "vgg16_scratch_seed2026",
    "vgg16_imagenet",
    "vgg16_imagenet_seed7",
    "vgg16_imagenet_seed2026",
    "vgg16_imagenet_448",
    "vgg16_imagenet_448_seed7",
    "vgg16_imagenet_448_seed2026",
    "ensemble",
)


def _model(auc: float) -> dict[str, object]:
    interval = {
        "estimate": auc,
        "ci_lower": auc - 0.05,
        "ci_upper": auc + 0.05,
    }
    return {
        "n_cases": 645,
        "n_patients": 349,
        "threshold": 0.5,
        "metrics": {
            "auc": interval,
            "accuracy": {"estimate": 0.6},
            "sensitivity": {"estimate": 0.7},
            "specificity": {"estimate": 0.5},
        },
    }


def _paired(estimate: float) -> dict[str, object]:
    return {
        "first_minus_second": {
            "auc": {
                "estimate": estimate,
                "ci_lower": estimate - 0.02,
                "ci_upper": estimate + 0.02,
                "p_two_sided": 0.05,
            }
        }
    }


def _statistics() -> dict[str, object]:
    comparisons = {
        "vgg16_imagenet_minus_vgg16_scratch": _paired(0.04),
        "vgg16_imagenet_seed7_minus_vgg16_scratch_seed7": _paired(0.01),
        "vgg16_imagenet_seed2026_minus_vgg16_scratch_seed2026": _paired(0.05),
        "vgg16_imagenet_448_minus_vgg16_imagenet": _paired(-0.05),
        "vgg16_imagenet_448_seed7_minus_vgg16_imagenet_seed7": _paired(0.08),
        "vgg16_imagenet_448_seed2026_minus_vgg16_imagenet_seed2026": _paired(0.05),
        "ensemble_minus_vgg16_imagenet": _paired(0.01),
    }
    return {
        "method": {"n_resamples": 2000},
        "models": {name: _model(0.60 + index / 100) for index, name in enumerate(RUNS)},
        "paired_comparisons": comparisons,
        "seed_repeats": {
            "vgg16_scratch": {
                "auc_mean": 0.61,
                "auc_standard_deviation": 0.01,
                "auc_min": 0.60,
                "auc_max": 0.62,
            }
        },
    }


def test_stale_findings_report_line_numbers():
    findings = stale_findings("Current text.\nOld mean 0.7257.\n")

    assert findings == [
        {
            "line": 2,
            "marker": "old 448-pixel three-seed mean",
            "excerpt": "Old mean 0.7257.",
        }
    ]


def test_stale_findings_allow_labelled_historical_numbers():
    text = (
        "### 5.8 Cold external evaluation\nEarlier internal AUC was 0.7250.\n"
        "### 5.9 Historical audit\nOriginal mean was 0.7257.\n"
        "### 5.10 Current limits\nCurrent mean is still 0.7257.\n"
    )

    findings = stale_findings(text)

    assert [(finding["line"], finding["marker"]) for finding in findings] == [
        (6, "old 448-pixel three-seed mean")
    ]


def test_report_pack_renders_tables_and_read_only_scan(tmp_path):
    report = tmp_path / "FinalReport.md"
    report.write_text(
        "# Report\n\nThe gain is positive at every seed.\n\n"
        "*Table 1. Results.*\n\n*Figure 1. Plot.*\n"
    )
    metrics = {"runs": [{"model": name} for name in RUNS]}
    statistics = _statistics()
    frozen = {"figures": [{"path": "one.png"}], "source_snapshot": "a" * 64}

    rendered, findings = build_report_pack(metrics, statistics, frozen, report)

    assert "## Corrected model table" in rendered
    assert "`vgg16_imagenet_448_seed2026`" in rendered
    assert "448 minus 224 seed 42: -0.0500" in rendered
    assert "Figure captions: 1 (contiguous 1-1)" in rendered
    assert findings[0]["line"] == 3
    assert findings[0]["marker"] == "obsolete all-seed resolution claim"
