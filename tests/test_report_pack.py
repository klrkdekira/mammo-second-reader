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


def test_stale_findings_detect_completed_work_described_as_unrun():
    text = (
        "Five more architectures are available under `future_extensions`.\n"
        "No patch model was trained.\n"
        "Retraining on the corrected manifests would be future work.\n"
        "The automated suite passed all 201 tests.\n"
    )

    findings = stale_findings(text)

    assert [finding["marker"] for finding in findings] == [
        "removed future-architecture claim",
        "completed patch experiment described as unrun",
        "corrected retraining described as unrun",
        "old automated test count",
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


def test_report_pack_derives_narrative_from_current_intervals(tmp_path):
    report = tmp_path / "FinalReport.md"
    report.write_text("# Report\n")
    metrics = {"runs": [{"model": name} for name in RUNS]}
    statistics = _statistics()
    for name in (
        "vgg16_imagenet_minus_vgg16_scratch",
        "vgg16_imagenet_seed7_minus_vgg16_scratch_seed7",
        "vgg16_imagenet_seed2026_minus_vgg16_scratch_seed2026",
        "ensemble_minus_vgg16_imagenet",
    ):
        auc = statistics["paired_comparisons"][name]["first_minus_second"]["auc"]
        auc["ci_lower"] = 0.001
        auc["ci_upper"] = 0.08
    frozen = {"figures": [], "source_snapshot": "a" * 64}

    rendered, _ = build_report_pack(metrics, statistics, frozen, report)

    assert "all three paired intervals are above zero" in rendered
    assert "ensemble is numerically above seed-42 VGG-16" in rendered
    assert "paired AUC interval above zero" in rendered


def test_report_pack_fails_closed_on_chapter_word_limits(tmp_path):
    report = tmp_path / "FinalReport.md"
    report.write_text(
        "# Chapter 1: Introduction\n" + "word " * 999 + "\n"
        "# Chapter 5: Evaluation\n" + "word " * 2501 + "\n"
        "# Chapter 6: Conclusion\n" + "word " * 999 + "\n"
    )
    metrics = {"runs": [{"model": name} for name in RUNS]}
    frozen = {"figures": [], "source_snapshot": "a" * 64}

    rendered, findings = build_report_pack(metrics, _statistics(), frozen, report)

    assert "| 5 | 2,504 | 2,500 | over |" in rendered
    assert "chapter word limit" in {finding["marker"] for finding in findings}


def test_report_pack_fails_closed_on_total_word_limit(tmp_path):
    report = tmp_path / "FinalReport.md"
    chapter_sizes = (950, 2450, 1950, 1950, 2450, 950)
    report.write_text(
        "".join(
            f"# Chapter {chapter}: Title\n" + "word " * size + "\n"
            for chapter, size in enumerate(chapter_sizes, start=1)
        )
    )
    metrics = {"runs": [{"model": name} for name in RUNS]}
    frozen = {"figures": [], "source_snapshot": "a" * 64}

    _, findings = build_report_pack(metrics, _statistics(), frozen, report)

    assert "total word limit" in {finding["marker"] for finding in findings}


def test_report_pack_flags_an_unnumbered_linked_image(tmp_path):
    report = tmp_path / "FinalReport.md"
    image = tmp_path / "plot.png"
    image.write_bytes(b"png")
    report.write_text("# Report\n\n![Plot](plot.png)\n\n*Unnumbered plot caption.*\n")
    metrics = {"runs": [{"model": name} for name in RUNS]}
    frozen = {"figures": [], "source_snapshot": "a" * 64}

    _, findings = build_report_pack(metrics, _statistics(), frozen, report)

    assert "image/caption count mismatch" in {finding["marker"] for finding in findings}
