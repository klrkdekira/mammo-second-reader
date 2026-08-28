"""Build a report-update pack from the frozen internal evidence."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

from src.evaluation.verify_bundle import verify_bundle


@dataclass(frozen=True)
class StaleMarker:
    label: str
    pattern: re.Pattern[str]


CHAPTER_WORD_LIMITS = {
    1: 1000,
    2: 2500,
    3: 2000,
    4: 2000,
    5: 2500,
    6: 1000,
}
TOTAL_WORD_LIMIT = 10_500


def _marker(label: str, pattern: str) -> StaleMarker:
    return StaleMarker(label, re.compile(pattern, re.IGNORECASE))


STALE_MARKERS = (
    _marker(
        "removed future-architecture claim",
        r"five (?:other|more) architectures|configs/future_extensions|"
        r"available under `?future_extensions`?",
    ),
    _marker("old validation size", r"\b248 validation images\b"),
    _marker("old automated test count", r"\b(?:all\s+)?(?:188|201) tests\b"),
    _marker("old 448-pixel three-seed mean", r"\b0\.7257\b"),
    _marker("old leakage-free 448-pixel mean", r"\b0\.7190\b"),
    _marker("old seed-42 448-pixel AUC", r"\b0\.7250\b"),
    _marker("old seed-42 224-pixel transfer AUC", r"\b0\.6713\b"),
    _marker("old seed-42 scratch AUC", r"\b0\.5960\b"),
    _marker("old ensemble AUC", r"\b0\.6867\b"),
    _marker("old transfer AUC difference", r"\+0\.0753\b"),
    _marker("old resolution AUC difference", r"\+0\.0537\b"),
    _marker("old corrected 448-pixel mean", r"\b0\.6957\b"),
    _marker("old corrected seed-42 448-pixel AUC", r"\b0\.6321\b"),
    _marker("old corrected seed-7 448-pixel AUC", r"\b0\.7207\b"),
    _marker("old corrected seed-2026 448-pixel AUC", r"\b0\.7343\b"),
    _marker("old corrected ensemble AUC", r"\b0\.6953\b"),
    _marker(
        "old corrected central comparison",
        r"[+−-](?:0\.0409|0\.0137|0\.0496|0\.0504|0\.0837|0\.0543|0\.0128)\b",
    ),
    _marker(
        "obsolete all-seed resolution claim",
        r"every matched interval lies above zero|gain is positive at every seed|"
        r"positive at all three seeds",
    ),
    _marker(
        "current evidence described as contaminated",
        r"all internal results rely on the same split, which contains|"
        r"limits every internal result|every internal result reported above",
    ),
    _marker(
        "completed patch experiment described as unrun",
        r"no patch model was trained|patch work stopped before model training|"
        r"patches were also prepared for a future experiment|"
        r"next proposed internal study would pretrain",
    ),
    _marker(
        "corrected retraining described as unrun",
        r"cannot repair the completed whole-image runs|"
        r"retraining on the corrected manifests would|"
        r"chapter 5 reports only the recalculation",
    ),
    _marker("removed Make target claim", r"`?make reproduce-existing`?"),
)


def _load(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return value


def _number(value: float | str, digits: int = 4, *, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _escape_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def stale_findings(text: str) -> list[dict[str, object]]:
    """Return line-specific markers that need review before submission."""
    findings: list[dict[str, object]] = []
    section = ""
    for line_number, line in enumerate(text.splitlines(), start=1):
        heading = re.match(r"^###\s+(5\.\d+)", line)
        if heading:
            section = heading.group(1)
        elif line.startswith("# "):
            section = ""
        for marker in STALE_MARKERS:
            historical_number = marker.label.startswith("old ") and section in {
                "5.8",
                "5.9",
            }
            if historical_number:
                continue
            if marker.pattern.search(line):
                findings.append(
                    {
                        "line": line_number,
                        "marker": marker.label,
                        "excerpt": line.strip()[:220],
                    }
                )
    return findings


def _caption_numbers(text: str, kind: str) -> list[int]:
    return [
        int(value)
        for value in re.findall(
            rf"^\*{kind}\s+(\d+)\.", text, flags=re.IGNORECASE | re.MULTILINE
        )
    ]


def _sequence_note(values: list[int]) -> str:
    if not values:
        return "none found"
    unique = set(values)
    missing = sorted(set(range(min(values), max(values) + 1)) - unique)
    duplicates = sorted(value for value in unique if values.count(value) > 1)
    if not missing and not duplicates:
        return f"contiguous {min(values)}-{max(values)}"
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if duplicates:
        parts.append(f"duplicates {duplicates}")
    return "; ".join(parts)


def _linked_images(text: str, report_source: Path) -> tuple[int, list[str]]:
    targets = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
    missing = []
    for target in targets:
        candidate = Path(target.strip("<>"))
        if not candidate.is_absolute():
            candidate = report_source.parent / candidate
        if not candidate.resolve().is_file():
            missing.append(target)
    return len(targets), missing


def _approximate_word_count(text: str) -> int:
    without_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_links = re.sub(r"!?\[([^]]*)\]\([^)]+\)", r"\1", without_code)
    return len(re.findall(r"\b[\w]+(?:[-’'][\w]+)*\b", without_links))


def _prose_only(text: str) -> str:
    without_code = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    without_math = re.sub(r"\$\$.*?\$\$", " ", without_code, flags=re.DOTALL)
    kept_lines = []
    for line in without_math.splitlines():
        if re.match(r"^\s*\|", line):
            continue
        if re.match(r"^\s*!\[", line):
            continue
        if re.match(r"^\s*\*(?:Figure|Table)\s+\d+\.", line, re.IGNORECASE):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def _chapter_word_counts(text: str) -> tuple[dict[int, int], dict[int, int]]:
    headings = list(re.finditer(r"^# Chapter ([1-6]):.*$", text, flags=re.MULTILINE))
    references = re.search(r"^# (?:References|Bibliography)\s*$", text, re.MULTILINE)
    counts: dict[int, int] = {}
    lines: dict[int, int] = {}
    for index, heading in enumerate(headings):
        chapter = int(heading.group(1))
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        if references and heading.start() < references.start() < end:
            end = references.start()
        counts[chapter] = _approximate_word_count(
            _prose_only(text[heading.start() : end])
        )
        lines[chapter] = text.count("\n", 0, heading.start()) + 1
    return counts, lines


def _comparison(statistics: dict[str, Any], name: str) -> dict[str, float | int]:
    comparisons = statistics["paired_comparisons"]
    try:
        value = comparisons[name]["first_minus_second"]["auc"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing AUC comparison: {name}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Invalid AUC comparison: {name}")
    return value


def _comparison_sentence(label: str, value: dict[str, float | int]) -> str:
    estimate = float(value["estimate"])
    lower = float(value["ci_lower"])
    upper = float(value["ci_upper"])
    return (
        f"{label}: {_number(estimate, signed=True)} "
        f"(95% CI {_number(lower, signed=True)} to "
        f"{_number(upper, signed=True)})."
    )


def _interval_reading(value: dict[str, float | int]) -> str:
    lower = float(value["ci_lower"])
    upper = float(value["ci_upper"])
    if lower > 0:
        return "interval above zero"
    if upper < 0:
        return "interval below zero"
    return "interval crosses zero"


def build_report_pack(
    metrics: dict[str, Any],
    statistics: dict[str, Any],
    frozen: dict[str, Any],
    report_source: Path,
) -> tuple[str, list[dict[str, object]]]:
    """Render corrected tables and a read-only source audit."""
    source_text = report_source.read_text()
    findings = stale_findings(source_text)
    chapter_counts, chapter_lines = _chapter_word_counts(source_text)
    for chapter, count in chapter_counts.items():
        limit = CHAPTER_WORD_LIMITS[chapter]
        if count > limit:
            findings.append(
                {
                    "line": chapter_lines[chapter],
                    "marker": "chapter word limit",
                    "excerpt": f"Chapter {chapter}: approximately {count:,}/{limit:,} words",
                }
            )
    prose_total = sum(chapter_counts.values())
    if prose_total > TOTAL_WORD_LIMIT:
        findings.append(
            {
                "line": 1,
                "marker": "total word limit",
                "excerpt": (
                    f"Approximately {prose_total:,}/{TOTAL_WORD_LIMIT:,} prose words"
                ),
            }
        )
    runs = metrics["runs"]
    models = statistics["models"]
    run_names = [str(run["model"]) for run in runs]
    if set(run_names) != set(models):
        raise ValueError("Metrics and statistics contain different run sets.")

    transfer_names = (
        "vgg16_imagenet_minus_vgg16_scratch",
        "vgg16_imagenet_seed7_minus_vgg16_scratch_seed7",
        "vgg16_imagenet_seed2026_minus_vgg16_scratch_seed2026",
    )
    resolution_names = (
        "vgg16_imagenet_448_minus_vgg16_imagenet",
        "vgg16_imagenet_448_seed7_minus_vgg16_imagenet_seed7",
        "vgg16_imagenet_448_seed2026_minus_vgg16_imagenet_seed2026",
    )
    transfer = [_comparison(statistics, name) for name in transfer_names]
    resolution = [_comparison(statistics, name) for name in resolution_names]
    ensemble = _comparison(statistics, "ensemble_minus_vgg16_imagenet")
    best_name = max(
        run_names, key=lambda name: float(models[name]["metrics"]["auc"]["estimate"])
    )
    best_auc = models[best_name]["metrics"]["auc"]

    transfer_readings = [_interval_reading(value) for value in transfer]
    if all(reading == "interval above zero" for reading in transfer_readings):
        transfer_summary = (
            "- Transfer is directionally positive at all three seeds, and all "
            "three paired intervals are above zero."
        )
    elif all(float(value["estimate"]) > 0 for value in transfer):
        crossing_seeds = [
            str(seed)
            for seed, reading in zip((42, 7, 2026), transfer_readings, strict=True)
            if reading == "interval crosses zero"
        ]
        transfer_summary = (
            "- Transfer is directionally positive at all three seeds, but the "
            f"paired interval crosses zero for seed(s) {', '.join(crossing_seeds)}."
        )
    else:
        transfer_summary = (
            "- The transfer effect changes direction across seeds; report the "
            "seed-specific estimates and intervals rather than a universal benefit."
        )

    resolution_estimates = [float(value["estimate"]) for value in resolution]
    if any(value > 0 for value in resolution_estimates) and any(
        value < 0 for value in resolution_estimates
    ):
        resolution_summary = (
            "- The 448-pixel effect changes sign across seeds. Report it as "
            "seed-dependent, not as a universal resolution benefit."
        )
    elif all(value > 0 for value in resolution_estimates):
        resolution_summary = (
            "- The 448-pixel AUC difference is positive at all three seeds; "
            "interpret each paired interval separately."
        )
    else:
        resolution_summary = (
            "- The 448-pixel AUC difference is non-positive at all three seeds; "
            "interpret each paired interval separately."
        )

    ensemble_reading = _interval_reading(ensemble)
    ensemble_summary = (
        "- The ensemble is numerically above seed-42 VGG-16, and its paired AUC "
        f"{ensemble_reading}."
    )

    lines = [
        "# Corrected report update pack",
        "",
        (
            "This file is generated from the frozen internal evidence. It does not "
            f"edit `{report_source.as_posix()}`."
        ),
        "",
        "## Evidence lock",
        "",
        f"- Internal result records: {len(run_names)}",
        (
            f"- Test set: {models[run_names[0]]['n_cases']} images from "
            f"{models[run_names[0]]['n_patients']} patients"
        ),
        "- Validation rows: 247",
        f"- Bootstrap: {statistics['method']['n_resamples']} patient-level resamples",
        f"- Frozen figures: {len(frozen.get('figures', []))}",
        f"- Source snapshot: `{frozen['source_snapshot']}`",
        "",
        "## Required narrative corrections",
        "",
        (
            "- Describe the original contaminated runs, the 595-image post-hoc "
            "subset, and the corrected 645-image reruns as three separate phases."
        ),
        (
            "- Treat all corrected internal estimates as controlled retrospective "
            "reruns because the test set had already informed development."
        ),
        (
            "- Keep the INbreast result attached to the earlier frozen model; it is "
            "not an external result for the corrected models."
        ),
        (
            f"- The highest single saved-run AUC is `{best_name}` at "
            f"{_number(best_auc['estimate'])} (95% CI "
            f"{_number(best_auc['ci_lower'])}-{_number(best_auc['ci_upper'])}); "
            "state this descriptively rather than as a test-selected winner."
        ),
        transfer_summary,
        resolution_summary,
        ensemble_summary,
        "",
        "Exact central comparisons:",
        "",
    ]
    for label, value in zip(
        ("Transfer seed 42", "Transfer seed 7", "Transfer seed 2026"),
        transfer,
        strict=True,
    ):
        lines.append(f"- {_comparison_sentence(label, value)}")
    for label, value in zip(
        ("448 minus 224 seed 42", "448 minus 224 seed 7", "448 minus 224 seed 2026"),
        resolution,
        strict=True,
    ):
        lines.append(f"- {_comparison_sentence(label, value)}")
    lines.append(f"- {_comparison_sentence('Ensemble minus seed-42 VGG-16', ensemble)}")

    lines.extend(
        [
            "",
            "## Corrected model table",
            "",
            "| Run | AUC (95% CI) | Accuracy | Sensitivity | Specificity | Threshold |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in run_names:
        model = models[name]
        model_metrics = model["metrics"]
        auc = model_metrics["auc"]
        lines.append(
            f"| `{name}` | {_number(auc['estimate'])} "
            f"({_number(auc['ci_lower'])}-{_number(auc['ci_upper'])}) | "
            f"{_number(model_metrics['accuracy']['estimate'])} | "
            f"{_number(model_metrics['sensitivity']['estimate'])} | "
            f"{_number(model_metrics['specificity']['estimate'])} | "
            f"{_number(model['threshold'])} |"
        )

    lines.extend(
        [
            "",
            "## Three-seed summaries",
            "",
            "| Configuration | Mean AUC | Sample SD | Range |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, summary in statistics["seed_repeats"].items():
        lines.append(
            f"| `{name}` | {_number(summary['auc_mean'])} | "
            f"{_number(summary['auc_standard_deviation'])} | "
            f"{_number(summary['auc_min'])}-{_number(summary['auc_max'])} |"
        )

    lines.extend(
        [
            "",
            "## Paired AUC comparisons",
            "",
            "| First minus second | Difference (95% CI) | Unadjusted p-value | Reading |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for name, comparison in statistics["paired_comparisons"].items():
        auc = comparison["first_minus_second"]["auc"]
        lower = float(auc["ci_lower"])
        upper = float(auc["ci_upper"])
        reading = _interval_reading(auc)
        lines.append(
            f"| `{name}` | {_number(auc['estimate'], signed=True)} "
            f"({_number(lower, signed=True)} to {_number(upper, signed=True)}) | "
            f"{_number(auc['p_two_sided'])} | {reading} |"
        )

    figure_numbers = _caption_numbers(source_text, "Figure")
    table_numbers = _caption_numbers(source_text, "Table")
    image_count, missing_images = _linked_images(source_text, report_source)
    if image_count != len(figure_numbers):
        findings.append(
            {
                "line": 1,
                "marker": "image/caption count mismatch",
                "excerpt": (
                    f"Linked images: {image_count}; numbered figure captions: "
                    f"{len(figure_numbers)}"
                ),
            }
        )
    lines.extend(
        [
            "",
            "## Read-only source audit",
            "",
            f"- Approximate Markdown word count: {_approximate_word_count(source_text):,}",
            f"- Approximate prose-only word count: {prose_total:,}",
            f"- Figure captions: {len(figure_numbers)} ({_sequence_note(figure_numbers)})",
            f"- Table captions: {len(table_numbers)} ({_sequence_note(table_numbers)})",
            f"- Linked images: {image_count}; missing files: {len(missing_images)}",
            f"- Stale markers: {len(findings)}",
            "",
        ]
    )
    if chapter_counts:
        lines.extend(
            [
                (
                    "Approximate chapter prose counts (tables, figure/table captions, "
                    "linked images, code blocks and display equations excluded):"
                ),
                "",
                "| Chapter | Approximate words | Limit | Status |",
                "| ---: | ---: | ---: | --- |",
            ]
        )
        for chapter in sorted(chapter_counts):
            count = chapter_counts[chapter]
            limit = CHAPTER_WORD_LIMITS[chapter]
            status = "over" if count > limit else "within"
            lines.append(f"| {chapter} | {count:,} | {limit:,} | {status} |")
        lines.append("")
    if missing_images:
        lines.append("Missing image links:")
        lines.append("")
        lines.extend(f"- `{target}`" for target in missing_images)
        lines.append("")
    if findings:
        lines.extend(
            [
                (
                    "The scan identifies text that needs contextual review. Values "
                    "in the historical sensitivity table or the locked external "
                    "section may remain when they are explicitly labelled as earlier "
                    "evidence; do not replace numbers blindly."
                ),
                "",
            ]
        )
        lines.extend(
            [
                "| Line | Review reason | Current excerpt |",
                "| ---: | --- | --- |",
            ]
        )
        for finding in findings:
            lines.append(
                f"| {finding['line']} | {_escape_cell(finding['marker'])} | "
                f"{_escape_cell(finding['excerpt'])} |"
            )
    else:
        lines.append("No configured stale markers were found.")
    lines.extend(
        [
            "",
            (
                "The word count is a reproducible approximation. Use the submission "
                "editor for the final declared count."
            ),
            "",
        ]
    )
    return "\n".join(lines), findings


def generate_report_pack(
    metrics_path: Path = Path("results/metrics.json"),
    statistics_path: Path = Path("results/statistics.json"),
    freeze_path: Path = Path("results/evidence-freeze.json"),
    report_source: Path = Path("report.md"),
    output_path: Path = Path("report-update.md"),
) -> list[dict[str, object]]:
    """Verify inputs and write a deterministic report-update pack."""
    verify_bundle(metrics_path, statistics_path, freeze_path)
    metrics = _load(metrics_path, "metrics")
    statistics = _load(statistics_path, "statistics")
    frozen = _load(freeze_path, "evidence freeze")
    rendered, findings = build_report_pack(metrics, statistics, frozen, report_source)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    return findings


@click.command()
@click.option(
    "--metrics",
    "metrics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/metrics.json"),
    show_default=True,
)
@click.option(
    "--statistics",
    "statistics_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/statistics.json"),
    show_default=True,
)
@click.option(
    "--freeze",
    "freeze_path",
    type=click.Path(exists=True, path_type=Path),
    default=Path("results/evidence-freeze.json"),
    show_default=True,
)
@click.option(
    "--report-source",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
)
@click.option("--fail-on-stale", is_flag=True, default=False)
def cli(
    metrics_path: Path,
    statistics_path: Path,
    freeze_path: Path,
    report_source: Path,
    output_path: Path,
    fail_on_stale: bool,
) -> None:
    """Generate corrected tables and scan the report source without editing it."""
    try:
        findings = generate_report_pack(
            metrics_path,
            statistics_path,
            freeze_path,
            report_source,
            output_path,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Wrote {output_path} with {len(findings)} stale marker(s).")
    if fail_on_stale and findings:
        raise click.ClickException(
            f"The report source still has {len(findings)} stale marker(s). "
            f"Review {output_path}, edit {report_source}, then rerun report-check."
        )


if __name__ == "__main__":
    cli()
