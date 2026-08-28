"""Safety contracts for destructive Make targets."""

from pathlib import Path


def _recipe(makefile: str, target: str) -> str:
    lines = makefile.splitlines()
    start = next(
        index for index, line in enumerate(lines) if line.startswith(f"{target}:")
    )
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line and not line.startswith(("\t", " ")):
            break
        body.append(line)
    return "\n".join(body)


def test_clean_deletes_active_outputs_without_archiving() -> None:
    makefile = Path("Makefile").read_text()
    clean = _recipe(makefile, "clean")
    clean_evidence = _recipe(makefile, "clean-evidence")

    assert "$(MAKE) clean-evidence" in clean
    assert "archive-evidence" not in clean
    assert "rm -rf models" in clean_evidence
    assert "results/evidence-freeze.json" in clean_evidence
    assert "results/qa_preprocessing" in clean_evidence


def test_pipeline_resets_through_clean_target() -> None:
    makefile = Path("Makefile").read_text()
    pipeline = _recipe(makefile, "pipeline")

    assert "$(MAKE) clean" in pipeline
    assert "archive-evidence" not in pipeline
