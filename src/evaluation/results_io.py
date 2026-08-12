"""Read and write evaluation results safely."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": []}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Results file is not valid JSON: {path}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        raise TypeError(f"Results file must contain a top-level runs list: {path}")
    for index, run in enumerate(data["runs"]):
        if not isinstance(run, dict) or not isinstance(run.get("model"), str):
            raise TypeError(f"Invalid run record at index {index} in {path}")
    return data


def write_json_atomic(path: Path, data: object) -> None:
    """Write JSON to a temporary file, then replace the old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(temporary, path)


def upsert_run_record(record: dict[str, object], path: Path) -> None:
    """Add or replace one model run while keeping the others."""
    model = record.get("model")
    if not isinstance(model, str) or not model:
        raise ValueError("A result record requires a non-empty model name.")
    data = _read_results(path)
    runs = list(data["runs"])
    by_model = {run["model"]: run for run in runs}
    by_model[model] = record
    data["runs"] = list(by_model.values())
    write_json_atomic(path, data)
