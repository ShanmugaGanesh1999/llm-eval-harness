"""Load, validate and align gold cases and predictions; dataset manifests."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .schemas import GoldCase, Prediction

M = TypeVar("M", bound=BaseModel)


class DataError(Exception):
    """Bad input data: malformed JSONL, id mismatch, manifest drift. CLI exit code 2."""


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: str | Path, model: type[M]) -> list[M]:
    path = Path(path)
    if not path.is_file():
        raise DataError(f"file not found: {path}")
    out: list[M] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(model.model_validate_json(line))
        except ValidationError as e:
            raise DataError(f"{path}:{n}: {e.errors()[0]['loc']} {e.errors()[0]['msg']}") from e
    return out


def load_gold(path: str | Path, verify_manifest: bool = True) -> list[GoldCase]:
    cases = read_jsonl(path, GoldCase)
    if not cases:
        raise DataError(f"gold set is empty: {path}")
    dupes = [i for i, c in Counter(c.id for c in cases).items() if c > 1]
    if dupes:
        raise DataError(f"duplicate gold ids: {dupes}")
    versions = {c.dataset_version for c in cases}
    if len(versions) != 1:
        raise DataError(f"gold file mixes dataset versions: {sorted(versions)}")
    if verify_manifest:
        check_manifest(Path(path))
    return cases


def align(gold: list[GoldCase], preds: list[Prediction]) -> dict[str, Prediction]:
    """Exactly one prediction per gold case, same task. Anything else is a data error, not a score."""
    dupes = [i for i, c in Counter(p.id for p in preds).items() if c > 1]
    gold_ids = {g.id for g in gold}
    by_id = {p.id: p for p in preds}
    missing = sorted(gold_ids - by_id.keys())
    extra = sorted(by_id.keys() - gold_ids)
    wrong_task = sorted(g.id for g in gold if g.id in by_id and by_id[g.id].task != g.task)
    problems = []
    if dupes:
        problems.append(f"duplicate prediction ids {sorted(dupes)}")
    if missing:
        problems.append(f"missing predictions for {missing}")
    if extra:
        problems.append(f"predictions without gold case {extra}")
    if wrong_task:
        problems.append(f"task mismatch for {wrong_task}")
    if problems:
        raise DataError("gold/prediction mismatch: " + "; ".join(problems))
    return by_id


def build_manifest(directory: str | Path, created: str | None = None) -> dict:
    directory = Path(directory)
    files, tasks = {}, Counter()
    for f in sorted(directory.glob("*.jsonl")):
        cases = read_jsonl(f, GoldCase)
        tasks.update(c.task for c in cases)
        files[f.name] = {"sha256": sha256_file(f), "cases": len(cases)}
    manifest = {"dataset_version": directory.name, "created": created or dt.date.today().isoformat(),
                "files": files, "cases_by_task": dict(sorted(tasks.items())),
                "note": "Any edit to a gold file requires a new dataset version directory."}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check_manifest(gold_path: Path) -> None:
    manifest_path = gold_path.parent / "manifest.json"
    if not manifest_path.is_file():
        return
    entry = json.loads(manifest_path.read_text()).get("files", {}).get(gold_path.name)
    if entry and entry["sha256"] != sha256_file(gold_path):
        raise DataError(f"{gold_path} does not match {manifest_path} sha256: gold edited without a version bump")
