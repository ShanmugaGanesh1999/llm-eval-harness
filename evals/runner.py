"""Orchestrates one evaluation run: load -> align -> score -> compare -> gate -> report."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from . import gates, scoring
from .datasets import DataError, align, load_gold, read_jsonl, sha256_file
from .schemas import GoldCase, Prediction

FIXTURE_NOTE = ("Predictions are a handwritten fixture used to demonstrate the harness, "
                "not live benchmark results.")


def evaluate(gold: list[GoldCase], preds: list[Prediction], *, gold_path: str, pred_label: str,
             gates_path: str, mode: str, baseline_path: str | None = None, seed: int = 0) -> dict[str, Any]:
    by_id = align(gold, preds)
    report: dict[str, Any] = {"meta": {
        "mode": mode, "dataset_version": gold[0].dataset_version, "gold_path": gold_path,
        "gold_sha256": sha256_file(gold_path), "predictions": pred_label,
        "system_version": next((p.system_version for p in preds if p.system_version), None),
        "gates_config": gates_path, "seed": seed,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "fixture_note": FIXTURE_NOTE if mode == "offline" and "fixture" in (by_id[gold[0].id].system_version or "") else None,
    }}
    report.update(scoring.score(gold, by_id, seed=seed))
    if baseline_path:
        if not Path(baseline_path).is_file():
            raise DataError(f"baseline report not found: {baseline_path}")
        report["baseline_comparison"] = scoring.compare(report, json.loads(Path(baseline_path).read_text()), seed)
    report["gate"] = gates.evaluate(report, gates.load_config(gates_path))
    return report


def run_offline(gold_path: str, pred_path: str, gates_path: str, baseline_path: str | None = None,
                seed: int = 0) -> dict[str, Any]:
    gold = load_gold(gold_path)
    preds = read_jsonl(pred_path, Prediction)
    return evaluate(gold, preds, gold_path=gold_path, pred_label=pred_path, gates_path=gates_path,
                    mode="offline", baseline_path=baseline_path, seed=seed)
