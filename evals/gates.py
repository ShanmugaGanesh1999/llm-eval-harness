"""Release gate: severity-tagged hard gates, thresholds and max regression vs a pinned baseline."""
from __future__ import annotations

import operator
from pathlib import Path
from typing import Any

import yaml

OPS = {">=": operator.ge, "<=": operator.le, "==": operator.eq, ">": operator.gt, "<": operator.lt}
# summary metric -> per-case field, to name the offending cases in failure messages
CASE_FIELD = {"unsafe_action_count": "unsafe_actions", "tenant_leakage_count": "leakage",
              "injection_violation_count": "injection_violations", "recall_at_5": "recall_at_5",
              "citation_validity": "citation_valid", "abstention_accuracy": "abstention_correct",
              "answer_accuracy": "answer_correct", "sql_execution_accuracy": "execution_correct",
              "sql_rejection_accuracy": "rejection_correct", "agent_status_accuracy": "status_correct",
              "pass_rate": "passed"}


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    for g in cfg.get("gates", []):
        if g.get("op") not in OPS:
            raise ValueError(f"gate {g.get('name')}: unsupported op {g.get('op')!r}")
    return cfg


def offending_cases(metric: str, cases: list[dict]) -> list[str]:
    f = CASE_FIELD.get(metric)
    if not f:
        return []
    out = []
    for c in cases:
        v = c[f] if f in c else c["checks"].get(f)
        if v is False or (isinstance(v, (int, float)) and not isinstance(v, bool) and
                          (v > 0 if metric.endswith("_count") else v < 1.0)):
            out.append(c["id"])
    return out


def evaluate(report: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    summary, cases, results = report["summary"], report["cases"], []
    for g in cfg.get("gates", []):
        actual = summary.get(g["metric"])
        r = {"name": g.get("name", g["metric"]), "kind": g.get("kind", "threshold"), "metric": g["metric"],
             "op": g["op"], "threshold": g["value"], "actual": actual, "severity": g.get("severity", "high")}
        if actual is None:
            r["status"], r["message"] = "not_applicable", "metric not computed for this dataset"
        elif OPS[g["op"]](round(actual, 9), g["value"]):
            r["status"], r["message"] = "pass", ""
        else:
            ids = offending_cases(g["metric"], cases)
            r["status"] = "fail"
            r["message"] = f"{g['metric']} = {actual:.4g}, required {g['op']} {g['value']}" + (f" (cases: {', '.join(ids)})" if ids else "")
        results.append(r)

    reg = cfg.get("regression") or {}
    comparison = report.get("baseline_comparison")
    if reg and comparison:
        for metric in reg.get("metrics", []):
            d = comparison["deltas"].get(metric)
            r = {"name": f"no_regression:{metric}", "kind": "regression", "metric": metric, "op": ">=",
                 "threshold": -reg["max_drop"], "actual": d["delta"] if d else None,
                 "severity": reg.get("severity", "high")}
            if d is None:
                r["status"], r["message"] = "not_applicable", "metric missing in baseline or current run"
            elif d["delta"] >= -reg["max_drop"] - 1e-9:
                r["status"], r["message"] = "pass", ""
            else:
                r["status"] = "fail"
                r["message"] = (f"{metric} regressed {d['baseline']:.3f} -> {d['current']:.3f} "
                                f"(delta {d['delta']:+.3f}, max drop {reg['max_drop']})")
            results.append(r)

    failures = [f"[{r['severity']}] {r['name']}: {r['message']}" for r in results if r["status"] == "fail"]
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    failures.sort(key=lambda s: order.get(s[1:s.index("]")], 9))
    return {"passed": not failures, "results": results, "failures": failures}
