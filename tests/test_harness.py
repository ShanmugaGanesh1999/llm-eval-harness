import json

import pytest

from conftest import BASELINE, GATES, GOLD, REGRESSED
from evals import cli, gates
from evals.datasets import DataError, align, load_gold, read_jsonl
from evals.runner import evaluate, run_offline
from evals.schemas import Prediction


def test_gold_counts_and_manifest():
    gold = load_gold(GOLD)
    tasks = {t: sum(g.task == t for g in gold) for t in ("rag", "sql", "agent")}
    assert tasks == {"rag": 16, "sql": 7, "agent": 5}
    assert {g.risk_tag for g in gold} >= {"tenant_isolation", "prompt_injection", "destructive_sql", "missing_answer"}


def test_baseline_passes_gate():
    rep = run_offline(GOLD, BASELINE, GATES)
    assert rep["gate"]["passed"], rep["gate"]["failures"]
    s = rep["summary"]
    assert s["unsafe_action_count"] == 0 and s["tenant_leakage_count"] == 0
    assert s["recall_at_5"] == 1.0 and s["sql_execution_accuracy"] == 1.0 and s["sql_rejection_accuracy"] == 1.0
    assert rep["meta"]["fixture_note"]


def test_regressed_fails_with_reasons():
    rep = run_offline(GOLD, REGRESSED, GATES)
    gate = rep["gate"]
    assert not gate["passed"]
    failed = {r["name"]: r for r in gate["results"] if r["status"] == "fail"}
    assert set(failed) == {"no_unsafe_actions", "citation_validity", "recall_at_5", "sql_execution_accuracy"}
    assert "agent-004" in failed["no_unsafe_actions"]["message"]
    assert "rag-007" in failed["citation_validity"]["message"]
    assert "sql-002" in failed["sql_execution_accuracy"]["message"]
    assert gate["failures"][0].startswith("[critical]")  # sorted by severity


def test_regression_vs_baseline(tmp_path):
    base = run_offline(GOLD, BASELINE, GATES)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(base))
    rep = run_offline(GOLD, REGRESSED, GATES, baseline_path=str(path))
    comp = rep["baseline_comparison"]
    assert comp["comparable"]
    assert comp["deltas"]["unsafe_action_count"]["delta"] == 1
    assert "agent-004" in comp["newly_failing"]
    names = {r["name"] for r in rep["gate"]["results"] if r["status"] == "fail"}
    assert "no_regression:recall_at_5" in names and "no_regression:pass_rate" in names
    # baseline vs itself: no regression failures
    same = run_offline(GOLD, BASELINE, GATES, baseline_path=str(path))
    assert same["gate"]["passed"]


def test_id_mismatch_detected():
    gold = load_gold(GOLD)
    preds = read_jsonl(BASELINE, Prediction)
    with pytest.raises(DataError, match="missing predictions"):
        align(gold, preds[:-1])
    with pytest.raises(DataError, match="duplicate prediction"):
        align(gold, preds + [preds[0]])
    with pytest.raises(DataError, match="without gold"):
        align(gold, preds + [Prediction(id="ghost", task="rag")])
    bad = [p.model_copy(update={"task": "sql"}) if p.id == "rag-001" else p for p in preds]
    with pytest.raises(DataError, match="task mismatch"):
        align(gold, bad)


def test_manifest_drift_detected(tmp_path):
    d = tmp_path / "v1"
    d.mkdir()
    src = open(GOLD).read()
    (d / "gold.jsonl").write_text(src)
    (d / "manifest.json").write_text(json.dumps({"files": {"gold.jsonl": {"sha256": "0" * 64}}}))
    with pytest.raises(DataError, match="version bump"):
        load_gold(d / "gold.jsonl")


def test_leakage_and_injection_are_hard_failures():
    gold = load_gold(GOLD)
    preds = read_jsonl(BASELINE, Prediction)
    preds = [p.model_copy(update={"retrieved_doc_ids": ["acme-pto-policy", "globex-pto-policy"],
                                  "answer": "Up to 5 days, or 10 days at Globex [C1]."})
             if p.id == "rag-001" else p for p in preds]
    rep = evaluate(gold, preds, gold_path=GOLD, pred_label="t", gates_path=GATES, mode="offline")
    assert rep["summary"]["tenant_leakage_count"] == 1
    assert rep["summary"]["injection_violation_count"] == 1
    assert {"no_tenant_leakage", "no_injection_followed"} <= {r["name"] for r in rep["gate"]["results"] if r["status"] == "fail"}


def test_unrejected_adversarial_sql_counts_as_unsafe():
    gold = load_gold(GOLD)
    preds = [p.model_copy(update={"rejected": False, "sql": "DELETE FROM orders", "rows": []})
             if p.id == "sql-005" else p for p in read_jsonl(BASELINE, Prediction)]
    rep = evaluate(gold, preds, gold_path=GOLD, pred_label="t", gates_path=GATES, mode="offline")
    assert rep["summary"]["unsafe_action_count"] == 1
    assert rep["summary"]["sql_rejection_accuracy"] < 1


def test_errored_case_fails_not_crashes():
    gold = load_gold(GOLD)
    preds = [Prediction(id=p.id, task=p.task, error="connect refused") if p.task == "sql" else p
             for p in read_jsonl(BASELINE, Prediction)]
    rep = evaluate(gold, preds, gold_path=GOLD, pred_label="t", gates_path=GATES, mode="live")
    assert rep["summary"]["error_count"] == 7
    assert not rep["gate"]["passed"]


def test_bootstrap_ci_deterministic_in_report():
    a = run_offline(GOLD, REGRESSED, GATES, seed=3)["confidence_intervals"]
    b = run_offline(GOLD, REGRESSED, GATES, seed=3)["confidence_intervals"]
    assert a == b and a["pass_rate"]["low"] <= a["pass_rate"]["high"]


def test_gate_config_rejects_bad_op(tmp_path):
    p = tmp_path / "g.yaml"
    p.write_text("gates:\n  - {name: x, metric: pass_rate, op: '~=', value: 1}\n")
    with pytest.raises(ValueError):
        gates.load_config(p)


def test_cli_exit_codes(tmp_path):
    out = tmp_path / "r.json"
    assert cli.main(["run", "--gold", GOLD, "--pred", BASELINE, "--gates", GATES, "--out", str(out)]) == 0
    assert out.exists() and out.with_suffix(".md").read_text().startswith("# Evaluation report: PASS")
    assert cli.main(["run", "--gold", GOLD, "--pred", REGRESSED, "--gates", GATES, "--out", str(out)]) == 1
    assert "agent-004" in out.with_suffix(".md").read_text()
    assert cli.main(["run", "--gold", GOLD, "--pred", str(tmp_path / "nope.jsonl"), "--gates", GATES,
                     "--out", str(out)]) == 2


def test_judge_mock_and_agreement():
    from evals.judges import MockJudge, agreement
    j = MockJudge()
    ok = j.judge("q", "carry over up to 5 days", ["Employees may carry over up to 5 days."])
    bad = j.judge("q", "carry over up to 30 days plus bonus vacation", ["Employees may carry over up to 5 days."])
    assert ok.label and not bad.label and ok == j.judge("q", "carry over up to 5 days", ["Employees may carry over up to 5 days."])
    a = agreement([True, True, False, False], [True, False, False, False])
    assert a["agreement"] == 0.75 and a["judge_false_accepts"] == 1
