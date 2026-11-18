"""CLI. Exit codes: 0 gate passed, 1 gate failed, 2 data/config error."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import live, report, runner
from .datasets import DataError, build_manifest, load_gold


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gold", default=os.getenv("EVAL_GOLD", "data/gold/v2/gold.jsonl"))
    p.add_argument("--out", default=os.getenv("EVAL_OUT", "reports/eval_report.json"))
    p.add_argument("--gates", default=os.getenv("EVAL_GATES", "config/gates.yaml"))
    p.add_argument("--baseline", default=os.getenv("EVAL_BASELINE") or None, help="pinned baseline report JSON")
    p.add_argument("--seed", type=int, default=int(os.getenv("EVAL_SEED", "0")))


def _finish(rep: dict, out: str) -> int:
    json_path, md_path = report.write(rep, out)
    s, gate = rep["summary"], rep["gate"]
    print(f"dataset={rep['meta']['dataset_version']} cases={s['n_cases']} pass_rate={s['pass_rate']:.3f} "
          f"recall@5={s['recall_at_5']} unsafe={s['unsafe_action_count']} leakage={s['tenant_leakage_count']}")
    print(f"report: {json_path} , {md_path}")
    if gate["passed"]:
        print("RELEASE GATE: PASS")
        return 0
    print("RELEASE GATE: FAIL")
    for f in gate["failures"]:
        print(f"  - {f}")
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evals")
    sub = ap.add_subparsers(dest="cmd")
    run_p = sub.add_parser("run", help="score saved predictions (offline)")
    _common(run_p)
    run_p.add_argument("--pred", default=os.getenv("EVAL_PRED", "data/predictions/baseline.jsonl"))
    live_p = sub.add_parser("live", help="call the real services, then score")
    _common(live_p)
    live_p.add_argument("--save-pred", default=None, help="also write captured predictions JSONL")
    man_p = sub.add_parser("manifest", help="rebuild manifest.json for a gold version directory")
    man_p.add_argument("--dir", default="data/gold/v2")

    argv = sys.argv[1:] if argv is None else argv
    if not argv:  # container default: EVAL_MODE picks the subcommand
        argv = ["live" if os.getenv("EVAL_MODE", "offline") == "live" else "run"]
    args = ap.parse_args(argv)
    try:
        if args.cmd == "manifest":
            print(json.dumps(build_manifest(args.dir), indent=2))
            return 0
        if args.cmd == "run":
            rep = runner.run_offline(args.gold, args.pred, args.gates, args.baseline, args.seed)
            return _finish(rep, args.out)
        gold = load_gold(args.gold)
        preds = live.collect(gold)
        if args.save_pred:
            Path(args.save_pred).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_pred).write_text("".join(p.model_dump_json() + "\n" for p in preds))
        rep = runner.evaluate(gold, preds, gold_path=args.gold, pred_label="live", gates_path=args.gates,
                              mode="live", baseline_path=args.baseline, seed=args.seed)
        return _finish(rep, args.out)
    except (DataError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
