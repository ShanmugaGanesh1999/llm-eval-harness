# P5: LLM evaluation harness

A versioned golden dataset, deterministic metrics, and a release gate for the RAG (P1), guarded
agent (P3) and Text-to-SQL (P4) projects. It scores saved predictions offline, or calls the live
services and scores what they return. The exit code is the gate verdict, so CI can block a merge.

No LLM judge is used by default and no paid keys are needed. The prediction files in
`data/predictions/` are **handwritten fixtures** for the demo, not benchmark results.

## Quickstart

```bash
docker compose build
docker compose run --rm eval-runner            # baseline fixture: RELEASE GATE: PASS, exit 0
docker compose run --rm eval-runner python -m evals run \
  --pred data/predictions/regressed.jsonl      # RELEASE GATE: FAIL, exit 1
docker compose run --rm eval-runner pytest -q  # unit + live-adapter tests (no services needed)
./scripts/demo.sh                              # baseline -> regressed vs baseline -> markdown report
docker compose down -v
```

Reports are written to `./reports/` (bind mount): `eval_report.json` and `eval_report.md`.

Exit codes: `0` gate passed, `1` gate failed, `2` data or config error (missing file, gold/prediction
id mismatch, gold file edited without a version bump).

## CLI

```bash
python -m evals run  --gold data/gold/v2/gold.jsonl --pred data/predictions/baseline.jsonl \
                     --out reports/eval_report.json [--baseline reports/baseline.json] [--seed 0]
python -m evals live --gold data/gold/v2/gold.jsonl --out reports/live_report.json \
                     [--save-pred reports/live_predictions.jsonl]
python -m evals manifest --dir data/gold/v2      # recompute sha256 manifest after creating a new version
```

With no arguments (the container default) `EVAL_MODE=offline|live` picks `run` or `live`.

## Live mode

```bash
# From the top-level stack (all five projects on one network):
EVAL_MODE=live docker compose run --rm eval-runner
```

Base URLs come from `RAG_API_URL`, `SQL_API_URL`, `AGENT_API_URL` (defaults `http://rag-api:8000`,
`http://sql-api:8000`, `http://agent-api:8000`). A down or misbehaving service is recorded as an
error on each affected case; the run finishes and the gate fails on `no_harness_errors`.
Agent cases close tickets, so run live mode against a freshly seeded P3.

## What is measured

| Area | Metrics |
|---|---|
| Retrieval | Recall@5 (None when no relevant docs), MRR@10, nDCG@10 |
| Grounding | Citation reference validity (cited ⊆ retrieved), forbidden-content hits |
| Answers | Abstention accuracy and balanced accuracy, answer correctness (key facts / normalized / numeric) |
| Safety | Tenant leakage count, injection violations, unsafe action count, adversarial SQL rejection |
| SQL | Execution accuracy (order-insensitive row multiset, numeric tolerance) |
| Agent | Final status accuracy, unauthorized writes |
| Ops | Latency p50/p95, cost per case and per successful task, error count |
| Stats | Bootstrap 95% CI for headline metrics, paired bootstrap of deltas vs baseline, per-segment and per-risk breakdown |

Gates live in `config/gates.yaml`. The thresholds are illustrative demo values.

## Folder map

```text
05-eval-harness/
├── evals/                 # cli, datasets (load/align/manifest), metrics, scoring, gates, report, live, judges/
├── config/gates.yaml      # hard gates, thresholds, max regression vs baseline
├── data/gold/v1/, v2/     # gold.jsonl (28 cases) + manifest.json (sha256); v2 fixes one label, see v2/CHANGELOG.md
├── data/predictions/      # baseline.jsonl (passes), regressed.jsonl (fails); handwritten fixtures
├── tests/                 # metrics, harness/gates, live adapter with httpx.MockTransport
├── scripts/demo.sh
└── docs/system-design.md
```
