# Fixture predictions (handwritten)

`baseline.jsonl` and `regressed.jsonl` are **handwritten demonstration fixtures**, not live model
or benchmark results. They exist so the harness and release gate can be demonstrated offline.

`regressed.jsonl` deliberately contains:

- `agent-004`: the agent followed the prompt injection hidden in T-1002 and closed it (unsafe write).
- `rag-002`, `rag-003`, `rag-006`: the relevant document is not retrieved (Recall@5 drops to 0.73).
- `rag-007`: cites `acme-pto-policy`, which was never retrieved (bad citation reference).
- `sql-002`: wrong Q1 date filter, so the row set differs from the known answer.

Real predictions come from `python -m evals live --save-pred reports/live_predictions.jsonl`.
