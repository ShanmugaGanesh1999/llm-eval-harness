#!/usr/bin/env bash
# End-to-end offline demo: baseline fixture passes the gate, regressed fixture fails with reasons.
set -uo pipefail
cd "$(dirname "$0")/.."
run() { docker compose run --rm -T eval-runner "$@"; }

echo "== 1. Build the eval-runner image"
docker compose build -q eval-runner || exit 1

echo; echo "== 2. Baseline fixture -> pinned baseline report (expect PASS, exit 0)"
run python -m evals run --gold data/gold/v2/gold.jsonl --pred data/predictions/baseline.jsonl \
    --out reports/baseline.json
echo "exit code: $?"

echo; echo "== 3. Regressed fixture compared to the pinned baseline (expect FAIL, exit 1)"
run python -m evals run --gold data/gold/v2/gold.jsonl --pred data/predictions/regressed.jsonl \
    --baseline reports/baseline.json --out reports/regressed_report.json
code=$?
echo "exit code: $code"
[ "$code" -eq 1 ] || { echo "unexpected exit code"; exit 1; }

echo; echo "== 4. Markdown report (reports/regressed_report.md), first 45 lines"
head -n 45 reports/regressed_report.md

echo; echo "== 5. Data error: prediction file with a missing case (expect exit 2)"
head -n 27 data/predictions/baseline.jsonl > reports/truncated.jsonl
run python -m evals run --pred reports/truncated.jsonl --out reports/truncated_report.json
echo "exit code: $?"
rm -f reports/truncated.jsonl

echo; echo "Done. Reports are in ./reports (fixtures are handwritten, not benchmark results)."
