"""JSON + Markdown report writers."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.2f}"
    return str(v)


def to_markdown(report: dict[str, Any]) -> str:
    meta, s, gate = report["meta"], report["summary"], report["gate"]
    lines = [f"# Evaluation report: {'PASS' if gate['passed'] else 'FAIL'}", "",
             f"- Mode: `{meta['mode']}`; dataset `{meta['dataset_version']}` (sha256 `{meta['gold_sha256'][:12]}`)",
             f"- Predictions: `{meta['predictions']}`; system version `{meta.get('system_version') or 'n/a'}`",
             f"- Gates: `{meta['gates_config']}` (thresholds are illustrative); bootstrap seed {meta['seed']}",
             f"- Cases: {s['n_cases']} {s['n_by_task']}", ""]
    if meta.get("fixture_note"):
        lines += [f"> {meta['fixture_note']}", ""]
    lines += ["## Release gate", ""]
    if gate["failures"]:
        lines += [f"- {f}" for f in gate["failures"]] + [""]
    lines += ["| Gate | Kind | Severity | Actual | Required | Status |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['kind']} | {r['severity']} | {_fmt(r['actual'])} | {r['op']} {r['threshold']} | {r['status']} |"
              for r in gate["results"]]
    ci = report["confidence_intervals"]
    lines += ["", "## Metrics", "", "| Metric | Value | 95% bootstrap CI |", "|---|---|---|"]
    for k, v in s.items():
        if k in ("n_cases", "n_by_task"):
            continue
        c = ci.get(k)
        lines.append(f"| {k} | {_fmt(v)} | {f'[{c['low']:.3f}, {c['high']:.3f}] n={c['n']}' if c else ''} |")
    if comp := report.get("baseline_comparison"):
        lines += ["", "## Baseline comparison", ""]
        if comp["warning"]:
            lines += [f"> {comp['warning']}", ""]
        lines += ["| Metric | Baseline | Current | Delta | Paired 95% CI of delta |", "|---|---|---|---|---|"]
        for k, d in comp["deltas"].items():
            if d["delta"] == 0:
                continue
            pb = comp["paired_bootstrap"].get(k)
            ci_txt = f"[{pb['low']:+.3f}, {pb['high']:+.3f}]" if pb and "low" in pb else ""
            lines.append(f"| {k} | {_fmt(d['baseline'])} | {_fmt(d['current'])} | {d['delta']:+.4g} | {ci_txt} |")
        lines += ["", f"Newly failing cases: {', '.join(comp['newly_failing']) or 'none'}"]
    lines += ["", "## Segments", "", "| Segment | n | Pass rate | Failed |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {_fmt(v['pass_rate'])} | {', '.join(v['failed'])} |" for k, v in report["segments"].items()]
    failed = [c for c in report["cases"] if not c["passed"]]
    lines += ["", "## Failed cases", ""]
    lines += [f"- **{c['id']}** ({c['task']}, {c['risk_tag']}): {'; '.join(c['failures'])}" for c in failed] or ["None."]
    return "\n".join(lines) + "\n"


def write(report: dict[str, Any], out: str | Path) -> tuple[Path, Path]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    md = out.with_suffix(".md")
    md.write_text(to_markdown(report))
    return out, md
