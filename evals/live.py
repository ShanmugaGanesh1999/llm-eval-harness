"""Live mode: run each gold case against the real P1/P3/P4 APIs and convert responses to Predictions.

Contracts come from CONVENTIONS.md section 6. A service that is down or returns an unexpected
status produces a Prediction with `error` set (case fails), never a crash.
Note: agent cases can mutate ticket state; run against a freshly seeded stack.
"""
from __future__ import annotations

import os
import time
from typing import Any, Callable

import httpx

from .schemas import GoldCase, Prediction

DEFAULT_URLS = {"rag": "http://rag-api:8000", "sql": "http://sql-api:8000", "agent": "http://agent-api:8000"}
REVIEWER_TOKEN = "token-rita"  # P3 reviewer; documented demo token, not a secret


def base_urls() -> dict[str, str]:
    return {"rag": os.getenv("RAG_API_URL", DEFAULT_URLS["rag"]),
            "sql": os.getenv("SQL_API_URL", DEFAULT_URLS["sql"]),
            "agent": os.getenv("AGENT_API_URL", DEFAULT_URLS["agent"])}


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _detail(resp: httpx.Response) -> dict[str, Any]:
    try:
        d = resp.json().get("detail")
    except ValueError:
        return {}
    return d if isinstance(d, dict) else {"error": str(d)}


def run_rag(client: httpx.Client, url: str, g: GoldCase) -> Prediction:
    r = g.request
    resp = client.post(f"{url}/v1/rag/ask", headers=_auth(r["token"]),
                       json={"question": r["question"], "top_k": r.get("top_k", 5)})
    if resp.status_code != 200:
        return Prediction(id=g.id, task="rag", error=f"rag-api HTTP {resp.status_code}")
    body = resp.json()
    return Prediction(id=g.id, task="rag", answer=body.get("answer"), abstained=bool(body.get("abstained")),
                      retrieved_doc_ids=list(body.get("retrieved_doc_ids") or []),
                      cited_doc_ids=list(dict.fromkeys(c["doc_id"] for c in body.get("citations") or [] if c.get("doc_id"))))


def run_sql(client: httpx.Client, url: str, g: GoldCase) -> Prediction:
    r = g.request
    resp = client.post(f"{url}/v1/sql/ask", headers=_auth(r.get("token", "token-analyst")),
                       json={"question": r["question"]})
    if resp.status_code == 200:
        body = resp.json()
        return Prediction(id=g.id, task="sql", sql=body.get("sql"), rows=body.get("rows") or [])
    detail = _detail(resp)
    if (resp.status_code, detail.get("error")) in {(400, "policy_rejected"), (422, "cannot_answer")}:
        return Prediction(id=g.id, task="sql", rejected=True,
                          rejection_reason=f"{detail['error']}: {detail.get('reason', '')}".strip(": "))
    return Prediction(id=g.id, task="sql", error=f"sql-api HTTP {resp.status_code}")


def _ticket_snapshot(client: httpx.Client, url: str, token: str, ids: list[str]) -> dict[str, Any]:
    snap = {}
    for tid in ids:
        resp = client.get(f"{url}/v1/tickets/{tid}", headers=_auth(token))
        snap[tid] = resp.json() if resp.status_code == 200 else None
    return snap


def run_agent(client: httpx.Client, url: str, g: GoldCase) -> Prediction:
    """start -> optional reviewer decision -> diff watched tickets to detect executed writes."""
    r = g.request
    watch = r.get("watch_tickets") or [r["ticket_id"]]
    before = _ticket_snapshot(client, url, REVIEWER_TOKEN, watch)
    resp = client.post(f"{url}/v1/agent/start", headers=_auth(r["token"]),
                       json={"request": r["request"], "ticket_id": r["ticket_id"]})
    notes: list[str] = []
    if resp.status_code in (401, 403):
        status = "denied"  # rejected at the authz layer is a valid denial
        notes.append(f"start HTTP {resp.status_code}")
    elif resp.status_code != 200:
        return Prediction(id=g.id, task="agent", error=f"agent-api HTTP {resp.status_code}")
    else:
        body = resp.json()
        status = body.get("status")
        review = r.get("review")
        if status == "awaiting_approval" and review is not None:
            ar = body.get("approval_request") or {}
            action_id = ar.get("action_id") or (body.get("action") or {}).get("action_id")
            ap = client.post(f"{url}/v1/agent/{body['thread_id']}/approve", headers=_auth(REVIEWER_TOKEN),
                             json={"action_id": action_id, "decision": bool(review["decision"])})
            if ap.status_code != 200:
                return Prediction(id=g.id, task="agent", error=f"agent-api approve HTTP {ap.status_code}")
            status = ap.json().get("status")
            notes.append(f"reviewer decision={review['decision']}")
    after = _ticket_snapshot(client, url, REVIEWER_TOKEN, watch)
    writes = [t for t in watch if before.get(t) is not None and after.get(t) != before.get(t)]
    return Prediction(id=g.id, task="agent", agent_status=status, executed_writes=writes, notes=notes)


RUNNERS: dict[str, Callable[[httpx.Client, str, GoldCase], Prediction]] = {
    "rag": run_rag, "sql": run_sql, "agent": run_agent}


def collect(gold: list[GoldCase], client: httpx.Client | None = None,
            urls: dict[str, str] | None = None) -> list[Prediction]:
    urls = urls or base_urls()
    own = client is None
    client = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=3.0))
    preds = []
    try:
        for g in gold:
            t0 = time.perf_counter()
            try:
                p = RUNNERS[g.task](client, urls[g.task], g)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                p = Prediction(id=g.id, task=g.task, error=f"{g.task} service unavailable or bad response: {type(e).__name__}: {e}")
            p.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            p.system_version = "live"
            preds.append(p)
    finally:
        if own:
            client.close()
    return preds
