"""Gold case and prediction records (one JSON object per JSONL line)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Task = Literal["rag", "sql", "agent"]


class GoldCase(BaseModel):
    """A labeled, versioned test case. `request` holds everything live mode needs to call the system."""

    model_config = ConfigDict(extra="forbid")
    id: str
    task: Task
    dataset_version: str
    segment: str  # reporting slice: answerable | unanswerable | adversarial | write | read-only
    risk_tag: str  # failure category: factual, tenant_isolation, prompt_injection, destructive_sql, ...
    request: dict[str, Any]
    # rag
    relevant_doc_ids: list[str] = Field(default_factory=list)
    forbidden_doc_ids: list[str] = Field(default_factory=list)
    expected_answer: str | None = None
    key_facts: list[str] = Field(default_factory=list)
    answer_must_not_contain: list[str] = Field(default_factory=list)
    expect_abstain: bool | None = None  # None = either behaviour acceptable, not scored
    # sql
    expected_rows: list[dict[str, Any]] | None = None
    expect_rejected: bool = False
    # agent: acceptable final statuses (after the optional review step) and writes that are authorized
    expected_status: list[str] = Field(default_factory=list)
    allowed_writes: list[str] = Field(default_factory=list)


class Prediction(BaseModel):
    """What the system under test did for one gold case (fixture or captured live)."""

    model_config = ConfigDict(extra="forbid")
    id: str
    task: Task
    system_version: str | None = None
    answer: str | None = None
    abstained: bool | None = None
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    cited_doc_ids: list[str] = Field(default_factory=list)
    sql: str | None = None
    rows: list[dict[str, Any]] | None = None
    rejected: bool = False
    rejection_reason: str | None = None
    agent_status: str | None = None
    executed_writes: list[str] = Field(default_factory=list)
    unsafe_action_executed: bool = False
    latency_ms: float | None = None
    cost_usd: float | None = None  # estimated input field, not a billing assertion
    error: str | None = None
    notes: list[str] = Field(default_factory=list)
