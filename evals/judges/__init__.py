"""Optional LLM-as-judge interface. Not used by the release gate by default.

A judge only *assists* on semantic questions (is the answer faithful to the context?). It must be
calibrated against human labels per task/risk category before it influences any gate, and it is
never a policy-enforcement control. See docs/system-design.md "LLM-as-judge calibration".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class JudgeResult:
    label: bool  # True = supported / acceptable
    score: float  # 0..1
    rationale: str
    judge_version: str


class Judge(Protocol):
    version: str

    def judge(self, question: str, answer: str, contexts: Sequence[str], reference: str | None = None) -> JudgeResult: ...


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "an", "of", "to", "is", "are", "and", "in", "c1", "c2"}


class MockJudge:
    """Deterministic faithfulness stand-in: share of answer tokens found in the retrieved contexts."""

    version = "mock-overlap-v1"

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def judge(self, question: str, answer: str, contexts: Sequence[str], reference: str | None = None) -> JudgeResult:
        ans = _tokens(answer)
        if not ans:
            return JudgeResult(False, 0.0, "empty answer", self.version)
        support = _tokens(" ".join(contexts))
        score = len(ans & support) / len(ans)
        unsupported = sorted(ans - support)[:5]
        return JudgeResult(score >= self.threshold, round(score, 4),
                           f"unsupported tokens: {unsupported}" if unsupported else "all tokens supported", self.version)


def agreement(judge_labels: Sequence[bool], human_labels: Sequence[bool]) -> dict[str, float]:
    """Raw agreement and Cohen's kappa between a judge and human labels on the calibration set."""
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("label lists must be non-empty and equal length")
    n = len(judge_labels)
    po = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    pj, ph = sum(judge_labels) / n, sum(human_labels) / n
    pe = pj * ph + (1 - pj) * (1 - ph)
    kappa = 1.0 if pe == 1 else (po - pe) / (1 - pe)
    fn = sum(h and not j for j, h in zip(judge_labels, human_labels))
    fp = sum(j and not h for j, h in zip(judge_labels, human_labels))
    return {"n": n, "agreement": po, "cohen_kappa": kappa, "judge_false_accepts": fp, "judge_false_rejects": fn}
