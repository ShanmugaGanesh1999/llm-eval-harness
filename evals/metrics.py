"""Pure metric functions.

Convention: `None` means "not applicable" and is excluded from averages. It is never a free
perfect score (e.g. Recall@K on an unanswerable question).
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Iterable, Sequence


def dedupe(ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 5) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(dedupe(retrieved)[:k]) & rel) / len(rel)


def mrr_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    for rank, doc in enumerate(dedupe(retrieved)[:k], start=1):
        if doc in rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int = 10) -> float | None:
    """Binary-relevance nDCG: DCG = sum 1/log2(rank+1) over relevant hits, normalized by the ideal DCG."""
    rel = set(relevant)
    if not rel:
        return None
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(dedupe(retrieved)[:k]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def citation_validity(cited: Sequence[str], retrieved: Sequence[str]) -> bool | None:
    """Cited IDs must be a subset of retrieved IDs. Reference validity only, not claim support."""
    if not cited:
        return None
    return set(cited) <= set(retrieved)


def leakage_count(forbidden: Iterable[str], *id_lists: Iterable[str]) -> int:
    seen: set[str] = set()
    for ids in id_lists:
        seen |= set(ids)
    return len(set(forbidden) & seen)


def abstention_correct(abstained: bool | None, expect_abstain: bool | None) -> bool | None:
    if expect_abstain is None:
        return None
    if abstained is None:  # no observable behaviour (e.g. error) is never correct
        return False
    return abstained == expect_abstain


def balanced_accuracy(pairs: Iterable[tuple[bool, bool]]) -> float | None:
    """Mean of per-class recall over (expected, predicted) pairs; robust when one class is rare."""
    by_class: dict[bool, list[bool]] = {True: [], False: []}
    for expected, predicted in pairs:
        by_class[expected].append(expected == predicted)
    recalls = [sum(v) / len(v) for v in by_class.values() if v]
    return sum(recalls) / len(recalls) if recalls else None


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: _NUM_WORDS[m.group(1)], s)
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # 1,200 -> 1200
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)  # keep decimal points only
    s = re.sub(r"[^a-z0-9.$@\-\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _contains(haystack: str, needle: str) -> bool:
    """Whole-token containment so '5 days' does not match '15 days'."""
    return bool(needle) and re.search(r"(?<![\w.])" + re.escape(needle) + r"(?![\w])", haystack) is not None


def extract_numbers(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", normalize_text(s))]


def answer_correct(answer: str | None, expected: str | None, key_facts: Sequence[str] = (),
                   rel_tol: float = 1e-4) -> bool | None:
    """Key-fact containment if facts are given, else normalized match, else numeric tolerance."""
    if not expected and not key_facts:
        return None
    if not answer:
        return False
    ans = normalize_text(answer)
    if key_facts:
        return all(_contains(ans, normalize_text(f)) for f in key_facts)
    if _contains(ans, normalize_text(expected or "")):
        return True
    wanted, got = extract_numbers(expected or ""), extract_numbers(answer)
    return bool(wanted) and all(any(abs(w - g) <= rel_tol * max(1.0, abs(w)) for g in got) for w in wanted)


def forbidden_content_hits(answer: str | None, phrases: Sequence[str]) -> int:
    if not answer:
        return 0
    ans = normalize_text(answer)
    return sum(1 for p in phrases if _contains(ans, normalize_text(p)))


def _norm_value(v: Any) -> tuple[str, Any]:
    if isinstance(v, bool) or v is None:
        return ("s", str(v).lower())
    if isinstance(v, (int, float)):
        return ("n", float(v))
    text = str(v).strip()
    try:
        return ("n", float(text.replace(",", "")))
    except ValueError:
        return ("s", text.lower())


def _values_equal(a: tuple[str, Any], b: tuple[str, Any], abs_tol: float, rel_tol: float) -> bool:
    if a[0] != b[0]:
        return False
    if a[0] == "n":
        return abs(a[1] - b[1]) <= max(abs_tol, rel_tol * abs(b[1]))
    return a[1] == b[1]


def rows_match(predicted: Sequence[dict[str, Any]], expected: Sequence[dict[str, Any]],
               abs_tol: float = 0.01, rel_tol: float = 1e-6) -> bool:
    """Order-insensitive multiset comparison of rows. Column names are ignored (models alias freely);
    each row is compared as its sorted bag of normalized values with numeric tolerance."""
    def norm(row: dict[str, Any]) -> list[tuple[str, Any]]:
        return sorted((_norm_value(v) for v in row.values()), key=lambda t: (t[0], t[1]))

    if len(predicted) != len(expected):
        return False
    pending = [norm(r) for r in predicted]
    for exp_row in map(norm, expected):
        for i, cand in enumerate(pending):
            if len(cand) == len(exp_row) and all(_values_equal(c, e, abs_tol, rel_tol)
                                                 for c, e in zip(cand, exp_row)):
                pending.pop(i)
                break
        else:
            return False
    return True


def mean(values: Iterable[Any]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def percentile(values: Iterable[float | None], q: float) -> float | None:
    """Linear-interpolated percentile, q in [0, 100]."""
    xs = sorted(float(v) for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def bootstrap_ci(values: Iterable[Any], *, seed: int = 0, iterations: int = 1000,
                 alpha: float = 0.05) -> dict[str, float] | None:
    """Percentile bootstrap CI of the mean. Deterministic for a given seed."""
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return None
    rng = random.Random(seed)
    means = [sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(iterations)]
    return {"low": percentile(means, 100 * alpha / 2), "high": percentile(means, 100 * (1 - alpha / 2)),
            "n": len(xs)}


def paired_bootstrap_delta(current: dict[str, Any], baseline: dict[str, Any], *, seed: int = 0,
                           iterations: int = 1000, alpha: float = 0.05) -> dict[str, float] | None:
    """CI of mean(current - baseline) over case IDs present in both runs (paired resampling)."""
    ids = sorted(i for i in current if i in baseline and current[i] is not None and baseline[i] is not None)
    if len(ids) < 2:
        return None
    diffs = [float(current[i]) - float(baseline[i]) for i in ids]
    ci = bootstrap_ci(diffs, seed=seed, iterations=iterations, alpha=alpha)
    return {"mean_delta": sum(diffs) / len(diffs), **(ci or {})}
