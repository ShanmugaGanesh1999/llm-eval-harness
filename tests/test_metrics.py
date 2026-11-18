import pytest

from evals import metrics as m


def test_recall_none_when_no_relevant():
    assert m.recall_at_k(["a"], [], 5) is None


def test_recall_basic_and_cutoff_and_dupes():
    assert m.recall_at_k(["a", "b", "c"], ["a", "c"], 5) == 1.0
    assert m.recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0
    assert m.recall_at_k(["a", "a", "a", "b"], ["a", "b"], 2) == 1.0  # duplicates do not consume slots
    with pytest.raises(ValueError):
        m.recall_at_k(["a"], ["a"], 0)


def test_mrr():
    assert m.mrr_at_k(["x", "a"], ["a"], 10) == 0.5
    assert m.mrr_at_k(["x"], ["a"], 10) == 0.0
    assert m.mrr_at_k(["a"], [], 10) is None


def test_ndcg():
    assert m.ndcg_at_k(["a", "b"], ["a", "b"], 10) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x", "a"], ["a"], 10) == pytest.approx(1 / __import__("math").log2(3))
    assert m.ndcg_at_k(["x"], ["a"], 10) == 0.0
    assert m.ndcg_at_k([], [], 10) is None


def test_citation_validity():
    assert m.citation_validity(["a"], ["a", "b"]) is True
    assert m.citation_validity(["c"], ["a", "b"]) is False
    assert m.citation_validity([], ["a"]) is None


def test_leakage_count():
    assert m.leakage_count(["g1", "g2"], ["a", "g1"], ["g1", "g2"]) == 2
    assert m.leakage_count([], ["a"]) == 0


def test_abstention_and_balanced_accuracy():
    assert m.abstention_correct(True, True) is True
    assert m.abstention_correct(False, True) is False
    assert m.abstention_correct(None, True) is False
    assert m.abstention_correct(True, None) is None
    # 9 answerable answered correctly, 1 unanswerable answered wrongly: accuracy 0.9, balanced 0.5
    pairs = [(False, False)] * 9 + [(True, False)]
    assert m.balanced_accuracy(pairs) == pytest.approx(0.5)
    assert m.balanced_accuracy([]) is None


def test_answer_correct_variants():
    assert m.answer_correct("You can carry over up to five days.", "5 days") is True
    assert m.answer_correct("Carry over up to 15 days.", "5 days") is False  # token boundary
    assert m.answer_correct("Total is 9,850.00 dollars", "9850") is True  # numeric tolerance
    assert m.answer_correct("Total is 9900", "9850") is False
    assert m.answer_correct("Report to security@acme.example within 1 hour.", None,
                            ["security@acme.example", "1 hour"]) is True
    assert m.answer_correct("Report within 1 hour.", None, ["security@acme.example", "1 hour"]) is False
    assert m.answer_correct("Meals up to $75/day.", None, ["$75"]) is True
    assert m.answer_correct(None, "5 days") is False
    assert m.answer_correct("anything", None) is None


def test_forbidden_content_hits():
    assert m.forbidden_content_hits("Globex allows 10 days", ["10 days", "$75"]) == 1
    assert m.forbidden_content_hits(None, ["x"]) == 0


def test_rows_match_order_insensitive_with_tolerance():
    exp = [{"region": "West", "revenue": 2900}, {"region": "North", "revenue": 2800}]
    assert m.rows_match([{"r": "north", "total": "2800.00"}, {"r": "West", "total": 2900.004}], exp)
    assert not m.rows_match([{"r": "North", "total": 2800}, {"r": "West", "total": 2901}], exp)
    assert not m.rows_match([{"r": "North", "total": 2800}], exp)  # missing row
    # duplicates are a multiset, not a set
    assert not m.rows_match([{"a": 1}, {"a": 1}], [{"a": 1}, {"a": 2}])
    assert m.rows_match([], [])


def test_percentile():
    assert m.percentile([10, 20, 30, 40], 50) == 25
    assert m.percentile([1], 95) == 1
    assert m.percentile([None], 50) is None


def test_bootstrap_deterministic_and_bounded():
    xs = [1, 0, 1, 1, 0, 1, 1, 1]
    a, b = m.bootstrap_ci(xs, seed=7), m.bootstrap_ci(xs, seed=7)
    assert a == b
    assert 0 <= a["low"] <= m.mean(xs) <= a["high"] <= 1
    assert m.bootstrap_ci([1], seed=0) is None


def test_paired_bootstrap_delta():
    cur = {"a": 1, "b": 0, "c": 1, "d": 1}
    base = {"a": 1, "b": 1, "c": 1, "d": 1}
    r = m.paired_bootstrap_delta(cur, base, seed=0)
    assert r["mean_delta"] == pytest.approx(-0.25)
    assert r["low"] <= r["mean_delta"] <= r["high"]
