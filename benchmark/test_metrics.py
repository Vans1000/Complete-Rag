"""Sanity tests for metrics.py.  Run: python benchmark/test_metrics.py"""
import math
from metrics import (ndcg_at_k, recall_at_k, mrr_at_k, precision_at_k, hit_at_k, token_f1,
                     exact_match, answer_contains, percentile)


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_retrieval():
    qrels = {"a": 1, "b": 1}
    assert close(ndcg_at_k(["a", "b", "c"], qrels, 10), 1.0)
    assert close(ndcg_at_k(["c", "d"], qrels, 10), 0.0)
    # relevant at rank 2 only: DCG = 1/log2(3); IDCG = 1 + 1/log2(3)
    exp = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert close(ndcg_at_k(["c", "a"], qrels, 10), exp)
    # graded relevance
    g = {"x": 2, "y": 1}
    exp = ((2 ** 1 - 1) + (2 ** 2 - 1) / math.log2(3)) / ((2 ** 2 - 1) + (2 ** 1 - 1) / math.log2(3))
    assert close(ndcg_at_k(["y", "x"], g, 10), exp)
    assert close(recall_at_k(["a", "c"], qrels, 2), 0.5)
    assert close(mrr_at_k(["c", "d", "b"], qrels, 10), 1 / 3)
    assert close(mrr_at_k(["c", "d", "b"], qrels, 2), 0.0)
    assert close(precision_at_k(["a", "c", "b", "d"], qrels, 4), 0.5)
    assert hit_at_k(["z", "a"], qrels, 2) == 1.0


def test_qa():
    assert exact_match("The Eiffel Tower.", ["eiffel tower"]) == 1.0
    assert close(token_f1("paris france", ["paris"]), 2 * 0.5 * 1 / 1.5)
    assert answer_contains("It was built in 1889 (Source: x)", ["1889"]) == 1.0
    assert answer_contains("It was built in 18890", ["1889"]) == 0.0  # whole-token match only
    assert answer_contains("anything", [""]) == 0.0


def test_percentile():
    assert close(percentile([1, 2, 3, 4], 50), 2.5)
    assert close(percentile([5], 95), 5)


if __name__ == "__main__":
    test_retrieval(); test_qa(); test_percentile()
    print("all metric tests passed")
