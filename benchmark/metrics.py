"""Pure-Python IR / QA metrics. No third-party deps so they can be unit-tested anywhere."""
import math
import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Sequence

def dcg(gains: Sequence[float]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: List[str], qrels: Dict[str, int], k: int) -> float:
    gains = [qrels.get(d, 0) for d in ranked[:k]]
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: List[str], qrels: Dict[str, int], k: int) -> float:
    relevant = {d for d, r in qrels.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: List[str], qrels: Dict[str, int], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if qrels.get(d, 0) > 0) / k


def mrr_at_k(ranked: List[str], qrels: Dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if qrels.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(ranked: List[str], qrels: Dict[str, int], k: int) -> float:
    return 1.0 if any(qrels.get(d, 0) > 0 for d in ranked[:k]) else 0.0


def dedupe_keep_order(ids: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: List[str]) -> float:
    p = normalize_answer(pred)
    return float(any(p == normalize_answer(g) for g in golds))


def token_f1(pred: str, golds: List[str]) -> float:
    best = 0.0
    p_toks = normalize_answer(pred).split()
    for g in golds:
        g_toks = normalize_answer(g).split()
        common = Counter(p_toks) & Counter(g_toks)
        same = sum(common.values())
        if same == 0:
            continue
        prec, rec = same / len(p_toks), same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def answer_contains(pred: str, golds: List[str]) -> float:
    """1 if any normalised gold answer appears inside the normalised response.
    The right 'accuracy' proxy for verbose, citation-heavy generative answers."""
    p = f" {normalize_answer(pred)} "
    return float(any(f" {normalize_answer(g)} " in p for g in golds if normalize_answer(g)))

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_summary(seconds: List[float]) -> Dict[str, float]:
    ms = [s * 1000 for s in seconds]
    return {
        "mean_ms": sum(ms) / len(ms) if ms else float("nan"),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "max_ms": max(ms) if ms else float("nan"),
    }


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")
