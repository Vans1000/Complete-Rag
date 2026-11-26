import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from common import (add_model_args, load_tokenizer, load_vector_db, load_beir, sample_queries,
                    ensure_ingested, source_of, device_info, collection_name_for, gpu_memory_note)
from metrics import (ndcg_at_k, recall_at_k, mrr_at_k, precision_at_k, dedupe_keep_order,
                     latency_summary, mean)

ALL_MODES = ["dense", "sparse", "hybrid", "full", "sparse_ce"]


def encode_query(tok, q):
    from qdrant_client import models
    vec = tok.TextDenseVector(q, is_query=True)
    dense = None if vec is None else vec.tolist()
    sp = tok.SparseTokens(q)
    idx = sp.nonzero().flatten().cpu().tolist()
    vals = sp[idx].cpu().tolist()
    return dense, models.SparseVector(indices=idx, values=vals)


def run_mode(mode, tok, vdb, q, top_k):
    from qdrant_client import models
    if mode == "full":
        return vdb.Search(tokenizer=tok, query_text=q, top_k=top_k)
    dense, sparse = encode_query(tok, q)
    c, name = vdb.client, vdb.collection_name
    if dense is None and mode in ("dense", "hybrid"):
        raise ValueError(f"mode '{mode}' needs a text embedder (got --text_dense_model None)")
    if mode == "dense":
        return c.query_points(name, query=dense, using="text_dense", limit=top_k, with_payload=True).points
    if mode == "sparse":
        return c.query_points(name, query=sparse, using="text_sparse", limit=top_k, with_payload=True).points
    if mode == "sparse_ce":
        pts = c.query_points(name, query=sparse, using="text_sparse", limit=top_k, with_payload=True).points
        if not pts:
            return pts
        ce = tok.CrossEncoderRanking(q, [p.payload.get("caption", p.payload.get("text", "")) for p in pts])
        return [p for _, p in sorted(zip(ce, pts), key=lambda t: t[0], reverse=True)]
    if mode == "hybrid":
        return c.query_points(
            name,
            prefetch=[models.Prefetch(query=dense, using="text_dense", limit=top_k),
                      models.Prefetch(query=sparse, using="text_sparse", limit=top_k)],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k, with_payload=True,
        ).points
    raise ValueError(mode)


def evaluate(mode, tok, vdb, queries, qrels, top_k, warmup):
    per_query, latencies = [], []
    qids = list(queries)
    for qid in qids[:warmup]:
        run_mode(mode, tok, vdb, queries[qid], top_k)
    for n, qid in enumerate(qids, 1):
        t0 = time.perf_counter()
        points = run_mode(mode, tok, vdb, queries[qid], top_k)
        latencies.append(time.perf_counter() - t0)
        ranked = dedupe_keep_order(source_of(p) for p in points)
        rel = qrels[qid]
        per_query.append({
            "qid": qid, "mode": mode, "ranked": ranked[:20],
            "ndcg@10": ndcg_at_k(ranked, rel, 10), "mrr@10": mrr_at_k(ranked, rel, 10),
            "p@5": precision_at_k(ranked, rel, 5),
            "recall@5": recall_at_k(ranked, rel, 5), "recall@10": recall_at_k(ranked, rel, 10),
            f"recall@{top_k}": recall_at_k(ranked, rel, top_k),
        })
        if n % 50 == 0:
            print(f"  [{mode}] {n}/{len(qids)}  nDCG@10 so far={mean([r['ndcg@10'] for r in per_query]):.4f}"
                  f"{gpu_memory_note()}", flush=True)
    keys = [k for k in per_query[0] if k not in ("qid", "mode", "ranked")]
    summary = {k: mean([r[k] for r in per_query]) for k in keys}
    summary.update(latency_summary(latencies))
    summary["qps"] = len(latencies) / sum(latencies) if latencies else float("nan")
    return summary, per_query


def markdown_table(results, top_k):
    cols = ["ndcg@10", "mrr@10", "p@5", "recall@5", "recall@10", f"recall@{top_k}", "p50_ms", "p95_ms", "qps"]
    lines = ["| mode | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for mode, s in results.items():
        cells = [f"{s[c]:.1f}" if c.endswith("_ms") or c == "qps" else f"{s[c]:.4f}" for c in cols]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="scifact",
                   help="BEIR dataset name on HF hub: scifact, nfcorpus, fiqa, arguana, scidocs, trec-covid ...")
    p.add_argument("--split", default="test")
    p.add_argument("--collection", default=None,
                   help="Qdrant collection (default: bench_<dataset>_<text model>_c<chunk size>)")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    p.add_argument("--top_k", type=int, default=20,
                   help="Chunks retrieved per query (also the rerank depth for 'full'). App default is 5-10.")
    p.add_argument("--max_queries", type=int, default=0, help="Subsample queries (0 = all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=256, help="Ingestion embedding batch size")
    p.add_argument("--reingest", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="benchmark/results")
    add_model_args(p)
    args = p.parse_args()

    collection = args.collection or collection_name_for(args, args.dataset)
    print(f"Loading BEIR/{args.dataset} ...")
    corpus, queries, qrels = load_beir(args.dataset, args.split)
    queries = sample_queries(queries, args.max_queries, args.seed)
    print(f"  corpus={len(corpus)} docs, queries={len(queries)}")

    print("Loading models ...")
    t0 = time.perf_counter()
    tok = load_tokenizer(args)
    model_load_s = time.perf_counter() - t0
    vdb = load_vector_db(args, collection, tok)
    if tok.text_dense_model is None:
        dropped = [m for m in args.modes if m in ("dense", "hybrid")]
        if dropped:
            print(f"No text embedder: skipping modes {dropped}")
        args.modes = [m for m in args.modes if m not in ("dense", "hybrid")]

    ingest_stats = ensure_ingested(tok, vdb, corpus, args.dataset, args.batch_size, force=args.reingest,
                                   chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

    results, all_rows = {}, []
    for mode in args.modes:
        print(f"Evaluating mode '{mode}' ...")
        summary, rows = evaluate(mode, tok, vdb, queries, qrels, args.top_k, args.warmup)
        results[mode] = summary
        all_rows.extend(rows)

    table = markdown_table(results, args.top_k)
    print("\n" + table + "\n")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out / f"retrieval_{args.dataset}_{stamp}"
    report = {
        "dataset": args.dataset, "split": args.split, "collection": collection,
        "n_docs": len(corpus), "n_queries": len(queries), "top_k": args.top_k,
        "models": {k: getattr(args, k) for k in ("text_dense_model", "sparse_model", "cross_encoder_model")},
        "chunking": {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap},
        "rerank_multiplier": args.rerank_multiplier, "rerank_pool": vdb.candidate_pool(args.top_k),
        "env": device_info(), "model_load_s": model_load_s, "ingestion": ingest_stats,
        "results": results,
    }
    base.with_suffix(".json").write_text(json.dumps(report, indent=2))
    with open(f"{base}_per_query.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    base.with_suffix(".md").write_text(
        f"# Retrieval benchmark — BEIR/{args.dataset} ({len(queries)} queries, top_k={args.top_k})\n\n"
        f"Text embedder: {args.text_dense_model} · chunks: {args.chunk_size}/{args.chunk_overlap} · "
        f"rerank pool: {vdb.candidate_pool(args.top_k)} (x{args.rerank_multiplier})  \n"
        f"Device: {report['env'].get('device', '?')}\n\n{table}\n")
    print(f"Saved {base}.json / .md / _per_query.jsonl")


if __name__ == "__main__":
    main()