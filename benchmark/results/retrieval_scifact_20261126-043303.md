# Retrieval benchmark — BEIR/scifact (300 queries, top_k=10)

Text embedder: None · chunks: 1200/200 · rerank pool: 40 (x4)  
Device: mps

| mode | ndcg@10 | mrr@10 | p@5 | recall@5 | recall@10 | recall@10 | p50_ms | p95_ms | qps |
|---|---|---|---|---|---|---|---|---|---|
| sparse | 0.7107 | 0.6854 | 0.1727 | 0.7837 | 0.8127 | 0.8127 | 22.1 | 29.8 | 43.5 |
| full | 0.7534 | 0.7316 | 0.1793 | 0.8071 | 0.8427 | 0.8427 | 833.5 | 1081.3 | 1.2 |
| sparse_ce | 0.7456 | 0.7308 | 0.1767 | 0.7940 | 0.8127 | 0.8127 | 208.9 | 261.0 | 4.6 |
