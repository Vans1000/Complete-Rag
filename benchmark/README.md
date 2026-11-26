# Complete-RAG benchmark suite

Drop this `benchmark/` folder into the repo root (next to `backend/`). It imports your real
`Tokenizer`, `VectorDatabase`, `LLM.RAGChat` and ingestion settings, so the numbers reflect the app
as it actually runs, not a re-implementation.

## What it measures

| Script | Question it answers | Data |
|---|---|---|
| `bench_retrieval.py` | How good is each retrieval stage, and what does each one cost? | BEIR (SciFact by default; also NFCorpus, FiQA, ArguAna, SciDocs, TREC-COVID ...) |
| `bench_e2e.py` | Does the full pipeline produce correct answers, and how fast? | SQuAD v1.1 validation, or your own JSONL QA set |
| `test_metrics.py` | Are the metric implementations correct? | built-in cases |

**Retrieval ablation modes** (each one is a stage of `VectorDatabase.Search`):

- `dense`: the text embedder (`--text_dense_model`) only
- `sparse`: SPLADE-v3 `text_sparse` only
- `hybrid`: dense + sparse with Qdrant RRF, no reranking
- `full`: `VectorDatabase.Search()` exactly as the app calls it (RRF + cross-encoder rerank)
- `sparse_ce`: SPLADE candidates reranked by the cross-encoder, with no dense step (tests dropping CLIP for text)

Metrics: nDCG@10 (the standard BEIR headline number), MRR@10, P@5, Recall@5/10/k, p50/p95 latency, QPS,
plus ingestion throughput (chunks/s, split into embedding vs Qdrant upload time).

**End-to-end metrics:**
- `gold_hit@1`, `gold_hit@k`: was the gold passage retrieved?
- `answer_in_context`: does the context sent to the LLM contain the answer?
- `answer_contains`: does the LLM's response contain a gold answer? (the main accuracy number for verbose generative answers)
- `llm_judge_acc`: with `--judge`
- `token_f1` / `exact_match`: shown for reference; they're low by design because the system prompt asks for long, cited answers
- `abstention_rate`: same failure phrases `RAGChat._evaluate_answer` uses
- Latency split into retrieval / generation / total

## Setup

```bash
docker run -p 6333:6333 qdrant/qdrant          # if not already running
cd backend && source .venv/bin/activate         # the project's env already has every dependency
cd ..
python benchmark/test_metrics.py                # quick sanity check
```

## Run

```bash
# Retrieval, ~5k docs / 300 queries, takes a few minutes on a GPU
python benchmark/bench_retrieval.py --dataset scifact

# A harder, larger set
python benchmark/bench_retrieval.py --dataset fiqa --max_queries 300

# Compare configurations (each gets its own collection, ingested once):
python benchmark/bench_retrieval.py --dataset scifact                                # bge-small, 1200-char chunks, pool x4
python benchmark/bench_retrieval.py --dataset scifact --text_dense_model None        # SPLADE-only (+ sparse_ce)
python benchmark/bench_retrieval.py --dataset scifact --text_dense_model BAAI/bge-base-en-v1.5
python benchmark/bench_retrieval.py --dataset scifact --modes full --rerank_multiplier 1   # old pool size
python benchmark/bench_retrieval.py --dataset scifact --chunk_size 3000 --chunk_overlap 500 # old chunking

# End-to-end with OpenAI
OPENAI_API_KEY=... python benchmark/bench_e2e.py --llm_provider openai --llm_model gpt-4o-mini --n 200 --judge

# End-to-end with Ollama or LM Studio
python benchmark/bench_e2e.py --llm_provider ollama --llm_model llama3.1 --n 100
python benchmark/bench_e2e.py --llm_provider openai --llm_base_url http://localhost:1234/v1 --llm_model google/gemma-4-31b

# Your own documents: ingest with main.py as usual, then
python benchmark/bench_e2e.py --qa_file my_qa.jsonl --collection my_collection ...
#   my_qa.jsonl lines: {"question": "...", "answers": ["..."], "gold_doc": "<file name or source_id, optional>"}
```

Every run writes `benchmark/results/*.json` (full report), `*.md` (table), and `*_per_query.jsonl`
(for digging into failures). Corpora go into separate `bench_*` collections and are only ingested
once; pass `--reingest` after changing chunking or embedding models.

Useful flags:
- `--load_vision`: also loads the 1B vision reranker. It's off by default because text-only corpora never call it.
- `--text_dense_model`, `--chunk_size`, `--chunk_overlap`, `--rerank_multiplier`: defaults come from `backend/rag_config.py`.
  Collections are named `bench_<dataset>_<text model>_c<chunk size>`, so configurations never mix.
- `--sparse_model`, `--cross_encoder_model`: compare swapped-in models.
- `--temperature`: generation runs at 0 for reproducibility; the app's default is 0.7.

## Reference points

For context on SciFact nDCG@10, published BEIR results put BM25 at about 0.67, and strong dense or SPLADE
models land around 0.65–0.72. Use those as a rough scale, not exact targets: the chunking here
(3000/500 characters) and the RRF settings differ from the published setups.

## Things found while wiring this up (items 2 and 3 are now fixed in `backend/`)

1. **`RAGChat.query(stream=False)` returns a generator, not a dict.** The function contains `yield`,
   so Python makes the whole function a generator, and the `return response` in the non-stream branch
   never reaches the caller. That breaks `POST /chat` (non-streaming) and `self_correction` with
   `stream=False`. The benchmark works around it by calling `_retrieve_and_generate` directly.
   Fix: split the streaming branch into a separate helper generator and `return` that from `query`.
2. **`VectorDatabase.Search` drops `node_ids` when `tree_level` is set.** Lines 88–91 overwrite the
   filter built from `must_clauses`, so TreeRAG's top-down traversal never actually narrows the search to
   child nodes. Deleting those four lines fixes it.
3. **Expect weak `dense` numbers.** CLIP's text encoder truncates input at 77 tokens (about 300 characters),
   but chunks are up to 3000 characters, so the dense vector only sees the start of each chunk. If
   `dense` scores well below `sparse`, a text embedding model for `text_dense` (e.g. `BAAI/bge-small-en-v1.5`
   at 384 dims, which needs a new collection size) or smaller chunks should help. CLIP can stay
   on the image vectors.
4. In `full` mode the cross-encoder term (`2·sigmoid(score)`, range 0–2) swamps the RRF score
   (about 0.03 max), so the final order is effectively cross-encoder only. Comparing `hybrid` with `full` shows whether
   that's a good thing on your data.
