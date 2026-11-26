import os
import sys
import time
import uuid
import random
from pathlib import Path
from typing import Dict, List, Tuple

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import rag_config
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULTS = {
    "dense_model": "clip-ViT-B-32",
    "sparse_model": "naver/splade-v3",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "vision_rerank_model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
}


def add_model_args(p):
    p.add_argument("--dense_model", default=DEFAULTS["dense_model"], help="CLIP model (images/captions)")
    p.add_argument("--text_dense_model", default=rag_config.TEXT_DENSE_MODEL,
                   help="Text-chunk embedder; 'None' = SPLADE-only text retrieval")
    p.add_argument("--chunk_size", type=int, default=rag_config.CHUNK_SIZE)
    p.add_argument("--chunk_overlap", type=int, default=rag_config.CHUNK_OVERLAP)
    p.add_argument("--rerank_multiplier", type=int, default=rag_config.RERANK_MULTIPLIER,
                   help="Search() candidate pool = top_k * this (1 = old behaviour)")
    p.add_argument("--sparse_model", default=DEFAULTS["sparse_model"])
    p.add_argument("--cross_encoder_model", default=DEFAULTS["cross_encoder_model"])
    p.add_argument("--vision_rerank_model", default=DEFAULTS["vision_rerank_model"])
    p.add_argument("--load_vision", action="store_true",
                   help="Also load the 1B vision-language reranker. Not needed for text-only "
                        "benchmarks (it is only invoked when image points are retrieved).")
    p.add_argument("--qdrant_host", default="localhost")
    p.add_argument("--qdrant_port", type=int, default=6333)

def load_tokenizer(args):
    import Tokenizer as T
    return T.Tokenizer(
        dense_model=args.dense_model, text_dense_model=args.text_dense_model,
        sparse_model=args.sparse_model, cross_encoder_model=args.cross_encoder_model,
        vision_rerank_model=args.vision_rerank_model if args.load_vision else "None",
        caption_model="None",
    )


def collection_name_for(args, dataset_tag):
    import re
    model = "sparseonly" if str(args.text_dense_model) == "None" else \
        re.sub(r"[^a-z0-9]+", "_", args.text_dense_model.split("/")[-1].lower()).strip("_")
    return f"bench_{dataset_tag.replace('-', '_')}_{model}_c{args.chunk_size}"


def load_vector_db(args, collection, tokenizer=None):
    from VectorDatabase import VectorDatabase
    return VectorDatabase(host=args.qdrant_host, port=args.qdrant_port, collection_name=collection,
                          text_dense_dim=getattr(tokenizer, "text_dense_dim", None),
                          rerank_multiplier=args.rerank_multiplier)


def collection_count(vdb) -> int:
    return vdb.client.count(collection_name=vdb.collection_name, exact=True).count

def load_beir(name: str, split: str = "test"):
    from datasets import load_dataset
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels", split=split)

    corpus = {str(r["_id"]): {"title": r.get("title") or "", "text": r["text"] or ""} for r in corpus_ds}
    qrels: Dict[str, Dict[str, int]] = {}
    for r in qrels_ds:
        qid, did, score = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score
    queries = {str(r["_id"]): r["text"] for r in queries_ds if str(r["_id"]) in qrels}
    return corpus, queries, qrels


def load_squad(max_questions: int, seed: int = 0):
    import hashlib
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad", split="validation")
    corpus, questions = {}, []
    for r in ds:
        cid = hashlib.md5(r["context"].encode()).hexdigest()[:16]
        corpus.setdefault(cid, {"title": r["title"].replace("_", " "), "text": r["context"]})
        questions.append({"id": r["id"], "question": r["question"],
                          "answers": list(dict.fromkeys(r["answers"]["text"])), "gold_doc": cid})
    rng = random.Random(seed)
    rng.shuffle(questions)
    return corpus, questions[:max_questions]


def sample_queries(queries: Dict[str, str], n: int, seed: int = 0) -> Dict[str, str]:
    if not n or n >= len(queries):
        return queries
    keys = sorted(queries)
    random.Random(seed).shuffle(keys)
    return {k: queries[k] for k in keys[:n]}

def chunk_corpus(corpus, dataset_tag, chunk_size=rag_config.CHUNK_SIZE,
                 chunk_overlap=rag_config.CHUNK_OVERLAP) -> Tuple[List[str], List[dict]]:
    """Same splitter as HuggingFaceDataset.py."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", " ", ""])
    texts, metas = [], []
    for doc_id, d in corpus.items():
        full = f"{d['title']}\n{d['text']}".strip() if d.get("title") else d["text"]
        if not full:
            continue
        for i, chunk in enumerate(splitter.split_text(full), start=1):
            texts.append(chunk)
            metas.append({"text": chunk, "source_id": doc_id, "chunk_idx": i, "type": "text",
                          "path": f"bench://{dataset_tag}/{doc_id}"})
    return texts, metas


def ingest(tokenizer, vdb, texts, metas, batch_size=256, log_every=10) -> Dict[str, float]:
    t_embed = t_upload = 0.0
    n = len(texts)
    t_start = time.perf_counter()
    for b, start in enumerate(range(0, n, batch_size)):
        bt, bm = texts[start:start + batch_size], metas[start:start + batch_size]
        t0 = time.perf_counter()
        dense = tokenizer.TextDenseVectors(bt)  
        sparse = tokenizer.BatchSparseTokens(bt)
        t1 = time.perf_counter()
        for i in range(len(bt)):
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{bm[i]['path']}_ch{bm[i]['chunk_idx']}"))
            vdb.upload_text(doc_id=pid, dense_vector=None if dense is None else dense[i],
                            sparse_vector=sparse[i], metadata=bm[i])
        t2 = time.perf_counter()
        t_embed += t1 - t0
        t_upload += t2 - t1
        if b % log_every == 0:
            done = min(start + batch_size, n)
            print(f"  ingested {done}/{n} chunks ({done / (t2 - t_start):.1f} chunks/s)", flush=True)
    total = time.perf_counter() - t_start
    return {"chunks": n, "total_s": total, "embed_s": t_embed, "upload_s": t_upload,
            "chunks_per_s": n / total if total else float("nan")}


def ensure_ingested(tokenizer, vdb, corpus, dataset_tag, batch_size, force=False,
                    chunk_size=rag_config.CHUNK_SIZE, chunk_overlap=rag_config.CHUNK_OVERLAP):
    texts, metas = chunk_corpus(corpus, dataset_tag, chunk_size, chunk_overlap)
    have = collection_count(vdb)
    if have >= len(texts) and not force:
        print(f"Collection '{vdb.collection_name}' already has {have} points "
              f"(need {len(texts)}); skipping ingestion. Use --reingest to redo.")
        return None
    print(f"Ingesting {len(corpus)} docs -> {len(texts)} chunks into '{vdb.collection_name}'")
    return ingest(tokenizer, vdb, texts, metas, batch_size=batch_size)


def source_of(point) -> str:
    return str(point.payload.get("source_id") or point.payload.get("path", "").rsplit("/", 1)[-1])


def device_info() -> Dict[str, str]:
    import platform
    info = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["device"] = f"cuda: {torch.cuda.get_device_name(0)}"
        elif torch.backends.mps.is_available():
            info["device"] = "mps"
        else:
            info["device"] = "cpu"
    except Exception:
        pass
    return info
