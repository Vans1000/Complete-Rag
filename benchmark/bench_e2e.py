import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from common import (add_model_args, load_tokenizer, load_vector_db, load_squad, ensure_ingested,
                    source_of, device_info, collection_name_for)
from metrics import answer_contains, token_f1, exact_match, latency_summary, mean, normalize_answer

ABSTAIN = ["does not contain", "does not have", "no information", "i am sorry", "i apologize",
           "unable to find", "no relevant context", "not found in", "cannot answer", "don't have access"]

JUDGE_PROMPT = """You are grading a question-answering system.
Question: {q}
Gold answer(s): {gold}
System response: {resp}

Does the system response correctly answer the question, consistent with a gold answer?
Ignore citations, formatting and extra detail. Reply with exactly one word: CORRECT or INCORRECT."""


def build_llm(provider, model, base_url, temperature):
    from LLM import OpenAILLM, OllamaLLM
    if provider == "openai":
        llm = OpenAILLM(model=model, base_url=base_url)
    else:
        llm = OllamaLLM(model=model, base_url=base_url or "http://localhost:11434")
    orig = llm.chat
    llm.chat = lambda messages, stream=False, **kw: orig(messages, stream=stream, **{"temperature": temperature, **kw})
    return llm


def load_qa_file(path):
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            if line.strip():
                r = json.loads(line)
                ans = r.get("answers") or ([r["answer"]] if "answer" in r else [])
                out.append({"id": str(r.get("id", i)), "question": r["question"],
                            "answers": ans, "gold_doc": r.get("gold_doc")})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=200, help="Number of SQuAD questions to sample")
    p.add_argument("--qa_file", default=None, help="Custom JSONL QA set instead of SQuAD")
    p.add_argument("--collection", default=None, help="Qdrant collection (default: bench_squad)")
    p.add_argument("--top_k", type=int, default=5, help="RAGChat default is 5")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--reingest", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm_provider", choices=["openai", "ollama"], default="openai")
    p.add_argument("--llm_model", default="gpt-4o-mini")
    p.add_argument("--llm_base_url", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--judge", action="store_true", help="Also grade answers with an LLM judge")
    p.add_argument("--judge_provider", choices=["openai", "ollama"], default=None, help="Defaults to --llm_provider")
    p.add_argument("--judge_model", default=None, help="Defaults to --llm_model")
    p.add_argument("--judge_base_url", default=None)
    p.add_argument("--out_dir", default="benchmark/results")
    add_model_args(p)
    args = p.parse_args()

    from LLM import RAGChat

    print("Loading models ...")
    tok = load_tokenizer(args)

    if args.qa_file:
        collection = args.collection or "documents"
        questions = load_qa_file(args.qa_file)
        vdb = load_vector_db(args, collection, tok)
        ingest_stats = None
        tag = Path(args.qa_file).stem
    else:
        collection = args.collection or collection_name_for(args, "squad")
        print("Loading SQuAD v1.1 validation ...")
        corpus, questions = load_squad(args.n, args.seed)
        print(f"  corpus={len(corpus)} paragraphs, questions={len(questions)}")
        vdb = load_vector_db(args, collection, tok)
        ingest_stats = ensure_ingested(tok, vdb, corpus, "squad", args.batch_size, force=args.reingest,
                                       chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        tag = "squad"

    llm = build_llm(args.llm_provider, args.llm_model, args.llm_base_url, args.temperature)
    try:
        llm.chat([{"role": "user", "content": "Reply with OK."}], stream=False, max_tokens=5)
    except Exception as e:
        where = args.llm_base_url or getattr(llm, "base_url", "?")
        raise SystemExit(
            f"\nLLM preflight failed: {e!r}\n  provider={args.llm_provider} model={args.llm_model} endpoint={where}\n"
            "  Is the server running? For LM Studio / mlx_lm / vLLM pass --llm_base_url http://localhost:<port>/v1\n"
            "  (check OPENAI_BASE_URL too: it is used when --llm_base_url is not given).")
    judge = None
    if args.judge:
        judge = build_llm(args.judge_provider or args.llm_provider, args.judge_model or args.llm_model,
                          args.judge_base_url if args.judge_base_url else args.llm_base_url, 0.0)

    probe = {}
    orig_search = vdb.Search

    def timed_search(*a, **kw):
        t0 = time.perf_counter()
        res = orig_search(*a, **kw)
        probe["search_s"] = probe.get("search_s", 0.0) + time.perf_counter() - t0
        probe.setdefault("retrieved", []).extend(source_of(r) for r in res)
        return res
    vdb.Search = timed_search

    orig_gen = llm.generate_with_context

    def timed_gen(*a, **kw):
        t0 = time.perf_counter()
        out = orig_gen(*a, **kw)
        probe["gen_s"] = time.perf_counter() - t0
        return out
    llm.generate_with_context = timed_gen

    rag = RAGChat(llm=llm, vector_db=vdb, tokenizer=tok)

    rows = []
    for i, q in enumerate(questions, 1):
        probe.clear()
        rag.clear_history()  
        t0 = time.perf_counter()
        try:
            result, sources, context = rag._retrieve_and_generate(
                q["question"], q["question"], False, False, args.top_k, False, False)
            answer = result.get("answer", "") if isinstance(result, dict) else str(result)
            err = None
        except Exception as e:  # keep going; count as failure
            answer, context, err = "", "", repr(e)
        total_s = time.perf_counter() - t0

        retrieved = list(dict.fromkeys(probe.get("retrieved", [])))
        gold = q.get("gold_doc")
        row = {
            "id": q["id"], "question": q["question"], "answers": q["answers"], "response": answer,
            "error": err,
            "retrieved": retrieved[:args.top_k],
            "gold_hit@k": (float(gold in retrieved[:args.top_k]) if gold else None),
            "gold_hit@1": (float(bool(retrieved) and retrieved[0] == gold) if gold else None),
            "answer_in_context": answer_contains(context, q["answers"]),
            "context_chars": len(context),
            "contains": answer_contains(answer, q["answers"]),
            "f1": token_f1(answer, q["answers"]),
            "em": exact_match(answer, q["answers"]),
            "abstained": float(any(s in answer.lower() for s in ABSTAIN)),
            "search_s": probe.get("search_s", float("nan")),
            "gen_s": probe.get("gen_s", float("nan")),
            "total_s": total_s,
        }
        if judge and answer:
            verdict = judge.chat([{"role": "user", "content": JUDGE_PROMPT.format(
                q=q["question"], gold=" | ".join(q["answers"]), resp=answer[:6000])}], stream=False)
            row["judge"] = float(normalize_answer(str(verdict)).startswith("correct"))
        rows.append(row)
        if i % 10 == 0 or i == len(questions):
            print(f"  {i}/{len(questions)}  contains={mean([r['contains'] for r in rows]):.3f}  "
                  f"ctx_has_answer={mean([r['answer_in_context'] for r in rows]):.3f}  "
                  f"avg_total={mean([r['total_s'] for r in rows]):.2f}s", flush=True)

    ok = [r for r in rows if r["error"] is None]
    searched = [r for r in rows if r["retrieved"]]

    def m(key, pool=None):
        vals = [r[key] for r in (ok if pool is None else pool) if r.get(key) is not None]
        return mean(vals) if vals else None

    summary = {
        "n": len(rows), "errors": len(rows) - len(ok),
        "gold_hit@1": m("gold_hit@1", searched), f"gold_hit@{args.top_k}": m("gold_hit@k", searched),
        "answer_in_context": m("answer_in_context"),
        "answer_contains": m("contains"), "token_f1": m("f1"), "exact_match": m("em"),
        "abstention_rate": m("abstained"), "llm_judge_acc": m("judge") if judge else None,
        "avg_context_chars": m("context_chars"),
        "latency_retrieval": latency_summary([r["search_s"] for r in searched]),
        "latency_generation": latency_summary([r["gen_s"] for r in ok]),
        "latency_total": latency_summary([r["total_s"] for r in ok]),
    }

    def fmt(v):
        return "n/a" if v is None else f"{v:.4f}"
    md = [f"# End-to-end benchmark — {tag} ({len(rows)} questions, top_k={args.top_k})",
          f"LLM: {args.llm_provider}/{args.llm_model} (temperature {args.temperature})  ",
          f"Text embedder: {args.text_dense_model} · chunks {args.chunk_size} · rerank x{args.rerank_multiplier}  ",
          f"Device: {device_info().get('device', '?')}", "",
          "| metric | value |", "|---|---|"]
    for k in ["gold_hit@1", f"gold_hit@{args.top_k}", "answer_in_context", "answer_contains",
              "llm_judge_acc", "token_f1", "exact_match", "abstention_rate"]:
        md.append(f"| {k} | {fmt(summary[k])} |")
    md += ["", "| stage | mean ms | p50 ms | p95 ms |", "|---|---|---|---|"]
    for stage in ["retrieval", "generation", "total"]:
        s = summary[f"latency_{stage}"]
        md.append(f"| {stage} | {s['mean_ms']:.0f} | {s['p50_ms']:.0f} | {s['p95_ms']:.0f} |")
    md_text = "\n".join(md)
    print("\n" + md_text + "\n")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = out / f"e2e_{tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    base.with_suffix(".json").write_text(json.dumps({
        "dataset": tag, "collection": collection, "top_k": args.top_k,
        "text_dense_model": args.text_dense_model, "chunk_size": args.chunk_size,
        "rerank_multiplier": args.rerank_multiplier,
        "llm": {"provider": args.llm_provider, "model": args.llm_model, "temperature": args.temperature},
        "env": device_info(), "ingestion": ingest_stats, "summary": summary}, indent=2))
    base.with_suffix(".md").write_text(md_text + "\n")
    with open(f"{base}_per_question.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {base}.json / .md / _per_question.jsonl")


if __name__ == "__main__":
    main()
