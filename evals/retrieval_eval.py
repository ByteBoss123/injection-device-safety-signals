"""Retrieval quality without ids in the query: lexical (local BM25, Azure keyword) vs Azure hybrid.

Queries are paraphrases of real recall reasons, generated once by gpt-4o-mini and saved
(evals/paraphrase_queries.json) so every retriever sees identical queries. Target = the source recall.
"""
import json, random, sys
from rag.retrieval import load_docs, BM25Retriever, AzureSearchRetriever
from rag.llm import AzureChat, embedder

SCHEMA = {"name": "p", "strict": True, "schema": {"type": "object", "additionalProperties": False,
          "required": ["query"], "properties": {"query": {"type": "string"}}}}


def make_queries(docs, n, out):
    rnd = random.Random(11); llm = AzureChat()
    recs = [d for d in docs if d["type"] == "recall" and len(d["text"]) > 300]
    qs = []
    for d in rnd.sample(recs, n):
        reason = d["text"].split("Reason: ", 1)[1].split(" Root cause:", 1)[0]
        r = llm([{"role": "user", "content": "Rewrite this FDA recall reason as a short question an analyst might search with. "
                  "Do not copy phrases longer than 3 words, and do not include firm names, lot numbers or ids.\n\n" + reason}],
                response_schema=SCHEMA, temperature=0.3)
        qs.append({"target": d["id"], "query": json.loads(r.content)["query"]})
    json.dump(qs, open(out, "w"), indent=1)
    return qs


def reason_groups(docs):
    """Recalls with identical reason text are equally correct answers to a paraphrase of that reason."""
    key = lambda d: d["text"].split("Reason: ", 1)[1].split(" Root cause:", 1)[0].strip().lower()
    by = {}
    for d in docs:
        if d["type"] == "recall" and "Reason: " in d["text"]:
            by.setdefault(key(d), set()).add(d["id"])
    return {i: g for g in by.values() for i in g}


def recall_at(ret, qs, groups=None, ks=(1, 5, 10)):
    hits = {k: 0 for k in ks}; mrr = 0.0
    for q in qs:
        rel = groups.get(q["target"], {q["target"]}) if groups else {q["target"]}
        ids = [h.id for h in ret.search(q["query"], k=max(ks))]
        for k in ks:
            hits[k] += any(i in rel for i in ids[:k])
        first = next((n for n, i in enumerate(ids) if i in rel), None)
        mrr += 1 / (first + 1) if first is not None else 0
    return {**{f"recall@{k}": round(v / len(qs), 3) for k, v in hits.items()}, "mrr@10": round(mrr / len(qs), 3)}


if __name__ == "__main__":
    docs = load_docs(sys.argv[1]); out = sys.argv[2]
    import os
    qp = f"{out}/paraphrase_queries.json"
    qs = json.load(open(qp)) if os.path.exists(qp) else make_queries(docs, 60, qp)  # reuse saved queries
    groups = reason_groups(docs)
    rets = {"bm25_local": BM25Retriever(docs), "azure_keyword": AzureSearchRetriever(embedder=None),
            "azure_hybrid": AzureSearchRetriever(embedder=embedder(dimensions=256))}
    res = {"n_queries": len(qs),
           "targets_with_duplicate_reason_text": sum(len(groups.get(q["target"], ())) > 1 for q in qs),
           "strict_single_id": {n: recall_at(r, qs) for n, r in rets.items()},
           "same_reason_group": {n: recall_at(r, qs, groups) for n, r in rets.items()}}
    json.dump(res, open(f"{out}/retrieval_eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
