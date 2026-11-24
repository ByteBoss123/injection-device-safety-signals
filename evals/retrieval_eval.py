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


def recall_at(ret, qs, ks=(1, 5, 10)):
    hits = {k: 0 for k in ks}; mrr = 0.0
    for q in qs:
        ids = [h.id for h in ret.search(q["query"], k=max(ks))]
        for k in ks:
            hits[k] += q["target"] in ids[:k]
        mrr += 1 / (ids.index(q["target"]) + 1) if q["target"] in ids else 0
    return {**{f"recall@{k}": round(v / len(qs), 3) for k, v in hits.items()}, "mrr@10": round(mrr / len(qs), 3)}


if __name__ == "__main__":
    docs = load_docs(sys.argv[1]); out = sys.argv[2]
    qs = make_queries(docs, 60, f"{out}/paraphrase_queries.json")
    res = {"n_queries": len(qs),
           "bm25_local": recall_at(BM25Retriever(docs), qs),
           "azure_keyword": recall_at(AzureSearchRetriever(embedder=None), qs),
           "azure_hybrid": recall_at(AzureSearchRetriever(embedder=embedder(dimensions=256)), qs)}
    json.dump(res, open(f"{out}/retrieval_eval.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
