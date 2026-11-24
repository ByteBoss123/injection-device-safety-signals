"""Retrievers. AzureSearchRetriever is the production path; BM25Retriever runs locally for tests and as
the lexical baseline in evals."""
import json, os, re
from dataclasses import dataclass, field


@dataclass
class Hit:
    id: str
    title: str
    text: str
    score: float
    meta: dict = field(default_factory=dict)


def load_docs(path):
    return [json.loads(l) for l in open(path)]


_tok = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(s):
    return _tok.findall(s.lower())


class BM25Retriever:
    def __init__(self, docs):
        from rank_bm25 import BM25Okapi
        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d["title"] + " " + d["text"]) for d in docs])

    def search(self, query, k=8, filters=None):
        scores = self.bm25.get_scores(tokenize(query))
        idx = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])
               if _match(self.docs[i], filters)][:k]
        return [Hit(self.docs[i]["id"], self.docs[i]["title"], self.docs[i]["text"], float(scores[i]),
                    {k2: self.docs[i].get(k2) for k2 in ("type", "product_code", "firm", "date")}) for i in idx]

    def get(self, doc_id):
        for d in self.docs:
            if d["id"] == doc_id:
                return Hit(d["id"], d["title"], d["text"], 1.0, {"type": d["type"]})
        return None


def _match(d, filters):
    if not filters:
        return True
    return all(d.get(k) == v for k, v in filters.items() if v)


class AzureSearchRetriever:
    """Hybrid (BM25 + vector) search on an Azure AI Search index built by rag/azure_index.py."""

    def __init__(self, endpoint=None, index=None, key=None, embedder=None):
        from azure.core.credentials import AzureKeyCredential
        from azure.search.documents import SearchClient
        self.client = SearchClient(endpoint or os.environ["AZURE_SEARCH_ENDPOINT"],
                                   index or os.environ.get("AZURE_SEARCH_INDEX", "fda-device-safety"),
                                   AzureKeyCredential(key or os.environ["AZURE_SEARCH_KEY"]))
        self.embedder = embedder

    def search(self, query, k=8, filters=None):
        from azure.search.documents.models import VectorizedQuery
        flt = " and ".join(f"{f} eq '{v.replace(chr(39), chr(39) * 2)}'" for f, v in (filters or {}).items() if v) or None
        vq = [VectorizedQuery(vector=self.embedder([query])[0], k_nearest_neighbors=k, fields="vector")] if self.embedder else None
        res = self.client.search(search_text=query, vector_queries=vq, filter=flt, top=k,
                                 select=["doc_id", "title", "text", "type", "product_code", "firm", "date"])
        return [Hit(r["doc_id"], r["title"], r["text"], float(r["@search.score"]),
                    {k2: r.get(k2) for k2 in ("type", "product_code", "firm", "date")}) for r in res]

    def get(self, doc_id):
        try:
            r = self.client.get_document(key=doc_key(doc_id))
            return Hit(doc_id, r["title"], r["text"], 1.0, {"type": r["type"]})
        except Exception:
            return None


def doc_key(doc_id):
    """Azure Search keys allow letters, digits, _ - = only."""
    return re.sub(r"[^A-Za-z0-9_\-=]", "_", doc_id.replace(":", "__"))
