"""Create and load the Azure AI Search index (hybrid: keyword + 256-dim vectors).

Env: AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
     AZURE_OPENAI_EMBED_DEPLOYMENT (text-embedding-3-small)
Usage: python -m rag.azure_index corpus.jsonl
"""
import os, sys, time
from rag.retrieval import load_docs, doc_key
from rag.llm import embedder

INDEX = os.environ.get("AZURE_SEARCH_INDEX", "fda-device-safety")
DIM = 256


def create_index():
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (SearchIndex, SimpleField, SearchableField, SearchField,
        SearchFieldDataType, VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile)
    cli = SearchIndexClient(os.environ["AZURE_SEARCH_ENDPOINT"], AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]))
    fields = [
        SimpleField(name="key", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="type", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="product_code", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SearchableField(name="firm", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="date", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="text", type=SearchFieldDataType.String, analyzer_name="en.lucene"),
        SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=DIM, vector_search_profile_name="hnsw"),
    ]
    vs = VectorSearch(algorithms=[HnswAlgorithmConfiguration(name="hnsw-alg")],
                      profiles=[VectorSearchProfile(name="hnsw", algorithm_configuration_name="hnsw-alg")])
    cli.create_or_update_index(SearchIndex(name=INDEX, fields=fields, vector_search=vs))


def upload(docs, batch=200):
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    sc = SearchClient(os.environ["AZURE_SEARCH_ENDPOINT"], INDEX, AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"]))
    emb = embedder(dimensions=DIM)
    n = 0
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        vecs = emb([d["title"] + "\n" + d["text"][:6000] for d in chunk])
        payload = [{"key": doc_key(d["id"]), "doc_id": d["id"], "type": d["type"], "product_code": d.get("product_code") or "",
                    "firm": d.get("firm") or "", "date": d.get("date") or "", "title": d["title"], "text": d["text"],
                    "vector": v} for d, v in zip(chunk, vecs)]
        res = sc.upload_documents(payload)
        n += sum(r.succeeded for r in res)
        time.sleep(0.2)
    return n


if __name__ == "__main__":
    docs = load_docs(sys.argv[1])
    create_index()
    print({"uploaded": upload(docs), "total": len(docs)})
