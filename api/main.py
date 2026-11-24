"""FastAPI service for the safety agent.

POST /ask                     {"question": "..."} -> run (answer, claims, citations, trace, pending action)
GET  /runs/{id}               stored run
POST /runs/{id}/approve       {"approved": bool, "reviewer": "..."} -> finalize or reject a drafted brief
GET  /health
Env: Azure variables (see README), MAUDE_PARQUET, MODELS_PATH.
"""
import os
from dataclasses import asdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rag.agent import SafetyAgent
from rag.llm import AzureChat, embedder
from rag.retrieval import AzureSearchRetriever
from rag.data_tools import make_stats_fn, make_classify_fn

app = FastAPI(title="Injection Device Safety Agent")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])
_agent, RUNS = None, {}


def agent():
    global _agent
    if _agent is None:
        _agent = SafetyAgent(AzureChat(), AzureSearchRetriever(embedder=embedder(dimensions=256)),
                             make_stats_fn(os.environ.get("MAUDE_PARQUET", "data/maude_events.parquet")),
                             make_classify_fn(os.environ.get("MODELS_PATH", "out/models.joblib")),
                             trace_path=os.environ.get("TRACE_PATH", "out/api_traces.jsonl"))
    return _agent


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


class Approval(BaseModel):
    approved: bool
    reviewer: str = Field(default="reviewer", max_length=100)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(body: Ask):
    run = agent().run(body.question)
    RUNS[run.id] = run
    return asdict(run)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    if run_id not in RUNS:
        raise HTTPException(404, "unknown run")
    return asdict(RUNS[run_id])


@app.post("/runs/{run_id}/approve")
def approve(run_id: str, body: Approval):
    run = RUNS.get(run_id)
    if not run or not run.pending_action:
        raise HTTPException(409, "no pending action for this run")
    if run.pending_action.get("approved") is not None:
        raise HTTPException(409, "already decided")
    return asdict(agent().approve(run, body.approved, body.reviewer))
