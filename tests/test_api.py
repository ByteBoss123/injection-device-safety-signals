import json
from fastapi.testclient import TestClient
import api.main as m
from rag.agent import SafetyAgent
from rag.llm import ScriptedChat, ChatResult, ToolCall
from rag.retrieval import BM25Retriever
from tests.test_agent import DOCS


def test_ask_then_reject_brief():
    script = [ChatResult(None, [ToolCall("1", "search_fda_records", {"query": "barrel cracks"})]),
              ChatResult(None, [ToolCall("2", "draft_safety_brief", {"title": "t", "summary": "s", "citations": ["RECALL:Z-0001-2024"]})]),
              ChatResult("ok", []),
              ChatResult(json.dumps({"answer": "a", "insufficient_evidence": False,
                                     "claims": [{"text": "c", "citations": ["RECALL:Z-0001-2024"]}]}), [])]
    m._agent = SafetyAgent(ScriptedChat(script), BM25Retriever(DOCS))
    c = TestClient(m.app)
    r = c.post("/ask", json={"question": "draft a brief on cracks"}).json()
    assert r["status"] == "awaiting_approval"
    d = c.post(f"/runs/{r['id']}/approve", json={"approved": False, "reviewer": "qa"}).json()
    assert d["pending_action"]["approved"] is False
    assert c.post(f"/runs/{r['id']}/approve", json={"approved": True}).status_code == 409
    assert c.post("/runs/doesnotexist/approve", json={"approved": True}).status_code == 409
