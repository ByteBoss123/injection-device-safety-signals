"""Citation validation: every claim must cite at least one document retrieved in this run."""
import re

ID_RX = re.compile(r"\b(?:RECALL|510K|MDR|STATS):[A-Za-z0-9\-_]+(?:\.[A-Za-z0-9\-_]+)*")


def validate(answer, retrieved_ids):
    retrieved_ids = set(retrieved_ids)
    kept, dropped = [], []
    for c in answer.get("claims", []):
        cites = [x for x in c.get("citations", []) if x in retrieved_ids]
        bad = [x for x in c.get("citations", []) if x not in retrieved_ids]
        (kept if cites else dropped).append({**c, "citations": cites, "invalid_citations": bad})
    # IDs mentioned inline in prose but never retrieved are also fabrications
    inline_bad = sorted({x for x in ID_RX.findall(answer.get("answer", "")) if x not in retrieved_ids})
    out = {**answer, "claims": kept, "dropped_claims": dropped, "uncited_inline_ids": inline_bad}
    if not kept and not answer.get("insufficient_evidence"):
        out["insufficient_evidence"] = True
        out["answer"] = "Insufficient cited evidence in the retrieved FDA records to answer this."
    return out
