"""Prompt-injection screening for retrieved text and output checks.

Retrieved documents are untrusted: a narrative or recall text could contain instructions. Layers:
  1. Documents are passed to the model as JSON data with an explicit 'untrusted' marker.
  2. Instruction-like spans are detected and redacted before the model sees them; the hit is traced.
  3. Side-effecting tools never run on model say-so: they create a pending action for human approval.
  4. Final answers may only cite documents actually retrieved in this run (rag/citations.py).
"""
import re

PATTERNS = [
    r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions|directions|rules)",
    r"disregard\s+(the\s+)?(system|previous|above)",
    r"you\s+are\s+now\s+",
    r"(reveal|print|show)\s+(your|the)\s+(system\s+prompt|instructions)",
    r"(call|invoke|use)\s+the\s+\w+\s+tool",
    r"(?:^|\n|\[)\s*system\s*:\s",
    r"new\s+instructions?\s*:",
    r"(respond|answer|reply)\s+only\s+with",
    r"approve\s+(this|the)\s+(brief|request|action)",
    r"do\s+not\s+(cite|mention|report)",
]
_rx = re.compile("|".join(f"(?:{p})" for p in PATTERNS), re.I)


def scan(text):
    return [m.group(0) for m in _rx.finditer(text or "")]


def redact(text):
    """Replace the sentence containing each instruction-like span."""
    if not scan(text):
        return text, []
    found, out = [], []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _rx.search(sent):
            found.append(sent); out.append("[REDACTED: instruction-like text in source document]")
        else:
            out.append(sent)
    return " ".join(out), found
