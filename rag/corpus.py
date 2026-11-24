"""Build the retrieval corpus from the raw openFDA pulls.

Every document carries a stable FDA identifier so answers can cite it:
  recall  -> "RECALL:<product_res_number>" (e.g. Z-1234-2024)
  510(k)  -> "510K:<k_number>"
  MAUDE   -> "MDR:<report_number>"
MAUDE narratives are limited to reports received from 2025-01-01 (the ML test window) to keep the
vector index small enough for the Azure AI Search free tier.
"""
import gzip, json, sys
import pandas as pd


def _clip(s, n):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " ..."


def recall_docs(path):
    for r in (json.loads(l) for l in gzip.open(path)):
        rid = r.get("product_res_number") or f"res{r.get('res_event_number')}"
        text = (f"Recall {rid} (event {r.get('res_event_number')}), initiated {r.get('event_date_initiated')}, "
                f"status {r.get('recall_status')}. Firm: {r.get('recalling_firm')}. Product code {r.get('product_code')}. "
                f"Product: {_clip(r.get('product_description'), 600)} Reason: {_clip(r.get('reason_for_recall'), 1200)} "
                f"Root cause: {r.get('root_cause_description')}. Action: {_clip(r.get('action'), 800)} "
                f"Quantity: {_clip(r.get('product_quantity'), 200)}")
        yield {"id": f"RECALL:{rid}", "type": "recall", "title": f"Recall {rid} - {r.get('recalling_firm')}",
               "text": text, "product_code": r.get("product_code"), "firm": r.get("recalling_firm"),
               "date": r.get("event_date_initiated"), "event_number": str(r.get("res_event_number")),
               "root_cause": r.get("root_cause_description"), "k_numbers": r.get("k_numbers") or []}


def k510_docs(path):
    for k in (json.loads(l) for l in gzip.open(path)):
        text = (f"510(k) {k.get('k_number')}: {k.get('device_name')} from {k.get('applicant')}. Product code "
                f"{k.get('product_code')}. Received {k.get('date_received')}, decision {k.get('decision_description')} "
                f"on {k.get('decision_date')}. Clearance type {k.get('clearance_type')}.")
        yield {"id": f"510K:{k.get('k_number')}", "type": "510k", "title": f"{k.get('k_number')} - {k.get('device_name')}",
               "text": text, "product_code": k.get("product_code"), "firm": k.get("applicant"),
               "date": k.get("decision_date"), "event_number": "", "root_cause": "", "k_numbers": [k.get("k_number")]}


def maude_docs(path, since="20250101"):
    df = pd.read_parquet(path)
    df = df[(df.date_received >= since) & (df.event_description.str.len() >= 20)]
    for r in df.itertuples():
        d = f"{r.date_received[:4]}-{r.date_received[4:6]}-{r.date_received[6:]}"
        text = (f"MDR {r.report_number} received {d}, event type {r.event_type}. Manufacturer {r.manufacturer}, "
                f"brand {r.brand_name}, product code {r.product_code}. FDA problem codes: {', '.join(sorted(set(r.product_problems)))}. "
                f"Narrative: {_clip(r.event_description, 1500)}")
        yield {"id": f"MDR:{r.report_number}", "type": "mdr", "title": f"MDR {r.report_number} - {r.brand_name}",
               "text": text, "product_code": r.product_code, "firm": r.manufacturer, "date": d,
               "event_number": "", "root_cause": "", "k_numbers": []}


def build(raw_dir, out_path):
    docs, seen = [], set()
    for gen in (recall_docs(f"{raw_dir}/recall.jsonl.gz"), k510_docs(f"{raw_dir}/510k.jsonl.gz"),
                maude_docs(f"{raw_dir}/maude_events.parquet")):
        for d in gen:
            if d["id"] in seen:  # recalls can share a product_res_number across pulls; keep first
                continue
            seen.add(d["id"]); docs.append(d)
    with open(out_path, "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    return docs


if __name__ == "__main__":
    ds = build(sys.argv[1], sys.argv[2])
    print(json.dumps(pd.Series([d["type"] for d in ds]).value_counts().to_dict()))
