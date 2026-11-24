"""Pull real openFDA device data for injection drug-delivery devices into S3.

Sources (all public, openFDA):
  - device/event   (MAUDE adverse event reports)
  - device/510k    (premarket notifications)
  - device/recall  (device recalls)

Paging uses openFDA's search_after (Link header), so pulls are not capped at the
25,000 'skip' limit.
"""
import json, re, sys, time, gzip, io
import boto3, requests, pandas as pd

BUCKET = sys.argv[1]
CODES = ["FMF", "MEG", "NSC", "PQX", "KZH"]
CODE_Q = " OR ".join(CODES)
API = "https://api.fda.gov/device"
s3 = boto3.client("s3")


def page_all(endpoint, search, limit=100, max_records=None):  # >100 needs an API key
    url, params = f"{API}/{endpoint}.json", {"search": search, "limit": limit}
    n = 0
    while url:
        for attempt in range(6):
            r = requests.get(url, params=params, timeout=120)
            if r.status_code == 200:
                break
            if r.status_code == 404:
                return
            time.sleep(2 ** attempt)
        r.raise_for_status()
        for rec in r.json().get("results", []):
            yield rec
            n += 1
            if max_records and n >= max_records:
                return
        m = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        url, params = (m.group(1), None) if m else (None, None)
        time.sleep(0.25)


def put_jsonl_gz(key, records):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        for rec in records:
            g.write((json.dumps(rec) + "\n").encode())
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())


def flatten_event(e):
    dev = (e.get("device") or [{}])[0]
    texts = e.get("mdr_text") or []
    by_type = {}
    for t in texts:
        by_type.setdefault(t.get("text_type_code", ""), []).append(t.get("text", ""))
    return {
        "mdr_report_key": e.get("mdr_report_key"),
        "report_number": e.get("report_number"),
        "date_received": e.get("date_received"),
        "date_of_event": e.get("date_of_event"),
        "event_type": e.get("event_type"),
        "product_code": dev.get("device_report_product_code"),
        "generic_name": dev.get("generic_name"),
        "brand_name": dev.get("brand_name"),
        "manufacturer": dev.get("manufacturer_d_name"),
        "product_problems": e.get("product_problems") or [],
        "patient_problems": sorted({p for pt in (e.get("patient") or []) for p in (pt.get("patient_problems") or [])}),
        "event_description": " ".join(by_type.get("Description of Event or Problem", [])),
        "mfr_narrative": " ".join(by_type.get("Additional Manufacturer Narrative", [])),
        "source_type": e.get("source_type") or [],
        "report_source_code": e.get("report_source_code"),
    }


def main():
    log = {}
    ev = [flatten_event(e) for e in page_all(
        "event", f"device.device_report_product_code:({CODE_Q}) AND date_received:[20200101 TO 20261231]")]
    df = pd.DataFrame(ev)
    buf = io.BytesIO(); df.to_parquet(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key="raw/maude_events.parquet", Body=buf.getvalue())
    log["maude_events"] = len(df)

    k510 = list(page_all("510k", f"product_code:({CODE_Q})"))
    put_jsonl_gz("raw/510k.jsonl.gz", k510); log["510k"] = len(k510)

    rec = list(page_all("recall", f"product_code:({CODE_Q})"))
    put_jsonl_gz("raw/recall.jsonl.gz", rec); log["recall"] = len(rec)

    s3.put_object(Bucket=BUCKET, Key="raw/pull_log.json", Body=json.dumps(log, indent=2).encode())
    print(json.dumps(log))


if __name__ == "__main__":
    main()
