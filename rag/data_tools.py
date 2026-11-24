"""Data-backed tools for the agent: MAUDE counts and the trained classifiers."""
import numpy as np, pandas as pd, joblib
from ml.common import report_categories, CATEGORIES


def make_stats_fn(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["date_received", "product_code", "manufacturer", "product_problems"])
    df["year"] = df.date_received.str[:4]
    df["cats"] = df.product_problems.apply(report_categories)

    def maude_stats(product_code="", manufacturer="", failure_mode=""):
        d = df
        if product_code:
            d = d[d.product_code == product_code.upper()]
        if manufacturer:
            d = d[d.manufacturer.str.contains(manufacturer, case=False, na=False, regex=False)]
        if failure_mode:
            d = d[d.cats.apply(lambda c: failure_mode in c)]
        return {"source": "openFDA MAUDE, received 2020-01..2026-10", "filters":
                {"product_code": product_code, "manufacturer": manufacturer, "failure_mode": failure_mode},
                "total": int(len(d)), "by_year": d.year.value_counts().sort_index().to_dict()}
    return maude_stats


def make_classify_fn(model_path):
    m = joblib.load(model_path)

    def classify_narrative(text):
        X = m["vec"].transform([text])
        probs = {c: float(m["models"][c].predict_proba(X)[0, 1]) for c in CATEGORIES}
        return {"failure_modes": [c for c in CATEGORIES if probs[c] >= m["thr"][c]],
                "probabilities": {c: round(p, 3) for c, p in probs.items()},
                "serious_event_probability": round(float(m["sev_model"].predict_proba(m["sev_vec"].transform([text]))[0, 1]), 3),
                "model": "tfidf+logreg, trained on MAUDE reports received before 2024-07"}
    return classify_narrative
