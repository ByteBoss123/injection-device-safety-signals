"""FDA injection-device failure-mode classifier (Kaggle version of ml/train_classifier.py).
Repo: https://github.com/ByteBoss123/injection-device-safety-signals
Time split: train < 2024-07, validation 2024-07..2024-12 (thresholds), test 2025-01..2026-10.
Expected (repo run): micro-F1 0.712 vs 0.429 keyword baseline; serious-event PR-AUC 0.870.
"""

import re

# FDA product-problem terms -> coarse failure-mode categories.
# Nonspecific terms ("Insufficient Information", "Defective Device", ...) are deliberately unmapped.
CATEGORY_TERMS = {
    "leak": ["Leak/Splash", "Fluid/Blood Leak"],
    "contamination": ["Device Contamination with Chemical or Other Material", "Contamination",
                      "Contamination /Decontamination Problem", "Delivered as Unsterile Product"],
    "breakage_material": ["Break", "Crack", "Material Integrity Problem", "Material Separation",
                          "Material Deformation", "Material Puncture/Hole", "Material Twisted/Bent",
                          "Detachment of Device or Device Component", "Defective Component",
                          "Material Too Rigid or Stiff"],
    "dosing_delivery": ["Volume Accuracy Problem", "Failure to Deliver", "Inaccurate Delivery",
                        "Insufficient Flow or Under Infusion", "Complete Blockage"],
    "mechanical_actuation": ["Physical Resistance/Sticking", "Mechanical Jam", "Difficult or Delayed Activation",
                             "Mechanical Problem", "Failure to Align", "Unintended Ejection",
                             "Connection Problem", "Loose or Intermittent Connection"],
    "labeling_packaging": ["Device Markings/Labelling Problem", "Packaging Problem",
                           "Manufacturing, Packaging or Shipping Problem"],
    "software_electronics": ["Computer Software Problem", "Data Problem", "Wireless Communication Problem",
                             "Battery Problem"],
    "use_interaction": ["Patient Device Interaction Problem", "Off-Label Use",
                        "Improper or Incorrect Procedure or Method"],
}
TERM_TO_CAT = {t: c for c, ts in CATEGORY_TERMS.items() for t in ts}
CATEGORIES = list(CATEGORY_TERMS)

# Keyword baseline (what a reviewer might grep for).
KEYWORDS = {
    "leak": r"\bleak", "contamination": r"contamin|foreign|particulate|unsterile|non-?sterile|debris|hair",
    "breakage_material": r"\bbr[oe]ke|crack|broken|detach|separat|bent|deform",
    "dosing_delivery": r"dose|dosing|volume|did not deliver|under ?dos|blockage|occlu|inaccura",
    "mechanical_actuation": r"stuck|stick|jam|resist|plunger|activat|eject|connect",
    "labeling_packaging": r"label|marking|packag|expir|scale|printing",
    "software_electronics": r"\bapp\b|software|bluetooth|wireless|battery|sync|data",
    "use_interaction": r"patient|user error|off-?label|procedure",
}


def report_categories(terms):
    return sorted({TERM_TO_CAT[t] for t in set(terms) if t in TERM_TO_CAT})


_STOP = {"inc", "llc", "ltd", "co", "corp", "company", "the", "medical", "de", "sa", "s", "a", "industrias"}
_ALIASES = {"bd": "becton", "embecta": "becton"}  # Embecta = BD's diabetes-care spin-off (2022)


def firm_key(name):
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t and t not in _STOP]
    if not toks:
        return "unknown"
    return _ALIASES.get(toks[0], toks[0])


import json, re, sys
import numpy as np, pandas as pd, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, average_precision_score,
                             roc_auc_score, precision_recall_curve)

import glob, os
df = pd.read_parquet(glob.glob("/kaggle/input/**/maude_events.parquet", recursive=True)[0])
OUT = "/kaggle/working"
df["text"] = df.event_description.fillna("").str.strip()
df = df[df.text.str.len() >= 20].copy()
df["norm"] = df.text.str.lower().str.replace(r"[^a-z ]+", " ", regex=True).str.split().str.join(" ")
df["cats"] = df.product_problems.apply(report_categories)
df["firm"] = df.manufacturer.apply(firm_key)
d = pd.to_datetime(df.date_received, format="%Y%m%d")
df["split"] = np.where(d < "2024-07-01", "train", np.where(d < "2025-01-01", "val", "test"))
seen = set(df.loc[df.split != "test", "norm"])
n_test_before = int((df.split == "test").sum())
df = df[~((df.split == "test") & df.norm.isin(seen))]
summary = {"n_reports_with_text": int(len(df)), "test_rows_removed_as_seen_text": n_test_before - int((df.split == "test").sum())}

# ---------- failure-mode classifier ----------
lab = df[df.cats.str.len() > 0].copy()
Y = np.array([[c in cs for c in CATEGORIES] for cs in lab.cats], dtype=int)
tr, va, te = (lab.split == s for s in ("train", "val", "test"))
summary["failure_mode_rows"] = {s: int(m.sum()) for s, m in zip(("train", "val", "test"), (tr, va, te))}
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True, max_features=200_000)
Xtr, Xva, Xte = vec.fit_transform(lab.text[tr]), vec.transform(lab.text[va]), vec.transform(lab.text[te])
models, thr, P_te = {}, {}, np.zeros((te.sum(), len(CATEGORIES)))
for j, c in enumerate(CATEGORIES):
    m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xtr, Y[tr.values, j])
    pv = m.predict_proba(Xva)[:, 1]
    p, r, t = precision_recall_curve(Y[va.values, j], pv)
    f = 2 * p * r / np.clip(p + r, 1e-9, None)
    thr[c] = float(t[np.nanargmax(f[:-1])]) if len(t) else 0.5
    models[c] = m
    P_te[:, j] = m.predict_proba(Xte)[:, 1]
Yte = Y[te.values]
pred = (P_te >= np.array([thr[c] for c in CATEGORIES])).astype(int)
base = np.array([[bool(re.search(KEYWORDS[c], t, re.I)) for c in CATEGORIES] for t in lab.text[te]], dtype=int)

def block(yt, yp, scores=None):
    per = {}
    for j, c in enumerate(CATEGORIES):
        per[c] = {"support": int(yt[:, j].sum()), "precision": round(precision_score(yt[:, j], yp[:, j], zero_division=0), 3),
                  "recall": round(recall_score(yt[:, j], yp[:, j], zero_division=0), 3),
                  "f1": round(f1_score(yt[:, j], yp[:, j], zero_division=0), 3)}
        if scores is not None and 0 < yt[:, j].sum() < len(yt):
            per[c]["pr_auc"] = round(average_precision_score(yt[:, j], scores[:, j]), 3)
    return {"micro_f1": round(f1_score(yt, yp, average="micro", zero_division=0), 3),
            "macro_f1": round(f1_score(yt, yp, average="macro", zero_division=0), 3), "per_class": per}

summary["failure_mode_test"] = {"tfidf_logreg": block(Yte, pred, P_te), "keyword_baseline": block(Yte, base)}
# generalization to manufacturers never seen in training
unseen = ~lab.firm[te].isin(set(lab.firm[tr])).values
if unseen.sum() >= 30:
    summary["failure_mode_test_unseen_firms"] = {"n": int(unseen.sum()),
        "micro_f1": round(f1_score(Yte[unseen], pred[unseen], average="micro", zero_division=0), 3),
        "keyword_micro_f1": round(f1_score(Yte[unseen], base[unseen], average="micro", zero_division=0), 3)}
else:
    summary["failure_mode_test_unseen_firms"] = {"n": int(unseen.sum()), "note": "too few to score"}

# ---------- serious-event (Injury/Death) triage ----------
sev = df[df.event_type.isin(["Malfunction", "Injury", "Death"])].copy()
y = sev.event_type.isin(["Injury", "Death"]).astype(int).values
s_tr, s_va, s_te = ((sev.split == s).values for s in ("train", "val", "test"))
v2 = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.6, sublinear_tf=True, max_features=200_000)
X2 = v2.fit_transform(sev.text[s_tr])
m2 = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X2, y[s_tr])
pt = m2.predict_proba(v2.transform(sev.text[s_te]))[:, 1]
kw = sev.text[s_te].str.contains(r"injur|hospital|emergency|surgery|infection|bleed|pain|death|died", case=False).astype(int).values
p, r, t = precision_recall_curve(y[s_te], pt)
r90 = np.where(r >= 0.90)[0]
summary["serious_event_test"] = {
    "n_test": int(s_te.sum()), "positive_rate": round(float(y[s_te].mean()), 4),
    "pr_auc": round(average_precision_score(y[s_te], pt), 3), "roc_auc": round(roc_auc_score(y[s_te], pt), 3),
    "precision_at_90pct_recall": round(float(p[r90].max()), 3) if len(r90) else None,
    "keyword_baseline": {"precision": round(precision_score(y[s_te], kw), 3), "recall": round(recall_score(y[s_te], kw), 3)}}

top = {}
for c in CATEGORIES:
    coef = models[c].coef_[0]; names = np.array(vec.get_feature_names_out())
    top[c] = names[np.argsort(coef)[-12:][::-1]].tolist()
summary["top_terms_per_category"] = summary_top = top
summary["thresholds"] = {c: round(v, 3) for c, v in thr.items()}
joblib.dump({"vec": vec, "models": models, "thr": thr, "sev_vec": v2, "sev_model": m2}, f"{OUT}/models.joblib")
json.dump(summary, open(f"{OUT}/classifier_metrics.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
