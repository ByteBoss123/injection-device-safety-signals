"""Monthly report-spike detection per (firm, failure mode), validated against real FDA recall dates.

Detector: for each firm x category x month, compare the count with the trailing 12-month median
(needs >= 6 prior months). Alert if count >= 10, count >= 3x baseline, and Poisson upper-tail
p < 0.001 with mean = max(median, 1).
Validation: does the same firm have a recall initiated within [-90, +180] days of the alert month?
Compared against the base rate over all firm-months that have any reports.
"""
import json, sys, gzip
import numpy as np, pandas as pd
from scipy.stats import poisson
from ml.common import report_categories, firm_key

ev = pd.read_parquet(sys.argv[1]); OUT = sys.argv[3]
rec = pd.DataFrame([json.loads(l) for l in gzip.open(sys.argv[2])])
ev["firm"] = ev.manufacturer.apply(firm_key)
ev["month"] = pd.to_datetime(ev.date_received, format="%Y%m%d").dt.to_period("M")
ev["doe"] = pd.to_datetime(ev.date_of_event, format="%Y%m%d", errors="coerce")
ev["recv"] = pd.to_datetime(ev.date_received, format="%Y%m%d")
ev["cat"] = ev.product_problems.apply(report_categories)
x = ev.explode("cat").dropna(subset=["cat"])
months = pd.period_range(ev.month.min(), ev.month.max(), freq="M")
last_month = ev.month.max()

alerts = []
for (f, c), g in x.groupby(["firm", "cat"]):
    s = g.groupby("month").size().reindex(months, fill_value=0)
    for i in range(12, len(s)):
        hist = s.iloc[i - 12:i]
        if (hist > 0).sum() < 6:
            continue
        med = float(hist.median()); n = int(s.iloc[i]); mu = max(med, 1.0)
        pval = float(poisson.sf(n - 1, mu))
        if n >= 10 and n >= 3 * mu and pval < 1e-3:
            cell = g[g.month == s.index[i]]
            lag = (cell.recv - cell.doe).dt.days
            alerts.append({"firm": f, "category": c, "month": str(s.index[i]), "count": n, "baseline_median": med,
                           "p_value": pval, "share_event_gt_180d_before_receipt": round(float((lag > 180).mean()), 3),
                           "n_distinct_receipt_days": int(cell.recv.nunique())})
A = pd.DataFrame(alerts)

rec["firm"] = rec.recalling_firm.apply(firm_key)
rec["init"] = pd.to_datetime(rec.event_date_initiated, errors="coerce")
rec = rec[(rec.init >= "2020-01-01")]

def recall_near(firm, month):
    start = month.to_timestamp() - pd.Timedelta(days=90)
    end = month.to_timestamp(how="end") + pd.Timedelta(days=180)
    r = rec[(rec.firm == firm) & rec.init.between(start, end)]
    return len(r), sorted(r.res_event_number.unique().tolist())

# Only months with a full +180-day follow-up window are scored.
cutoff = (last_month.to_timestamp(how="end") - pd.Timedelta(days=180)).to_period("M")
fm = ev[ev.month <= cutoff].groupby(["firm", "month"]).size().reset_index()[["firm", "month"]]
fm["hit"] = [recall_near(f, m)[0] > 0 for f, m in zip(fm.firm, fm.month)]
res = {"months": [str(months[0]), str(last_month)], "scored_through": str(cutoff),
       "recalls_2020_plus": int(len(rec)), "recall_firms": rec.firm.value_counts().to_dict(),
       "maude_firms_top": ev.firm.value_counts().head(15).to_dict(),
       "n_alerts": int(len(A))}
if len(A):
    A["month_p"] = pd.PeriodIndex(A.month, freq="M")
    A["batch_filing_suspect"] = (A.share_event_gt_180d_before_receipt >= 0.5) | (A.n_distinct_receipt_days <= 2)
    near = [recall_near(f, pd.Period(m, "M")) for f, m in zip(A.firm, A.month)]
    A["recalls_in_window"] = [n for n, _ in near]; A["recall_events"] = [e for _, e in near]
    sc = A[A.month_p <= cutoff]
    # firm-month level, so multiple categories in one firm-month don't count twice
    fma = sc.groupby(["firm", "month"]).agg(hit=("recalls_in_window", lambda v: (v > 0).any()),
                                             batch=("batch_filing_suspect", "all")).reset_index()
    res["scored_alert_firm_months"] = int(len(fma))
    res["alert_precision_recall_within_window"] = round(float(fma.hit.mean()), 3) if len(fma) else None
    res["base_rate_all_firm_months"] = round(float(fm.hit.mean()), 3)
    res["lift"] = round(res["alert_precision_recall_within_window"] / res["base_rate_all_firm_months"], 2) if res["base_rate_all_firm_months"] else None
    nb = fma[~fma.batch]
    res["non_batch_alert_firm_months"] = int(len(nb))
    res["non_batch_alert_precision"] = round(float(nb.hit.mean()), 3) if len(nb) else None
    # recall coverage: recalls (firm present in MAUDE) with an alert in the 180 days before initiation
    rf = rec[rec.firm.isin(set(ev.firm)) & (rec.init <= cutoff.to_timestamp(how="end"))]
    cov = [bool(((A.firm == r.firm) & (A.month_p.apply(lambda p: p.to_timestamp()) <= r.init) &
                 (A.month_p.apply(lambda p: p.to_timestamp()) >= r.init - pd.Timedelta(days=180))).any())
           for r in rf.drop_duplicates("res_event_number").itertuples()]
    res["recall_events_firm_in_maude"] = len(cov)
    res["recall_events_preceded_by_alert"] = int(sum(cov))
    A.drop(columns="month_p").to_csv(f"{OUT}/alerts.csv", index=False)
    res["alerts_preview"] = A.drop(columns="month_p").sort_values("count", ascending=False).head(25).to_dict("records")
json.dump(res, open(f"{OUT}/anomaly_metrics.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
