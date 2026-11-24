"""Kaggle Playground S6E11 - Predicting Credit Card Fraud.
5-fold stratified CV; LightGBM + XGBoost; out-of-fold ROC-AUC decides the blend weight.
Usage: python train.py <dir with train.csv/test.csv/sample_submission.csv> <out_dir>
"""
import json, sys, time
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D, OUT = sys.argv[1], sys.argv[2]
tr, te = pd.read_csv(f"{D}/train.csv"), pd.read_csv(f"{D}/test.csv")
y = tr.pop("is_fraud").astype(int).values
ids = te["id"].values
X = pd.concat([tr, te]).drop(columns="id").reset_index(drop=True)
cats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
for c in cats:
    X[c] = X[c].astype("category")
for c in X.columns:
    if X[c].dtype == bool:
        X[c] = X[c].astype(int)
# simple, leak-free extras: amount relative to balance, frequency of each category level
X["amt_to_balance"] = X.amount_usd / (X.account_balance_usd.abs() + 1)
for c in cats:
    X[c + "_freq"] = X[c].map(X[c].value_counts(normalize=True)).astype(float)
Xtr, Xte = X.iloc[: len(tr)], X.iloc[len(tr):]

skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = {"lgb": np.zeros(len(tr)), "xgb": np.zeros(len(tr))}
pred = {"lgb": np.zeros(len(te)), "xgb": np.zeros(len(te))}
log = {"rows": len(tr), "fraud_rate": float(y.mean()), "features": X.shape[1], "folds": []}
for k, (a, b) in enumerate(skf.split(Xtr, y)):
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, min_child_samples=100,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, verbose=-1)
    m.fit(Xtr.iloc[a], y[a], eval_set=[(Xtr.iloc[b], y[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof["lgb"][b] = m.predict_proba(Xtr.iloc[b])[:, 1]; pred["lgb"] += m.predict_proba(Xte)[:, 1] / 5
    x = xgb.XGBClassifier(n_estimators=5000, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.7,
                          min_child_weight=5, eval_metric="auc", early_stopping_rounds=200, tree_method="hist",
                          enable_categorical=True, n_jobs=-1)
    x.fit(Xtr.iloc[a], y[a], eval_set=[(Xtr.iloc[b], y[b])], verbose=False)
    oof["xgb"][b] = x.predict_proba(Xtr.iloc[b])[:, 1]; pred["xgb"] += x.predict_proba(Xte)[:, 1] / 5
    f = {"fold": k, "lgb_auc": round(roc_auc_score(y[b], oof["lgb"][b]), 5), "xgb_auc": round(roc_auc_score(y[b], oof["xgb"][b]), 5),
         "lgb_iters": int(m.best_iteration_), "xgb_iters": int(x.best_iteration), "sec": round(time.time() - t0)}
    log["folds"].append(f); print(f, flush=True)

best_w, best = 0.0, -1
for w in np.linspace(0, 1, 21):
    s = roc_auc_score(y, w * oof["lgb"] + (1 - w) * oof["xgb"])
    if s > best:
        best_w, best = w, s
log.update({"oof_auc_lgb": round(roc_auc_score(y, oof["lgb"]), 5), "oof_auc_xgb": round(roc_auc_score(y, oof["xgb"]), 5),
            "blend_weight_lgb": round(float(best_w), 2), "oof_auc_blend": round(best, 5)})
pd.DataFrame({"id": ids, "is_fraud": best_w * pred["lgb"] + (1 - best_w) * pred["xgb"]}).to_csv(f"{OUT}/submission.csv", index=False)
json.dump(log, open(f"{OUT}/cv_log.json", "w"), indent=1)
print(json.dumps({k: v for k, v in log.items() if k != "folds"}))
