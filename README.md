# Injection Device Safety Signal Analytics

Failure-mode classification, serious-event triage, and report-spike detection on real FDA adverse-event
reports for drug-delivery injection devices (piston syringes, antistick syringes, pen injectors,
epinephrine syringes, syringe needle introducers). Planned next: a cited RAG agent over FDA recall and 510(k)
records on Azure AI Search + Azure OpenAI, with evals and prompt-injection tests.

## Data (all public, openFDA, pulled Nov 24 2026)

| Source | Records | Scope |
|---|---|---|
| MAUDE device adverse events | 39,480 | Product codes FMF, MEG, NSC, PQX, KZH; received 2020-01 to 2026-10 |
| 510(k) premarket notifications | 964 | Same product codes, 1976-2026 |
| Device recalls | 332 | Same product codes (165 initiated since 2020) |

Labels are FDA's own coded `product_problems` terms (252 distinct), grouped into 8 failure modes
(`ml/common.py`). Nonspecific terms ("Insufficient Information", "Defective Device") are left unmapped.

## Results (held-out test set: reports received 2025-01 to 2026-10)

Split is by time: train before 2024-07, validation 2024-07 to 2024-12 (thresholds only), test 2025 onward.
161 test narratives that also appear verbatim in train/validation were removed.

**Failure-mode classifier** (TF-IDF + one-vs-rest logistic regression, 6,825 test reports)

| | Micro-F1 | Macro-F1 |
|---|---|---|
| Model | 0.712 | 0.700 |
| Keyword baseline | 0.429 | 0.466 |
| Model, 108 reports from manufacturers unseen in training | 0.690 | |

Strongest: contamination (F1 0.927), software/electronics (0.888). Weakest: mechanical/actuation
(recall 0.146 at the validation-tuned threshold; PR-AUC 0.424). The keyword baseline beats the model on
leak (0.814 vs 0.749).

**Serious-event triage** (Injury/Death vs Malfunction, 7,734 test reports, 9.8% positive)

- PR-AUC 0.870, ROC-AUC 0.973; precision 0.563 at 90% recall
- Keyword baseline: precision 0.343, recall 0.505
- Robustness: on test reports that never contain "injur", PR-AUC 0.879. Only 3.3% of serious reports use the
  word, so the model is not keying on it.

**Report-spike detector vs recalls: null result**

Monthly counts per manufacturer x failure mode; alert when count >= 10, >= 3x the trailing 12-month median,
and Poisson p < 0.001. Validated against real recall initiation dates for the same firm (window: 90 days
before to 180 days after; months scored through 2026-05).

- 60 alerts; 36 scored firm-months. Recall within window: 5.6% of alert months vs 7.0% base rate (lift 0.8).
- Excluding batch-filing artifacts (>=50% of events more than 180 days before receipt, or <=2 receipt days):
  0 of 27 alert months were followed by a recall.
- 3 of 28 recall events (firms present in MAUDE) were preceded by an alert.
- Why: 87 of 165 recalls since 2020 are one manufacturer, most with root cause "No Marketing Application"
  (regulatory action, not complaint-driven), while the firm with the most spikes (35 alerts) had no recalls.
  Recalls are a poor ground truth for complaint signals in this device class.
- The detector does catch filing artifacts: a 1,435-report labeling/packaging cluster in Oct 2026 with 83% of
  events dated more than 180 days before receipt.

## Known limitations

- Labels reflect manufacturer coding habits. For one pen-injector maker, "Patient Device Interaction Problem"
  is used for leadscrew faults, so the use-interaction class partly learns that firm's vocabulary.
- One manufacturer group accounts for 63% of reports; per-firm results vary.
- Firm matching between MAUDE and recalls uses normalized first tokens (`firm_key`); Embecta is mapped to BD
  (2022 spin-off).

## Layout

```
pull/pull_openfda.py      openFDA pull (search_after paging) -> S3
ml/common.py              failure-mode taxonomy, keyword baseline, firm normalization
ml/eda.py                 dataset profile
ml/train_classifier.py    failure-mode + serious-event models, time split, baselines
ml/anomaly.py             spike detector + recall validation
```
