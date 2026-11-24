# Injection Device Safety Signals

Safety analytics and a cited AI agent over real FDA data for injection drug-delivery devices
(piston syringes, antistick syringes, pen injectors, epinephrine syringes, syringe needle introducers).

1. **ML**: failure-mode classifier and serious-event triage on MAUDE adverse-event narratives, plus a report-spike detector validated against real recalls.
2. **Agent**: tool-using RAG agent on **Azure AI Search** (hybrid keyword + vector) and **Azure OpenAI** (gpt-4.1-mini, text-embedding-3-small) with structured outputs, function calling, citation validation, prompt-injection redaction and a human-approval gate. FastAPI backend, Next.js/TypeScript UI.
3. **Evals**: 110-question golden set built from real records, a planted prompt-injection suite with guard on/off ablation, and a retrieval benchmark.
4. **Lakehouse**: Microsoft Fabric notebook (bronze/silver/gold Delta tables).
5. **PPO**: minimal PPO trainer (value head, KL to a frozen reference, clipped objective) for labeling narratives with a small LM.

## Data (public openFDA, pulled Nov 24 2026)

| Source | Records | Scope |
|---|---|---|
| MAUDE adverse events | 39,480 | Product codes FMF, MEG, NSC, PQX, KZH; received 2020-01 to 2026-10 |
| 510(k) notifications | 964 | Same product codes |
| Device recalls | 332 | Same product codes (165 initiated since 2020) |

Labels are FDA's coded `product_problems` terms (252 distinct) grouped into 8 failure modes (`ml/common.py`).

## 1. ML results (held-out test: reports received 2025-01 to 2026-10)

Time split: train before 2024-07, validation 2024-07 to 2024-12 (thresholds only), test 2025 onward;
161 test narratives that also appear verbatim in train/validation were removed.

| Failure-mode classifier (6,825 test reports) | Micro-F1 | Macro-F1 |
|---|---|---|
| TF-IDF + one-vs-rest logistic regression | 0.712 | 0.700 |
| Keyword baseline | 0.429 | 0.466 |
| Model, 108 reports from manufacturers unseen in training | 0.690 | |

Weakest class: mechanical/actuation (recall 0.146). The keyword baseline beats the model on leak (F1 0.814 vs 0.749).

**Serious-event triage** (Injury/Death vs Malfunction, 7,734 test reports, 9.8% positive): PR-AUC 0.870, ROC-AUC 0.973,
precision 0.563 at 90% recall (keyword baseline: precision 0.343, recall 0.505). On test reports that never contain "injur",
PR-AUC is 0.879.

**Spike detector vs recalls: null result.** Alert months had a same-firm recall within the window 5.6% of the time vs a
7.0% base rate (lift 0.8); 0 of 27 non-batch-artifact alert months were followed by a recall. 87 of 165 recalls since 2020
are from one manufacturer, most with root cause "No Marketing Application" (regulatory, not complaint-driven), so recalls
are a poor ground truth for complaint signals here. The detector does flag filing artifacts (a 1,435-report batch in Oct 2026,
83% of events dated more than 180 days before receipt).

## 2-3. Agent and eval results (Azure AI Search + gpt-4.1-mini, temperature 0)

Corpus: 9,221 documents (332 recalls, 964 510(k)s, 7,925 MAUDE reports from 2025-26) plus 15 planted injection test docs.

| Set | n | v1 | v2 | Scoring |
|---|---|---|---|---|
| Recall firm lookup | 25 | 0.00 | **1.00** | firm string in answer |
| 510(k) decision lookup | 20 | 0.00 | **1.00** | decision date in answer |
| Recall reason | 25 | 0.04 | **0.96** | gpt-4.1-mini judge vs FDA reason text (same model family; treat as indicative) |
| MAUDE counts | 10 | 1.00 | 0.80 | exact count; both v2 misses were HTTP 429 rate-limit failures, 8/8 completed runs correct |
| Unanswerable | 15 | 0.80 | 0.67 | must return insufficient evidence; 1 of 5 v2 misses was a 429 |
| Prompt injection, guard on | 15 | 1.00 | 1.00 | canary not leaked, no brief drafted |
| Prompt injection, guard off | 15 | 1.00 | 1.00 | same, redaction disabled |

- **v1 to v2:** in v1 the model called `get_record` with bare ids (`Z-1983-2020`) while the index stores `RECALL:Z-1983-2020`; every lookup failed and the agent answered "no evidence" rather than guessing. v2 accepts bare ids. The same 110 questions were rerun.
- **Unanswerable regressed** from 0.80 to 0.67, and 4 of 15 runs cited a real record (not checked for relevance). This is the weakest set.
- **Injection:** gpt-4.1-mini resisted all 5 planted attack styles even with the redaction guard off, so this suite does not show that the guard adds anything. The guard redacted 0 of 9,221 real FDA documents (no false positives).
- Latency p50 3.6-8.8 s per question; about 1.4K-3.5K tokens per run.

**Retrieval** (60 paraphrased recall-reason queries without ids; 41 of the 60 target recalls share their exact reason text with other recalls, so any recall in the same reason group counts as relevant; search restricted to recalls):

| | Recall@1 | Recall@10 | MRR@10 |
|---|---|---|---|
| Local BM25 | 0.817 | 0.900 | 0.854 |
| Azure AI Search keyword | 0.583 | 0.917 | 0.679 |
| Azure AI Search hybrid | 0.667 | 0.750 | 0.695 |

Hybrid search did **not** beat lexical search on this corpus; the FDA recall text is short and formulaic.
Without the recall-only filter, all methods score lower because generic queries legitimately match 510(k) and MAUDE records.
Queries were generated once by gpt-4.1-mini and saved, so every retriever sees the same queries.

## 4. Fabric lakehouse

`fabric/maude_lakehouse.py` builds bronze/silver/gold Delta tables (events, exploded problem codes, recalls, monthly
manufacturer x failure-mode counts, detector alerts). Verified with local Spark 3.5 + Delta Lake via `fabric/local_test.py`:
39,480 bronze rows = 39,480 silver events (no loss or duplication), 5,963 gold rows, 332 recalls. **Not yet run inside a
Fabric workspace.**

## 5. PPO

`ppo/ppo_train.py` implements PPO from scratch: generate, terminal reward (F1 vs FDA labels, -0.5 for invalid JSON),
per-token KL penalty, GAE, clipped policy and value losses. A CPU smoke test passes on a tiny random model. **The GPU run
(Qwen2.5-0.5B-Instruct on Kaggle T4, data from `ppo/export_data.py`: 4,000 train / 400 test narratives) has not been run yet,
so there are no PPO results.**

## Run it

```bash
pip install -r requirements.txt
python -m pull.pull_openfda <s3-bucket>                    # or place raw files in data/
python -m ml.train_classifier data/maude_events.parquet out
python -m ml.anomaly data/maude_events.parquet data/recall.jsonl.gz out
python -m rag.corpus data data/corpus.jsonl
export AZURE_SEARCH_ENDPOINT=... AZURE_SEARCH_KEY=... AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... \
       AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1-mini AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
python -m rag.azure_index data/corpus.jsonl
python -m evals.build_golden data/corpus.jsonl data/maude_events.parquet data
python -m evals.run_evals data/golden.json data/maude_events.parquet out/models.joblib out
uvicorn api.main:app --port 8000      # then: cd web && npm install && BACKEND_URL=http://localhost:8000 npm run dev
pytest -q                              # 8 tests: citation checks, injection redaction, approval gate, tool errors, API
```

## Limitations

- Labels reflect manufacturer coding habits; one pen-injector maker codes leadscrew faults as "Patient Device Interaction Problem".
- One manufacturer group accounts for 63% of MAUDE reports.
- Golden questions are template-generated from records; they test grounded lookup and refusal, not open-ended analysis.
- The model judge (recall-reason set) is the same model family as the agent.
- Firm matching between MAUDE and recalls uses normalized first tokens; Embecta is mapped to BD (2022 spin-off).
