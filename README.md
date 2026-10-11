# Payroll Anomaly Detection — AI/GenAI edition

**Isolation Forest reached F1 of 0.652 (precision 0.537, recall 0.829) vs. the SQL-rules baseline's F1 of 0.939 (precision 1.00, recall 0.886) on injected anomalies.**

| Detector | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| SQL rules (baseline) | 1.00 | 0.886 | 0.939 | 31 | 0 | 4 |
| Isolation Forest | 0.537 | 0.829 | 0.652 | 29 | 25 | 6 |
| Combined | 0.561 | 0.914 | 0.696 | 32 | 25 | 3 |

Evaluated against `data/ground_truth.json` (employees deliberately seeded with an unexplained salary jump, duplicate payment, or OT outlier). `python eval.py` regenerates this table from `data/anomalies.csv` + ground truth → `eval_report.json`. The ML model trades precision for a few extra true positives the rules miss — on this dataset the rules baseline is still the stronger standalone detector; the model's value is as a secondary signal, not a replacement.

Copied from the Data Analyst version of this project (same SQL rules + Isolation Forest core, same evaluation) with one new layer: an LLM turns each flagged record into a plain-English explanation for a manager, instead of leaving it as a raw rule name + stat.

## LLM explanation layer (new in this edition)

`llm_layer/explain.py` + `api.py` (FastAPI):

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...   # optional — falls back to templated explanations without it
uvicorn api:app --reload
curl localhost:8000/explain/E0001/2024-06
curl localhost:8000/summary?top_n=5   # manager-facing digest of the top N flagged records
```

## Run

```
pip install -r requirements.txt
python run_detection.py
python eval.py
```

Output: `data/anomalies.csv` with columns `employee_id, pay_period, rule, detail` — `rule` is one of `salary_jump`, `duplicate_payment`, `ot_outlier`, or `ml_isolation_forest`.

## Tests

`pytest tests/ -q` — covers feature engineering (salary delta %), that an injected extreme outlier is correctly flagged by the Isolation Forest, and the LLM explanation layer.

## Dashboard

`AnomalyDashboard.pbip` — same PBIP pattern as the Analytics Dashboard project (see its README for the two Power BI Desktop preview flags to enable). Anomalies page: table of flagged records and a bar chart of count by `rule`.

Note: `AnomalyDashboard.SemanticModel/definition/tables/Anomalies.tmdl` hardcodes an absolute path to `data/anomalies.csv` (Power Query's `File.Contents()` requires an absolute path). Update it if you move this project to a different machine or folder.

## Tools

Python, scikit-learn (feature engineering + Isolation Forest), evaluation methodology (precision/recall/F1 against seeded ground truth), Claude API.

**Resume framing:** model, metrics, and baseline comparison — what was built, how it was evaluated, and how it stacks up against a simpler rules-based baseline.
