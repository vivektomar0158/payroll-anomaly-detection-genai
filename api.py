import csv
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from llm_layer.explain import explain_anomaly, summarize_top_anomalies

app = FastAPI(title="Payroll Anomaly Detection — AI/GenAI layer")

ANOMALIES_PATH = Path(__file__).parent / "data" / "anomalies.csv"


def _load_anomalies() -> list[dict]:
    if not ANOMALIES_PATH.exists():
        return []
    with open(ANOMALIES_PATH) as f:
        return list(csv.DictReader(f))


class ExplainResponse(BaseModel):
    employee_id: str
    pay_period: str
    rule: str
    detail: str
    explanation: str


@app.get("/explain/{employee_id}/{pay_period}", response_model=list[ExplainResponse])
async def explain(employee_id: str, pay_period: str):
    rows = [a for a in _load_anomalies() if a["employee_id"] == employee_id and a["pay_period"] == pay_period]
    if not rows:
        raise HTTPException(status_code=404, detail="No anomalies found for that employee/period")
    return [
        ExplainResponse(**r, explanation=explain_anomaly(r["employee_id"], r["pay_period"], r["rule"], r["detail"]))
        for r in rows
    ]


@app.get("/summary")
async def summary(top_n: int = 5):
    rows = _load_anomalies()
    return {"top_n": top_n, "summary": summarize_top_anomalies(rows, top_n), "total_anomalies": len(rows)}


@app.get("/health")
async def health():
    return {"status": "ok"}
