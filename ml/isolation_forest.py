"""Isolation Forest anomaly scoring over engineered per-employee-period features."""
import pandas as pd
from sklearn.ensemble import IsolationForest


def engineer_features(pay_periods: pd.DataFrame) -> pd.DataFrame:
    df = pay_periods.sort_values(["employee_id", "pay_period"]).copy()
    df["prev_gross"] = df.groupby("employee_id")["gross_pay"].shift(1)
    df["salary_delta_pct"] = ((df["gross_pay"] - df["prev_gross"]) / df["prev_gross"]).fillna(0)
    df["payments_in_period"] = df.groupby(["employee_id", "pay_period"])["employee_id"].transform("count")
    features = df.groupby(["employee_id", "pay_period"]).agg(
        gross_pay=("gross_pay", "first"),
        overtime_hours=("overtime_hours", "sum"),
        overtime_pay=("overtime_pay", "sum"),
        salary_delta_pct=("salary_delta_pct", "first"),
        payments_in_period=("payments_in_period", "first"),
    ).reset_index()
    return features


def score_anomalies(features: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    cols = ["gross_pay", "overtime_hours", "overtime_pay", "salary_delta_pct", "payments_in_period"]
    scores = model.fit_predict(features[cols])
    raw_scores = model.score_samples(features[cols])
    out = features.copy()
    out["is_anomaly"] = scores == -1
    out["anomaly_score"] = raw_scores
    return out.sort_values("anomaly_score")
