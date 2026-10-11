import pandas as pd
from ml.isolation_forest import engineer_features, score_anomalies


def test_engineer_features_computes_salary_delta():
    df = pd.DataFrame({
        "employee_id": ["E1", "E1", "E2"],
        "pay_period": ["2024-01", "2024-02", "2024-01"],
        "gross_pay": [1000, 2000, 1500],
        "overtime_hours": [2, 2, 5],
        "overtime_pay": [30, 30, 75],
        "deductions": [200, 400, 300], "bonus": [0, 0, 0], "net_pay": [800, 1600, 1200],
    })
    feats = engineer_features(df)
    e1_feb = feats[(feats.employee_id == "E1") & (feats.pay_period == "2024-02")].iloc[0]
    assert e1_feb["salary_delta_pct"] == 1.0  # doubled


def test_score_anomalies_flags_outlier_row():
    rows = []
    for i in range(30):
        rows.append({"employee_id": f"E{i}", "pay_period": "2024-01", "gross_pay": 100, "overtime_hours": 1,
                     "overtime_pay": 15, "deductions": 20, "bonus": 0, "net_pay": 80})
    rows.append({"employee_id": "E_OUTLIER", "pay_period": "2024-01", "gross_pay": 100000, "overtime_hours": 80,
                 "overtime_pay": 5000, "deductions": 20, "bonus": 0, "net_pay": 80})
    df = pd.DataFrame(rows)
    feats = engineer_features(df)
    scored = score_anomalies(feats, contamination=0.05)
    outlier_row = scored[scored.employee_id == "E_OUTLIER"].iloc[0]
    assert outlier_row["is_anomaly"] == True
