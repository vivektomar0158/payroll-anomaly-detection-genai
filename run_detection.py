"""Combines SQL rule flags with Isolation Forest ML scores into one anomalies.csv."""
import sqlite3
import pandas as pd
from ml.isolation_forest import engineer_features, score_anomalies


def main():
    conn = sqlite3.connect("data/payroll.db")
    conn.executescript(open("sql/rules.sql").read())
    conn.commit()

    salary_jumps = pd.read_sql("SELECT * FROM v_salary_jumps", conn)
    dup_payments = pd.read_sql("SELECT * FROM v_duplicate_payments", conn)
    ot_outliers = pd.read_sql("SELECT * FROM v_overtime_outliers", conn)
    pay_periods = pd.read_sql("SELECT * FROM pay_periods", conn)
    conn.close()

    features = engineer_features(pay_periods)
    ml_scored = score_anomalies(features)
    ml_flagged = ml_scored[ml_scored["is_anomaly"]]

    rows = []
    for _, r in salary_jumps.iterrows():
        rows.append({"employee_id": r["employee_id"], "pay_period": r["pay_period"],
                     "rule": "salary_jump", "detail": f"+{r['pct_change']}% vs prior period"})
    for _, r in dup_payments.iterrows():
        rows.append({"employee_id": r["employee_id"], "pay_period": r["pay_period"],
                     "rule": "duplicate_payment", "detail": f"{r['payment_count']} payments totaling {r['total_paid']}"})
    for _, r in ot_outliers.iterrows():
        rows.append({"employee_id": r["employee_id"], "pay_period": r["pay_period"],
                     "rule": "ot_outlier", "detail": f"z-score {r['z_score']}"})
    for _, r in ml_flagged.iterrows():
        rows.append({"employee_id": r["employee_id"], "pay_period": r["pay_period"],
                     "rule": "ml_isolation_forest", "detail": f"anomaly_score={round(r['anomaly_score'],3)}"})

    anomalies = pd.DataFrame(rows)
    anomalies.to_csv("data/anomalies.csv", index=False)
    print(f"salary_jumps={len(salary_jumps)} dup_payments={len(dup_payments)} "
          f"ot_outliers={len(ot_outliers)} ml_flagged={len(ml_flagged)} total_rows={len(anomalies)}")


if __name__ == "__main__":
    main()
