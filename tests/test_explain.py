from llm_layer.explain import explain_anomaly, summarize_top_anomalies


def test_template_explanation_for_salary_jump():
    out = explain_anomaly("E0001", "2024-06", "salary_jump", "+62.0% vs prior period")
    assert "E0001" in out and "2024-06" in out and "62.0%" in out


def test_template_explanation_unknown_rule_falls_back_generically():
    out = explain_anomaly("E0002", "2024-01", "some_new_rule", "detail here")
    assert "E0002" in out and "some_new_rule" in out


def test_summarize_top_anomalies_joins_multiple_explanations():
    anomalies = [
        {"employee_id": "E1", "pay_period": "2024-01", "rule": "duplicate_payment", "detail": "2 payments totaling 100"},
        {"employee_id": "E2", "pay_period": "2024-02", "rule": "ot_outlier", "detail": "z-score 4.0"},
    ]
    out = summarize_top_anomalies(anomalies, top_n=2)
    assert "E1" in out and "E2" in out
