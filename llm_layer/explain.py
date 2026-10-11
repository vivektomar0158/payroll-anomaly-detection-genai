"""LLM explanation layer — the AI/GenAI differentiator on top of the Data Analyst
version's pure stats/ML output. Turns a flagged row + its rule/detail into a
manager-facing natural-language explanation, falling back to a templated
explanation when no ANTHROPIC_API_KEY is configured."""
import os

SYSTEM_PROMPT = """You are a payroll analyst explaining why a payroll anomaly-detection system
flagged a specific record. Given the rule that fired and its detail, write ONE concise sentence
a non-technical manager could understand, including the concrete numbers. Do not speculate about
intent or accuse anyone of wrongdoing — describe the data pattern only."""


def explain_anomaly(employee_id: str, pay_period: str, rule: str, detail: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return _call_claude(employee_id, pay_period, rule, detail, api_key)
    return _template_explanation(employee_id, pay_period, rule, detail)


def _call_claude(employee_id, pay_period, rule, detail, api_key) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content":
            f"employee_id={employee_id}, pay_period={pay_period}, rule={rule}, detail={detail}"}],
    )
    return resp.content[0].text.strip()


def _template_explanation(employee_id, pay_period, rule, detail) -> str:
    templates = {
        "salary_jump": f"{employee_id}'s gross pay jumped sharply in {pay_period} ({detail}), well above a normal raise.",
        "duplicate_payment": f"{employee_id} appears to have been paid more than once in {pay_period} ({detail}).",
        "ot_outlier": f"{employee_id} logged unusually high overtime in {pay_period} ({detail}), far outside the typical range.",
        "ml_isolation_forest": f"{employee_id}'s {pay_period} pay record doesn't fit the usual pattern across pay, OT, and payment count ({detail}).",
    }
    return templates.get(rule, f"{employee_id} was flagged in {pay_period} by rule '{rule}': {detail}")


def summarize_top_anomalies(anomalies: list[dict], top_n: int = 5) -> str:
    """One narrative paragraph covering the top N flagged records, for a manager-facing digest."""
    lines = [explain_anomaly(a["employee_id"], a["pay_period"], a["rule"], a["detail"]) for a in anomalies[:top_n]]
    return " ".join(lines)
