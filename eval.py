"""Evaluates the SQL rules and the Isolation Forest against the known-seeded ground
truth (data/ground_truth.json), at the employee level: precision/recall/F1 for each
detector, plus a rules-vs-ML comparison table. Run after run_detection.py."""
import json

import pandas as pd


def load_ground_truth() -> set[str]:
    with open("data/ground_truth.json") as f:
        gt = json.load(f)
    ids = set(gt["jump_employees"]) | set(gt["dup_targets"]) | set(gt["ot_outlier_targets"])
    return {f"E{i:04d}" for i in ids}


def prf(flagged: set[str], truth: set[str]) -> dict:
    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 3),
            "recall": round(recall, 3), "f1": round(f1, 3)}


def main():
    truth = load_ground_truth()
    df = pd.read_csv("data/anomalies.csv")

    rules_flagged = set(df[df.rule != "ml_isolation_forest"].employee_id)
    ml_flagged = set(df[df.rule == "ml_isolation_forest"].employee_id)
    combined_flagged = rules_flagged | ml_flagged

    results = {
        "rules_baseline": prf(rules_flagged, truth),
        "ml_isolation_forest": prf(ml_flagged, truth),
        "combined": prf(combined_flagged, truth),
    }

    print(f"Ground truth: {len(truth)} seeded-anomalous employees\n")
    print(f"{'detector':<20}{'precision':>10}{'recall':>10}{'f1':>10}{'tp':>6}{'fp':>6}{'fn':>6}")
    for name, r in results.items():
        print(f"{name:<20}{r['precision']:>10}{r['recall']:>10}{r['f1']:>10}{r['tp']:>6}{r['fp']:>6}{r['fn']:>6}")

    with open("eval_report.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nWritten to eval_report.json")


if __name__ == "__main__":
    main()
