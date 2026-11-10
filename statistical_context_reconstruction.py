"""Paired Wilcoxon tests versus baseline with Holm correction."""
import csv
from pathlib import Path
from scipy.stats import wilcoxon

BASE = Path("results/context_reconstruction")
INPUT = BASE / "reconstruction_evaluated.csv"
OUTPUT = BASE / "reconstruction_statistical_tests.csv"
METRICS = ["bleu", "meteor", "rouge1", "rouge2", "rougeL", "bertscore_f1"]

def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    m = len(pvalues)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run evaluate_context_reconstruction.py first.")
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    by_condition = {}
    for row in rows:
        by_condition.setdefault(float(row["compression_ratio"]), {})[row["article_id"]] = row

    if 0.0 not in by_condition:
        raise RuntimeError("Baseline condition (0%) is missing.")
    tests = []
    for ratio in sorted(r for r in by_condition if r != 0.0):
        common = sorted(set(by_condition[0.0]) & set(by_condition[ratio]))
        for metric in METRICS:
            baseline = [float(by_condition[0.0][aid][metric]) for aid in common]
            compressed = [float(by_condition[ratio][aid][metric]) for aid in common]
            try:
                stat, p = wilcoxon(baseline, compressed, alternative="two-sided")
            except ValueError:
                stat, p = 0.0, 1.0
            tests.append({
                "compression_ratio": ratio,
                "condition": f"{int(ratio*100)}%",
                "metric": metric,
                "n_articles": len(common),
                "wilcoxon_statistic": float(stat),
                "p_value": float(p)
            })
    adjusted = holm_adjust([t["p_value"] for t in tests])
    for test, p_adj in zip(tests, adjusted):
        test["holm_adjusted_p"] = round(p_adj, 8)
        test["significant_at_0.05"] = p_adj < 0.05
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        writer.writeheader()
        writer.writerows(tests)
    print(f"Statistical tests saved: {OUTPUT}")

if __name__ == "__main__":
    main()
