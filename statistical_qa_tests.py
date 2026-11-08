"""
Paired statistical tests for the QA experiment.
Run from the project root:
    python statistical_qa_tests.py

Compares each compression condition with baseline (0%) for Exact Match and
token-level F1 using paired two-sided Wilcoxon signed-rank tests.
Holm correction is applied across all 10 comparisons.
Writes results/qa/qa_statistical_tests.csv.
"""
from pathlib import Path
import re
import string
from collections import Counter

import pandas as pd
from scipy.stats import wilcoxon

INPUT_PATH = Path("results/qa/qa_results.csv")
OUTPUT_PATH = Path("results/qa/qa_statistical_tests.csv")
RATIOS = [0.20, 0.35, 0.50, 0.65, 0.80]


def normalize_answer(text):
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    pred = normalize_answer(prediction).split()
    ref = normalize_answer(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving original order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running_max = max(running_max, value)
        adjusted[index] = running_max
    return adjusted


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run the QA experiment first."
        )

    df = pd.read_csv(INPUT_PATH).fillna("")
    df["exact_match"] = [
        exact_match(p, r)
        for p, r in zip(df["predicted_answer"], df["reference_answer"])
    ]
    df["token_f1"] = [
        token_f1(p, r)
        for p, r in zip(df["predicted_answer"], df["reference_answer"])
    ]

    rows = []
    for metric in ["exact_match", "token_f1"]:
        wide = df.pivot(
            index="question_id",
            columns="compression_ratio",
            values=metric
        )
        for ratio in RATIOS:
            paired = wide[[0.0, ratio]].dropna()
            baseline = paired[0.0]
            compressed = paired[ratio]
            differences = compressed - baseline

            if len(paired) < 2 or (differences == 0).all():
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(
                    baseline,
                    compressed,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto"
                )
                statistic, p_value = float(result.statistic), float(result.pvalue)

            rows.append({
                "metric": metric,
                "compression_ratio": ratio,
                "n_paired_questions": len(paired),
                "baseline_mean": baseline.mean(),
                "compressed_mean": compressed.mean(),
                "mean_change": differences.mean(),
                "wilcoxon_statistic": statistic,
                "p_value": p_value
            })

    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, adjusted_p in zip(rows, adjusted):
        row["holm_adjusted_p"] = adjusted_p
        row["significant_after_holm_0.05"] = adjusted_p < 0.05

    result_df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    display = result_df.copy()
    for col in [
        "baseline_mean", "compressed_mean", "mean_change",
        "p_value", "holm_adjusted_p"
    ]:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    print("\\nPAIRED WILCOXON TESTS (Holm-adjusted across 10 tests)")
    print(display.to_string(index=False))
    print(f"\\nSaved: {OUTPUT_PATH}")
    print(
        "\\nInterpretation: p < 0.05 after Holm correction indicates a "
        "statistically significant difference from baseline. These tests "
        "measure agreement with the generated reference answers; they do "
        "not independently establish factual correctness."
    )


if __name__ == "__main__":
    main()
