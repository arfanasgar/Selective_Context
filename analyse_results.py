"""
Statistical analysis for Selective Context BBC summarisation project.

Input:
    results/final_metrics.csv

Outputs:
    results/statistical_tests.csv
    results/aggregate_results.csv

Tests:
    Paired Wilcoxon signed-rank tests comparing each compression ratio
    against the uncompressed baseline, with Holm correction.
    Reports paired effect size (rank-biserial correlation).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

INPUT = Path("results/final_metrics.csv")
TESTS_OUT = Path("results/statistical_tests.csv")
AGG_OUT = Path("results/aggregate_results.csv")

METRICS = [
    "rouge1_f",
    "rouge2_f",
    "rougeL_f",
    "bertscore_f1",
    "token_reduction_pct",
    "summary_seconds",
]
QUALITY_METRICS = ["rouge1_f", "rouge2_f", "rougeL_f", "bertscore_f1"]


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max
    return adjusted


def paired_rank_biserial(baseline, compressed):
    """Matched-pairs rank-biserial effect size from non-zero differences."""
    differences = np.asarray(compressed) - np.asarray(baseline)
    differences = differences[np.abs(differences) > 1e-12]
    if len(differences) == 0:
        return 0.0
    ranks = pd.Series(np.abs(differences)).rank(method="average").to_numpy()
    positive = ranks[differences > 0].sum()
    negative = ranks[differences < 0].sum()
    denom = positive + negative
    return float((positive - negative) / denom) if denom else 0.0


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"{INPUT} not found. Run this script from the project root."
        )

    df = pd.read_csv(INPUT)
    required = {"article_id", "compression_ratio", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in final_metrics.csv: {sorted(missing)}")

    df["compression_ratio"] = pd.to_numeric(df["compression_ratio"])
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    # Aggregate means, standard deviations and valid sample counts.
    aggregate = (
        df.groupby("compression_ratio")[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(x) for x in col if str(x) != "").rstrip("_")
        if isinstance(col, tuple) else col
        for col in aggregate.columns
    ]
    aggregate.to_csv(AGG_OUT, index=False)

    baseline = df[df["compression_ratio"] == 0.0].set_index("article_id")
    rows = []

    for metric in METRICS:
        metric_rows = []
        for ratio in sorted(r for r in df["compression_ratio"].dropna().unique() if r > 0):
            compressed = df[df["compression_ratio"] == ratio].set_index("article_id")

            paired = pd.concat(
                [baseline[metric].rename("baseline"),
                 compressed[metric].rename("compressed")],
                axis=1,
                join="inner",
            ).dropna()

            x = paired["baseline"].to_numpy()
            y = paired["compressed"].to_numpy()

            if len(paired) < 3:
                continue

            try:
                result = wilcoxon(
                    x, y,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                p_value = float(result.pvalue)
                statistic = float(result.statistic)
            except ValueError:
                p_value = 1.0
                statistic = 0.0

            metric_rows.append({
                "metric": metric,
                "compression_ratio": ratio,
                "n_pairs": len(paired),
                "baseline_mean": float(np.mean(x)),
                "compressed_mean": float(np.mean(y)),
                "mean_difference_compressed_minus_baseline": float(np.mean(y - x)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_effect": paired_rank_biserial(x, y),
            })

        if metric_rows:
            adjusted = holm_adjust([row["p_value"] for row in metric_rows])
            for row, p_adj in zip(metric_rows, adjusted):
                row["holm_adjusted_p"] = float(p_adj)
                row["significant_at_0_05"] = bool(p_adj < 0.05)
                rows.append(row)

    tests = pd.DataFrame(rows)
    tests.to_csv(TESTS_OUT, index=False)

    print("\n=== Aggregate results ===")
    print(aggregate.to_string(index=False))
    print("\n=== Paired Wilcoxon tests vs baseline ===")
    print(tests.to_string(index=False))
    print(f"\nSaved: {AGG_OUT}")
    print(f"Saved: {TESTS_OUT}")
    print(
        "\nInterpretation: Holm-adjusted p-values account for the five "
        "compression comparisons within each metric. This analysis is "
        "exploratory because the sample contains 29 articles."
    )


if __name__ == "__main__":
    main()
