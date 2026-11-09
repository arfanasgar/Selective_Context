"""
Paired statistical significance analysis for corrected response generation.

Compares each compression condition (20, 35, 50, 65, 80%) against the
0% baseline for BERTScore F1 and ROUGE-1/2/L.

Tests:
- Two-sided paired Wilcoxon signed-rank test
- Holm correction across all metric x compression comparisons (20 tests)

Input:
results/response_generation_corrected/response_evaluated.csv

Outputs:
- response_statistical_tests.csv
- response_statistical_tests_summary.txt
"""
from pathlib import Path
import warnings

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

INPUT = Path("results/response_generation_corrected/response_evaluated.csv")
OUTPUT_CSV = Path("results/response_generation_corrected/response_statistical_tests.csv")
OUTPUT_TXT = Path("results/response_generation_corrected/response_statistical_tests_summary.txt")

METRICS = ["bertscore_f1", "rouge1", "rouge2", "rougeL"]
CONDITIONS = [20, 35, 50, 65, 80]
BASELINE = 0
ALPHA = 0.05


def run_test(baseline, compressed):
    """Run paired Wilcoxon, handling all-zero paired differences safely."""
    differences = compressed.to_numpy() - baseline.to_numpy()

    if len(differences) == 0:
        return float("nan"), 1.0, "No paired observations"

    if (differences == 0).all():
        return 0.0, 1.0, "All paired differences are zero"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = wilcoxon(
                baseline.to_numpy(),
                compressed.to_numpy(),
                alternative="two-sided",
                zero_method="wilcox",
                method="auto"
            )
            return float(result.statistic), float(result.pvalue), "OK"
        except ValueError as exc:
            return float("nan"), 1.0, f"Test issue: {exc}"


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT}. Run response evaluation before this script."
        )

    df = pd.read_csv(INPUT)

    required = {"example_id", "target_compression_percent", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df["target_compression_percent"] = pd.to_numeric(
        df["target_compression_percent"], errors="coerce"
    )

    results = []
    for metric in METRICS:
        baseline_df = df[df["target_compression_percent"] == BASELINE][
            ["example_id", metric]
        ].dropna()

        baseline_series = baseline_df.drop_duplicates(
            subset="example_id"
        ).set_index("example_id")[metric]

        for condition in CONDITIONS:
            condition_df = df[
                df["target_compression_percent"] == condition
            ][["example_id", metric]].dropna()

            condition_series = condition_df.drop_duplicates(
                subset="example_id"
            ).set_index("example_id")[metric]

            paired = pd.concat(
                [
                    baseline_series.rename("baseline"),
                    condition_series.rename("compressed")
                ],
                axis=1,
                join="inner"
            ).dropna()

            statistic, p_value, status = run_test(
                paired["baseline"], paired["compressed"]
            )

            baseline_mean = paired["baseline"].mean() if len(paired) else float("nan")
            compressed_mean = paired["compressed"].mean() if len(paired) else float("nan")
            mean_diff = (
                (paired["compressed"] - paired["baseline"]).mean()
                if len(paired) else float("nan")
            )

            results.append({
                "metric": metric,
                "compression_percent": condition,
                "n_pairs": len(paired),
                "baseline_mean": baseline_mean,
                "compressed_mean": compressed_mean,
                "mean_difference_compressed_minus_baseline": mean_diff,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "test_status": status
            })

    result_df = pd.DataFrame(results)

    # Correct all 20 planned comparisons together (4 metrics x 5 conditions).
    valid = result_df["p_value"].notna()
    result_df["p_adjusted_holm"] = float("nan")
    result_df["significant_after_holm_0_05"] = False

    if valid.any():
        adjusted = multipletests(
            result_df.loc[valid, "p_value"].astype(float).to_numpy(),
            alpha=ALPHA,
            method="holm"
        )
        result_df.loc[valid, "p_adjusted_holm"] = adjusted[1]
        result_df.loc[valid, "significant_after_holm_0_05"] = adjusted[0]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    lines = [
        "RESPONSE GENERATION: STATISTICAL SIGNIFICANCE ANALYSIS",
        "",
        "Test: two-sided paired Wilcoxon signed-rank test",
        "Reference: 0% compression baseline",
        "Multiple-comparison correction: Holm correction across 20 tests",
        f"Significance threshold: alpha = {ALPHA}",
        f"Input rows: {len(df)}",
        f"Unique examples: {df['example_id'].nunique()}",
        "",
        "RESULTS",
        result_df.to_string(index=False),
        "",
        "INTERPRETATION GUIDE",
        "- p_adjusted_holm < 0.05: statistically significant difference from baseline.",
        "- A negative mean difference means the compressed condition scored lower.",
        "- A positive mean difference means the compressed condition scored higher.",
        "- Statistical significance does not by itself establish practical importance.",
        "",
        "LIMITATIONS",
        "- The analysis uses the 20 paired dialogue examples in this project.",
        "- Dialogue examples are constructed from reviewed QA pairs, not naturally collected conversations.",
        "- ROUGE and BERTScore compare generated responses with short reference answers.",
        "- Results should be interpreted as exploratory and not broadly generalised."
    ]
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")

    print("\nRESPONSE GENERATION STATISTICAL TESTS")
    print(result_df.to_string(index=False))
    print(f"\nSaved CSV: {OUTPUT_CSV}")
    print(f"Saved report: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
