"""
BERTScore evaluation for the BBC Selective Context experiment.

Reads:
    results/generated_summaries.csv

Writes:
    results/bertscore_results.csv
    results/final_metrics.csv

Uses DistilRoBERTa as a lighter CPU-friendly BERTScore backbone.
Run from the project root:
    python evaluate_bertscore.py
"""

from pathlib import Path
import pandas as pd
from bert_score import score

INPUT_FILE = Path("results/generated_summaries.csv")
OUTPUT_FILE = Path("results/bertscore_results.csv")
MERGED_FILE = Path("results/final_metrics.csv")

MODEL_TYPE = "distilroberta-base"
BATCH_SIZE = 2
DEVICE = "cpu"


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {INPUT_FILE}. Run from the project root."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "article_id",
        "compression_ratio",
        "reference_description",
        "generated_summary",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df["reference_description"] = df["reference_description"].fillna("").astype(str)
    df["generated_summary"] = df["generated_summary"].fillna("").astype(str)

    valid = (
        df["reference_description"].str.strip().ne("")
        & df["generated_summary"].str.strip().ne("")
    )

    if not valid.all():
        print(f"Skipping {(~valid).sum()} rows with empty reference or summary.")

    work = df.loc[valid].copy()
    if work.empty:
        raise ValueError("No rows have both a reference and a generated summary.")

    references = work["reference_description"].tolist()
    predictions = work["generated_summary"].tolist()

    print(f"Evaluating {len(work)} summaries with {MODEL_TYPE} on {DEVICE}.")
    print("The first run downloads the evaluation model; later runs use the cache.")

    precision, recall, f1 = score(
        predictions,
        references,
        model_type=MODEL_TYPE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        lang="en",
        verbose=True,
        rescale_with_baseline=False,
    )

    work["bertscore_precision"] = precision.tolist()
    work["bertscore_recall"] = recall.tolist()
    work["bertscore_f1"] = f1.tolist()

    score_columns = [
        "article_id",
        "title",
        "compression_ratio",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]
    work[score_columns].to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Merge BERTScore onto the existing ROUGE/efficiency metrics.
    metrics_file = Path("results/experiment_results.csv")
    if metrics_file.exists():
        metrics = pd.read_csv(metrics_file)
        merged = metrics.merge(
            work[score_columns],
            on=["article_id", "title", "compression_ratio"],
            how="left",
            validate="one_to_one",
        )
        merged.to_csv(MERGED_FILE, index=False, encoding="utf-8-sig")
        print(f"Saved combined ROUGE, BERTScore and efficiency metrics: {MERGED_FILE}")

    print(f"Saved BERTScore details: {OUTPUT_FILE}")
    print("\nMean BERTScore F1 by compression condition:")
    print(
        work.groupby("compression_ratio")["bertscore_f1"]
        .mean()
        .round(4)
        .to_string()
    )
    print(
        "\nMethod note: BERTScore uses distilroberta-base, "
        "without baseline rescaling, on CPU."
    )


if __name__ == "__main__":
    main()
