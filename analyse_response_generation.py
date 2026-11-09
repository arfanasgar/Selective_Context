"""Aggregate corrected response-generation metrics and save plots."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

METRICS = ["bertscore_f1", "rouge1", "rouge2", "rougeL", "generation_time_sec"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/response_generation_corrected/response_evaluated.csv")
    parser.add_argument("--output-dir", default="results/response_generation_corrected")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    outdir = Path(args.output_dir)
    graphdir = outdir / "graphs"
    graphdir.mkdir(parents=True, exist_ok=True)

    aggregate = df.groupby("target_compression_percent").agg(
        **{metric: (metric, "mean") for metric in METRICS},
        actual_context_reduction=("actual_context_reduction", "mean"),
        original_context_tokens_gpt2=("original_context_tokens_gpt2", "mean"),
        retained_context_tokens_gpt2=("retained_context_tokens_gpt2", "mean"),
        model_fitted_context_tokens_t5=("model_fitted_context_tokens_t5", "mean")
    ).reset_index()
    aggregate.to_csv(outdir / "response_aggregate_results.csv", index=False)

    for metric in ["bertscore_f1", "rouge1", "rouge2", "rougeL"]:
        plt.figure(figsize=(8, 5))
        plt.plot(aggregate["target_compression_percent"], aggregate[metric], marker="o")
        plt.xlabel("Target article-context compression (%)")
        plt.ylabel(metric)
        plt.title(f"Corrected Response Generation: {metric}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(graphdir / f"{metric}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(aggregate["target_compression_percent"], aggregate["generation_time_sec"], marker="o")
    plt.xlabel("Target article-context compression (%)")
    plt.ylabel("Generation time (seconds)")
    plt.title("Corrected Response Generation: Runtime")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(graphdir / "runtime.png", dpi=200)
    plt.close()

    print("\nCorrected response-generation aggregate results:")
    print(aggregate.to_string(index=False))
    print(f"\nSaved CSV and graphs under {outdir}")

if __name__ == "__main__":
    main()
