"""Aggregate reconstruction metrics and create comparison graphs."""
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

BASE = Path("results/context_reconstruction")
INPUT = BASE / "reconstruction_evaluated.csv"
OUT = BASE / "reconstruction_aggregate_results.csv"
GRAPH_DIR = BASE / "graphs"
METRICS = ["bleu", "meteor", "rouge1", "rouge2", "rougeL", "bertscore_f1"]

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run evaluate_context_reconstruction.py first.")
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    groups = defaultdict(list)
    for row in rows:
        groups[float(row["compression_ratio"])].append(row)

    aggregate = []
    for ratio in sorted(groups):
        group = groups[ratio]
        item = {
            "compression_ratio": ratio,
            "condition": "baseline" if ratio == 0 else f"{int(ratio*100)}%",
            "num_articles": len(group)
        }
        for metric in METRICS:
            item[f"mean_{metric}"] = round(
                sum(float(r[metric]) for r in group) / len(group), 6
            )
        item["mean_generation_time_sec"] = round(
            sum(float(r["generation_time_sec"]) for r in group) / len(group), 4
        )
        aggregate.append(item)

    with OUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate)

    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    labels = [r["condition"] for r in aggregate]
    for metric in METRICS:
        values = [r[f"mean_{metric}"] for r in aggregate]
        plt.figure(figsize=(8, 5))
        plt.plot(labels, values, marker="o")
        plt.xlabel("Context compression (removed)")
        plt.ylabel(metric.replace("_", " ").upper())
        plt.title(f"Context Reconstruction: {metric.upper()}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(GRAPH_DIR / f"{metric}.png", dpi=200)
        plt.close()
    print(f"Aggregate results: {OUT}")
    print(f"Graphs saved in: {GRAPH_DIR}")

if __name__ == "__main__":
    main()
