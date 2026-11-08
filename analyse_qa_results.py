"""
Evaluate and aggregate QA predictions with Exact Match and token-level F1.
Creates aggregate CSV and two plots under results/qa/.

Run from the project root:
    python analyse_qa_results.py
"""
import csv
import re
import string
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

QA_DIR = Path("results/qa")
INPUT_PATH = QA_DIR / "qa_results.csv"
AGGREGATE_PATH = QA_DIR / "qa_aggregate_results.csv"
GRAPH_DIR = QA_DIR / "graphs"


def normalize_answer(text):
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = defaultdict(int)
    for token in pred_tokens:
        common[token] += 1
    overlap = 0
    ref_counts = defaultdict(int)
    for token in ref_tokens:
        ref_counts[token] += 1
    for token, count in common.items():
        overlap += min(count, ref_counts[token])

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run python run_qa_experiment.py first."
        )

    QA_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH).fillna("")
    if df.empty:
        raise RuntimeError("QA results CSV is empty.")

    df["exact_match"] = df.apply(
        lambda row: exact_match(row["predicted_answer"], row["reference_answer"]),
        axis=1
    )
    df["token_f1"] = df.apply(
        lambda row: token_f1(row["predicted_answer"], row["reference_answer"]),
        axis=1
    )

    aggregate = (
        df.groupby(["compression_ratio", "condition"], as_index=False)
        .agg(
            num_questions=("question_id", "count"),
            exact_match=("exact_match", "mean"),
            token_f1=("token_f1", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_qa_seconds=("qa_seconds", "mean")
        )
        .sort_values("compression_ratio")
    )
    aggregate.to_csv(AGGREGATE_PATH, index=False, encoding="utf-8-sig")

    # Plot F1 by compression ratio.
    plt.figure(figsize=(8, 5))
    plt.plot(
        aggregate["compression_ratio"] * 100,
        aggregate["token_f1"],
        marker="o"
    )
    plt.xlabel("Target compression ratio (%)")
    plt.ylabel("Mean token-level F1")
    plt.title("Question Answering Quality vs Context Compression")
    plt.xticks(aggregate["compression_ratio"] * 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "qa_f1.png", dpi=200)
    plt.close()

    # Plot exact match by compression ratio.
    plt.figure(figsize=(8, 5))
    plt.plot(
        aggregate["compression_ratio"] * 100,
        aggregate["exact_match"],
        marker="o"
    )
    plt.xlabel("Target compression ratio (%)")
    plt.ylabel("Exact Match")
    plt.title("Question Answering Exact Match vs Context Compression")
    plt.xticks(aggregate["compression_ratio"] * 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_DIR / "qa_exact_match.png", dpi=200)
    plt.close()

    print("\n========== QA AGGREGATE RESULTS ==========")
    print(aggregate.to_string(index=False))
    print(f"\nSaved: {AGGREGATE_PATH}")
    print(f"Graphs: {GRAPH_DIR}")
    print(
        "\nInterpretation note: QA questions and reference answers are automatically "
        "generated. Inspect them manually; Exact Match and token F1 measure agreement "
        "with those references, not independently verified factual correctness."
    )


if __name__ == "__main__":
    main()
