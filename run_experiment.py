"""
BBC News Selective Context experiment runner.

Runs one fixed sample through:
  - Original (uncompressed) article context
  - Selective Context phrase compression at 20%, 35%, 50%, 65%, and 80%

Outputs:
  results/experiment_results.csv
  results/generated_summaries.csv

Run from the project root:
  python run_experiment.py
"""

import csv
import json
import os
import time
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from context_manager import NewsContextManager


# -------------------- Experiment settings --------------------

DATASET_PATH = Path("Dataset/articles.json")
RESULTS_DIR = Path("results")
NUM_ARTICLES = 30

COMPRESSION_RATIOS = [0.20, 0.35, 0.50, 0.65, 0.80]
MASK_LEVEL = "phrase"

SUMMARY_MODEL_NAME = "sshleifer/distilbart-cnn-6-6"
MAX_INPUT_TOKENS = 1024
MAX_NEW_TOKENS = 100
MIN_NEW_TOKENS = 25
NUM_BEAMS = 2

# Set True only if you intentionally want to regenerate every summary.
FORCE_RERUN = False


def load_dataset_records(path):
    """Load either JSONL or a normal JSON list."""
    records = []
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            records = loaded
        elif isinstance(loaded, dict):
            records = [loaded]
    except json.JSONDecodeError:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_summarizer():
    print(f"\nLoading summarisation model: {SUMMARY_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME)
    model.to("cpu")
    model.eval()
    print("Summarisation model loaded on CPU.")
    return tokenizer, model


def summarise(text, tokenizer, model):
    """Generate one deterministic summary and return it with token count/time."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    input_tokens = int(encoded["input_ids"].shape[1])

    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )
    elapsed = time.perf_counter() - start
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary, input_tokens, elapsed


def rouge_scores(reference, prediction, scorer):
    scores = scorer.score(reference, prediction)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}. Run this script from the project root."
        )

    RESULTS_DIR.mkdir(exist_ok=True)
    records = load_dataset_records(DATASET_PATH)

    # The context manager uses the same local dataset and computes GPT-2
    # self-information once. The manager's phrase units are reused at all ratios.
    print(f"Preparing up to {NUM_ARTICLES} BBC articles...")
    manager = NewsContextManager(
        path=str(DATASET_PATH.parent),
        num_articles=NUM_ARTICLES,
        mask_ratio=COMPRESSION_RATIOS[0],
        compute_self_info=True,
    )

    # Match dataset metadata to the articles that survived context-length filtering.
    metadata_by_link = {
        str(record.get("link", "")): record for record in records
    }
    articles = []
    for article in manager.articles:
        metadata = metadata_by_link.get(article.entry_id, {})
        articles.append({
            "id": article.id,
            "title": article.title,
            "text": article.text,
            "reference": str(metadata.get("description") or "").strip(),
            "link": article.entry_id,
        })

    if not articles:
        raise RuntimeError("No usable articles were loaded from the dataset.")

    print(f"Using {len(articles)} articles.")
    tokenizer, model = load_summarizer()
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True,
    )

    summary_rows = []
    metric_rows = []

    # Baseline: summarise the original, uncompressed article.
    print("\n========== BASELINE: 0% compression ==========")
    for index, article in enumerate(articles, start=1):
        summary, token_count, elapsed = summarise(
            article["text"], tokenizer, model
        )
        reference = article["reference"]
        scores = rouge_scores(reference, summary, scorer) if reference else {
            "rouge1_f": None, "rouge2_f": None, "rougeL_f": None
        }

        summary_rows.append({
            "article_id": article["id"],
            "title": article["title"],
            "compression_ratio": 0.0,
            "input_tokens": token_count,
            "summary_seconds": round(elapsed, 4),
            "reference_description": reference,
            "generated_summary": summary,
        })
        metric_rows.append({
            "article_id": article["id"],
            "title": article["title"],
            "compression_ratio": 0.0,
            "input_tokens": token_count,
            "token_reduction_pct": 0.0,
            "summary_seconds": elapsed,
            **scores,
        })
        print(f"[Baseline {index}/{len(articles)}] {article['title']}")

    # Compressed conditions. The manager's mask_ratio is changed between runs.
    for ratio in COMPRESSION_RATIOS:
        print(f"\n========== SELECTIVE CONTEXT: {ratio:.0%} ==========")
        manager.mask_ratio = ratio
        compressed_contexts = manager.generate_context(
            mask_method="self-info",
            mask_level=MASK_LEVEL,
            num_articles=len(articles),
        )
        compressed_by_id = {context.id: context.context for context in compressed_contexts}

        for index, article in enumerate(articles, start=1):
            compressed_text = compressed_by_id.get(article["id"])
            if compressed_text is None:
                print(f"Skipping missing compressed context: {article['title']}")
                continue

            summary, token_count, elapsed = summarise(
                compressed_text, tokenizer, model
            )
            baseline_tokens = next(
                row["input_tokens"] for row in metric_rows
                if row["article_id"] == article["id"]
                and row["compression_ratio"] == 0.0
            )
            reduction = (
                (baseline_tokens - token_count) / baseline_tokens * 100
                if baseline_tokens else 0.0
            )
            reference = article["reference"]
            scores = rouge_scores(reference, summary, scorer) if reference else {
                "rouge1_f": None, "rouge2_f": None, "rougeL_f": None
            }

            summary_rows.append({
                "article_id": article["id"],
                "title": article["title"],
                "compression_ratio": ratio,
                "input_tokens": token_count,
                "summary_seconds": round(elapsed, 4),
                "reference_description": reference,
                "generated_summary": summary,
            })
            metric_rows.append({
                "article_id": article["id"],
                "title": article["title"],
                "compression_ratio": ratio,
                "input_tokens": token_count,
                "token_reduction_pct": reduction,
                "summary_seconds": elapsed,
                **scores,
            })
            print(f"[{index}/{len(articles)}] {article['title']}")

    metrics_path = RESULTS_DIR / "experiment_results.csv"
    summaries_path = RESULTS_DIR / "generated_summaries.csv"

    with metrics_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metric_rows)

    with summaries_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n========== EXPERIMENT COMPLETE ==========")
    print(f"Articles processed: {len(articles)}")
    print(f"Conditions: baseline + {len(COMPRESSION_RATIOS)} compression ratios")
    print(f"Metric rows: {len(metric_rows)}")
    print(f"Saved: {metrics_path}")
    print(f"Saved: {summaries_path}")
    print(
        "\nNote: ROUGE is calculated against the dataset's 'description' field. "
        "Treat it as a short editorial reference, not automatically as a verified "
        "gold-standard summary. Report this limitation in your project."
    )


if __name__ == "__main__":
    main()
