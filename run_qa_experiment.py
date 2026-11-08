"""
Run extractive QA on original and Selective-Context-compressed BBC articles.

Run from the project root:
    python run_qa_experiment.py

Writes only under results/qa/. Existing summarisation files are untouched.
"""
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from context_manager import NewsContextManager

DATASET_PATH = Path("Dataset/articles.json")
QA_DATASET_PATH = Path("results/qa/qa_dataset.json")
QA_DIR = Path("results/qa")

NUM_ARTICLES = 10
COMPRESSION_RATIOS = [0.20, 0.35, 0.50, 0.65, 0.80]
QA_MODEL_NAME = "deepset/roberta-base-squad2"
MAX_CONTEXT_TOKENS = 512
MAX_ANSWER_TOKENS = 40
FORCE_RERUN = False


def normalise(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_qa_dataset():
    if not QA_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{QA_DATASET_PATH} not found. Run python generate_qa_dataset.py first."
        )
    payload = json.loads(QA_DATASET_PATH.read_text(encoding="utf-8"))
    return payload["articles"][:NUM_ARTICLES]


def load_qa_model():
    print(f"Loading QA model: {QA_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL_NAME)
    model.to("cpu")
    model.eval()
    print("QA model loaded on CPU.")
    return tokenizer, model


def answer_question(context, question, tokenizer, model):
    encoded = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation="only_second",
        max_length=MAX_CONTEXT_TOKENS
    )
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**encoded)
    elapsed = time.perf_counter() - start_time

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]
    sequence_ids = encoded.sequence_ids(0)

    # Only allow answer spans from the context sequence, not the question.
    context_positions = [
        i for i, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1
    ]
    if not context_positions:
        return "", int(encoded["input_ids"].shape[1]), elapsed

    best_score = float("-inf")
    best_start = best_end = context_positions[0]
    for start in context_positions:
        upper = min(start + MAX_ANSWER_TOKENS, context_positions[-1] + 1)
        for end in range(start, upper):
            if sequence_ids[end] != 1:
                break
            score = float(start_logits[start] + end_logits[end])
            if score > best_score:
                best_score = score
                best_start, best_end = start, end

    answer_ids = encoded["input_ids"][0][best_start:best_end + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
    return answer, int(encoded["input_ids"].shape[1]), elapsed


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = QA_DIR / "qa_results.csv"
    if output_path.exists() and not FORCE_RERUN:
        print(f"Existing {output_path} will be overwritten by this run.")

    qa_articles = load_qa_dataset()
    if not qa_articles:
        raise RuntimeError("QA dataset contains no articles.")

    print("Preparing Selective Context phrase compression...")
    manager = NewsContextManager(
        path=str(DATASET_PATH.parent),
        num_articles=NUM_ARTICLES,
        mask_ratio=COMPRESSION_RATIOS[0],
        compute_self_info=True
    )

    # Match article IDs exactly as the original summarisation runner does.
    valid_ids = {article.id for article in manager.articles}
    qa_articles = [
        article for article in qa_articles
        if article["article_id"] in valid_ids
    ]
    if not qa_articles:
        raise RuntimeError(
            "No QA dataset article IDs matched NewsContextManager article IDs. "
            "Check the article_id construction in generate_qa_dataset.py."
        )

    tokenizer, model = load_qa_model()
    rows = []

    conditions = [(0.0, None)] + [(ratio, ratio) for ratio in COMPRESSION_RATIOS]
    for ratio, mask_ratio in conditions:
        label = "baseline" if ratio == 0 else f"{ratio:.2f}"
        print(f"\n========== QA CONDITION: {label} ==========")

        compressed_by_id = {}
        if mask_ratio is None:
            for article in qa_articles:
                compressed_by_id[article["article_id"]] = article["context"]
        else:
            manager.mask_ratio = mask_ratio
            contexts = manager.generate_context(
                mask_method="self-info",
                mask_level="phrase",
                num_articles=len(manager.articles)
            )
            compressed_by_id = {
                context.id: context.context for context in contexts
            }

        for article_index, article in enumerate(qa_articles, start=1):
            context = compressed_by_id.get(article["article_id"])
            if context is None:
                print(f"Skipping missing context: {article['title']}")
                continue

            for qa in article["qa_pairs"]:
                prediction, input_tokens, elapsed = answer_question(
                    context, qa["question"], tokenizer, model
                )
                rows.append({
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "question_id": qa["question_id"],
                    "question": qa["question"],
                    "reference_answer": qa["reference_answer"],
                    "predicted_answer": prediction,
                    "compression_ratio": ratio,
                    "condition": label,
                    "input_tokens": input_tokens,
                    "qa_seconds": round(elapsed, 5)
                })
            print(f"[{article_index}/{len(qa_articles)}] {article['title']}")

    if not rows:
        raise RuntimeError("No QA predictions were generated.")

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\nQA experiment complete.")
    print(f"Articles: {len(qa_articles)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Predictions: {len(rows)}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
