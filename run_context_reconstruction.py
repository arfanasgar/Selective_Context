"""
Run original-context reconstruction on the same 10 BBC articles used in QA
and response generation. Run from the project root.
"""
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from context_manager import NewsContextManager

DATASET_PATH = Path("Dataset/articles.json")
INPUT_PATH = Path("results/context_reconstruction/context_reconstruction_dataset.json")
OUT_DIR = Path("results/context_reconstruction")
MODEL_NAME = "google/flan-t5-base"
COMPRESSOR_MODEL = "gpt2"
RATIOS = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]
MAX_INPUT_TOKENS = 512
MAX_NEW_TOKENS = 512
FORCE_RERUN = False

def generate_reconstruction(text, tokenizer, model):
    prompt = (
        "The following news article has had some phrases removed. "
        "Reconstruct the original article as accurately as possible using "
        "the remaining context. Do not explain your answer.\n\n"
        f"COMPRESSED ARTICLE:\n{text}\n\n"
        "RECONSTRUCTED ARTICLE:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=MAX_INPUT_TOKENS)
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=2, do_sample=False, early_stopping=True
        )
    elapsed = time.perf_counter() - start
    prediction = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    return prediction, int(inputs["input_ids"].shape[1]), elapsed

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / "reconstruction_results.csv"
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing {INPUT_PATH}")
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    target_articles = payload["articles"]

    # Match the same manager configuration used by the project's QA runner.
    print("Preparing phrase-level Selective Context compression...")
    manager = NewsContextManager(
        path=str(DATASET_PATH.parent),
        num_articles=10,
        mask_ratio=0.20,
        compute_self_info=True
    )
    valid_ids = {a["article_id"] for a in target_articles}
    manager_ids = {a.id for a in manager.articles}
    missing = valid_ids - manager_ids
    if missing:
        raise RuntimeError(
            "Some target article IDs were not loaded by NewsContextManager: "
            + ", ".join(sorted(missing))
            + ". Check Dataset/articles.json ordering and article IDs."
        )

    print(f"Loading reconstruction model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to("cpu").eval()

    existing = []
    completed = set()
    if output_path.exists() and not FORCE_RERUN:
        with output_path.open(encoding="utf-8-sig", newline="") as f:
            existing = list(csv.DictReader(f))
        completed = {
            (r["article_id"], float(r["compression_ratio"])) for r in existing
        }
        print(f"Resume enabled: {len(completed)} existing rows found.")

    rows = existing[:]
    manager_by_id = {a.id: a for a in manager.articles}
    for ratio in RATIOS:
        condition = "baseline" if ratio == 0 else f"{int(ratio*100)}%"
        print(f"\n===== CONDITION: {condition} =====")
        if ratio == 0:
            compressed_by_id = {
                article_id: manager_by_id[article_id].sections[0]
                for article_id in valid_ids
            }
        else:
            manager.mask_ratio = ratio
            contexts = manager.generate_context(
                mask_method="self-info", mask_level="phrase",
                num_articles=len(manager.articles)
            )
            compressed_by_id = {c.id: c.context for c in contexts}

        for index, article in enumerate(target_articles, 1):
            key = (article["article_id"], float(ratio))
            if key in completed:
                print(f"[{index}/10] Skipping completed: {article['title']}")
                continue
            compressed = compressed_by_id.get(article["article_id"])
            if compressed is None:
                raise RuntimeError(f"Compressed context missing: {article['article_id']}")
            prediction, input_tokens, elapsed = generate_reconstruction(
                compressed, tokenizer, model
            )
            rows.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "compression_ratio": ratio,
                "condition": condition,
                "original_context": article["original_context"],
                "compressed_context": compressed,
                "reconstructed_context": prediction,
                "input_tokens": input_tokens,
                "generation_time_sec": round(elapsed, 4)
            })
            with output_path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[{index}/10] {article['title']} | {elapsed:.1f}s")

    print(f"\nDone. Saved {len(rows)} rows to {output_path}")

if __name__ == "__main__":
    main()
