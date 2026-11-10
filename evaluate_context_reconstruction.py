"""Calculate BLEU, METEOR, ROUGE and BERTScore for reconstructed articles."""
import csv
import re
from pathlib import Path

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score

BASE = Path("results/context_reconstruction")
INPUT = BASE / "reconstruction_results.csv"
OUTPUT = BASE / "reconstruction_evaluated.csv"

def tokens(text):
    return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"{INPUT} not found. Run run_context_reconstruction.py first.")
    with INPUT.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("No reconstruction results found.")

    # NLTK METEOR needs WordNet. Download only if absent.
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4")

    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    references = [r["original_context"] for r in rows]
    predictions = [r["reconstructed_context"] for r in rows]
    print("Calculating BERTScore (distilroberta-base)...")
    _, _, f1 = bert_score(
        predictions, references, lang="en",
        model_type="distilroberta-base", device="cpu", verbose=True
    )

    smoothing = SmoothingFunction().method1
    for i, row in enumerate(rows):
        ref_tokens = tokens(references[i])
        pred_tokens = tokens(predictions[i])
        row["bleu"] = sentence_bleu(
            [ref_tokens], pred_tokens, smoothing_function=smoothing
        ) if pred_tokens else 0.0
        row["meteor"] = meteor_score([ref_tokens], pred_tokens) if pred_tokens else 0.0
        scores = rouge.score(references[i], predictions[i])
        row["rouge1"] = scores["rouge1"].fmeasure
        row["rouge2"] = scores["rouge2"].fmeasure
        row["rougeL"] = scores["rougeL"].fmeasure
        row["bertscore_f1"] = float(f1[i].item())
        for metric in ["bleu", "meteor", "rouge1", "rouge2", "rougeL", "bertscore_f1"]:
            row[metric] = round(float(row[metric]), 6)

    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluated results: {OUTPUT}")

if __name__ == "__main__":
    main()
