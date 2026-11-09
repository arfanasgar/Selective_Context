"""Evaluate corrected response-generation outputs with BERTScore and ROUGE."""
import argparse
from pathlib import Path
import pandas as pd
from bert_score import score as bert_score
from rouge_score import rouge_scorer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/response_generation_corrected/response_results.csv")
    parser.add_argument("--output", default="results/response_generation_corrected/response_evaluated.csv")
    parser.add_argument("--bert-model", default="distilroberta-base")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    expected = df["example_id"].nunique() * df["compression_ratio"].nunique()
    if len(df) != expected:
        print(f"WARNING: found {len(df)} rows; expected {expected} based on unique examples/conditions.")

    predictions = df["generated_response"].fillna("").astype(str).tolist()
    references = df["reference_response"].fillna("").astype(str).tolist()
    print(f"Computing BERTScore with {args.bert_model}...")
    _, _, f1 = bert_score(
        predictions, references, model_type=args.bert_model,
        lang="en", verbose=True, device="cpu"
    )
    df["bertscore_f1"] = f1.cpu().numpy()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    df["rouge1"] = [s["rouge1"].fmeasure for s in scores]
    df["rouge2"] = [s["rouge2"].fmeasure for s in scores]
    df["rougeL"] = [s["rougeL"].fmeasure for s in scores]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} evaluated rows to {out}")

if __name__ == "__main__":
    main()
