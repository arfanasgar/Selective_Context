"""
Corrected response-generation experiment.

Fixes:
1. Full source article context is supplied in every prompt.
2. GPT-2 self-information ranks sentence/clause chunks from article context.
3. Compression is budgeted and measured using GPT-2 token counts.
4. CSV checkpoint is saved after every generated prediction.
5. Existing completed rows are skipped on restart.

Six target compression conditions: 0, 20, 35, 50, 65, 80 percent.
This is an adaptation using the project's BBC QA data, not an exact reproduction
of the original paper's ShareGPT conversation experiment.
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

RATIOS = [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]
OUTPUT_COLUMNS = [
    "example_id", "compression_ratio", "target_compression_percent",
    "original_context_tokens", "retained_context_tokens",
    "actual_context_reduction", "history_turns", "current_question",
    "reference_response", "generated_response", "generation_time_sec"
]

def split_chunks(text):
    # Sentence/clause-sized spans retain more meaning than arbitrary character slices.
    parts = re.split(r"(?<=[.!?;])\s+|(?<=,)\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]

class InformationCompressor:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval().to("cpu")

    @torch.inference_mode()
    def score(self, text):
        ids = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        if ids.shape[1] < 2:
            return 0.0
        logits = self.model(ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float((-token_log_probs).mean().item())

    def compress(self, text, target_ratio):
        original_ids = self.tokenizer.encode(text, add_special_tokens=False)
        original_count = len(original_ids)
        if target_ratio <= 0 or original_count == 0:
            return text, original_count, original_count, 0.0

        target_keep = max(1, int(round(original_count * (1.0 - target_ratio))))
        chunks = split_chunks(text)
        if len(chunks) <= 1:
            # For a single long span, token truncation enforces the token budget.
            kept_ids = original_ids[:target_keep]
            compressed = self.tokenizer.decode(kept_ids, skip_special_tokens=True)
            retained = len(self.tokenizer.encode(compressed, add_special_tokens=False))
            reduction = 1.0 - retained / max(1, original_count)
            return compressed, original_count, retained, reduction

        scored = []
        for idx, chunk in enumerate(chunks):
            ids = self.tokenizer.encode(chunk, add_special_tokens=False)
            if ids:
                scored.append((self.score(chunk), idx, chunk, ids))

        # Select high-information chunks until the token budget is reached.
        selected = []
        used = 0
        for score, idx, chunk, ids in sorted(scored, key=lambda x: x[0], reverse=True):
            remaining = target_keep - used
            if remaining <= 0:
                break
            if len(ids) <= remaining:
                selected.append((idx, ids))
                used += len(ids)
            else:
                # Fill the remaining token budget with the start of this final chunk.
                selected.append((idx, ids[:remaining]))
                used += remaining
                break

        selected.sort(key=lambda x: x[0])
        compressed = " ".join(
            self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            for _, ids in selected if ids
        ).strip()

        # Re-tokenize the final text and enforce the budget after joining/decoding.
        final_ids = self.tokenizer.encode(compressed, add_special_tokens=False)
        if len(final_ids) > target_keep:
            compressed = self.tokenizer.decode(final_ids[:target_keep], skip_special_tokens=True)
            final_ids = self.tokenizer.encode(compressed, add_special_tokens=False)

        retained = len(final_ids)
        reduction = 1.0 - retained / max(1, original_count)
        return compressed, original_count, retained, reduction

def make_prompt(article_context, history, question):
    history_text = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Assistant"
        history_text.append(f"{role}: {turn['content']}")
    dialogue = "\n".join(history_text) if history_text else "(No previous turns.)"
    return (
        "You are a factual assistant answering questions about a news article. "
        "Use the article as the primary evidence. Use the conversation history "
        "to understand the dialogue, but do not treat earlier answers as evidence "
        "when the article contradicts them. Answer the latest question concisely. "
        "If the answer is not stated in the article, say you cannot determine it.\n\n"
        f"LATEST USER QUESTION:\n{question}\n\n"
        f"CONVERSATION HISTORY:\n{dialogue}\n\n"
        f"ARTICLE CONTEXT:\n{article_context}\n\n"
        "ASSISTANT RESPONSE:"
    )

def fit_context_to_input_limit(article_context, history, question, tokenizer, max_input_tokens):
    # Reserve space for instructions, the current question, and previous dialogue.
    skeleton = make_prompt("", history, question)
    skeleton_ids = tokenizer(skeleton, add_special_tokens=True, truncation=False)["input_ids"]
    available = max(32, max_input_tokens - len(skeleton_ids) - 8)
    context_ids = tokenizer(
        article_context, add_special_tokens=False, truncation=True,
        max_length=available
    )["input_ids"]
    fitted_context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return fitted_context, len(context_ids), len(tokenizer(
        article_context, add_special_tokens=False, truncation=False
    )["input_ids"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/response_generation_corrected/response_dataset.json")
    parser.add_argument("--output", default="results/response_generation_corrected/response_results.csv")
    parser.add_argument("--generator", default="google/flan-t5-base")
    parser.add_argument("--compressor", default="gpt2")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true",
                        help="Start from scratch instead of resuming an existing checkpoint.")
    args = parser.parse_args()

    examples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    rows = []
    if output.exists() and not args.overwrite:
        old = pd.read_csv(output)
        if not old.empty:
            rows = old.to_dict("records")
            completed = {
                (str(r["example_id"]), int(round(float(r["compression_ratio"]) * 100)))
                for r in rows
            }
        print(f"Resuming: {len(completed)} predictions already saved.")

    print(f"Loading GPT-2 compressor: {args.compressor}")
    compressor = InformationCompressor(args.compressor)
    print(f"Loading response model: {args.generator}")
    tokenizer = AutoTokenizer.from_pretrained(args.generator)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.generator)
    model.eval().to("cpu")

    total = len(examples) * len(RATIOS)
    done = len(completed)

    for ex in examples:
        for ratio in RATIOS:
            pct = int(round(ratio * 100))
            key = (str(ex["example_id"]), pct)
            if key in completed:
                continue

            fitted_context, fitted_t5_tokens, full_t5_context_tokens = fit_context_to_input_limit(
                ex["article_context"], ex["history"], ex["current_question"],
                tokenizer, args.max_input_tokens
            )
            compressed_context, original_tokens, retained_tokens, actual_reduction = (
                compressor.compress(fitted_context, ratio)
            )
            prompt = make_prompt(compressed_context, ex["history"], ex["current_question"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            input_token_count = int(inputs["input_ids"].shape[1])
            if input_token_count > args.max_input_tokens:
                raise ValueError(
                    f"Prompt still exceeds max input tokens ({input_token_count} > "
                    f"{args.max_input_tokens}); lower max_input_tokens or shorten history."
                )

            start = time.perf_counter()
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=2,
                    do_sample=False
                )
            elapsed = time.perf_counter() - start
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

            row = {
                "example_id": ex["example_id"],
                "compression_ratio": ratio,
                "target_compression_percent": pct,
                "source_context_tokens_t5": full_t5_context_tokens,
                "model_fitted_context_tokens_t5": fitted_t5_tokens,
                "original_context_tokens_gpt2": original_tokens,
                "retained_context_tokens_gpt2": retained_tokens,
                "actual_context_reduction": actual_reduction,
                "history_turns": ex.get("history_turns", 0),
                "current_question": ex["current_question"],
                "reference_response": ex["reference_response"],
                "generated_response": prediction,
                "generation_time_sec": elapsed,
                "model_input_tokens": input_token_count
            }
            rows.append(row)
            done += 1

            # Save after every prediction so shutdowns do not erase progress.
            pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
            print(
                f"[{done}/{total}] {ex['example_id']} | target={pct}% | "
                f"actual context reduction={actual_reduction:.1%} | "
                f"input tokens={input_token_count} | saved"
            )

    print(f"\nCompleted {done}/{total} predictions.")
    print(f"Checkpoint/results saved to: {output}")

if __name__ == "__main__":
    main()
