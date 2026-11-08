"""
Create a small question-answering dataset from the local BBC News articles.

Run from the project root:
    python generate_qa_dataset.py

This script does not modify the original dataset or summarisation experiment files.
It uses a local T5 question-generation model and spaCy noun chunks to propose
answer spans. Review the generated questions in results/qa/qa_dataset.json.
"""
import json
import re
from pathlib import Path

import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DATASET_PATH = Path("Dataset/articles.json")
QA_DIR = Path("results/qa")
OUTPUT_PATH = QA_DIR / "qa_dataset.json"

NUM_ARTICLES = 10
QUESTIONS_PER_ARTICLE = 3
QG_MODEL_NAME = "valhalla/t5-base-qa-qg-hl"
MAX_CONTEXT_CHARS = 2500
MAX_NEW_TOKENS = 64


def load_records(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line {line_no}: {exc}")
    return records


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def choose_answer_spans(text, nlp, limit):
    """Choose distinct, informative noun chunks that occur in the article."""
    doc = nlp(text[:MAX_CONTEXT_CHARS])
    candidates = []
    seen = set()

    for chunk in doc.noun_chunks:
        answer = clean_text(chunk.text).strip(" ,.;:")
        key = answer.casefold()
        if not (2 <= len(answer.split()) <= 7):
            continue
        if len(answer) < 4 or key in seen:
            continue
        if answer.casefold() in {"the company", "the group", "the people"}:
            continue
        seen.add(key)
        candidates.append(answer)

    # Prefer named entities and longer informative noun phrases.
    entities = {clean_text(ent.text).casefold() for ent in doc.ents}
    candidates.sort(
        key=lambda x: (x.casefold() in entities, len(x.split()), len(x)),
        reverse=True
    )
    return candidates[:limit]


def generate_question(context, answer, tokenizer, model):
    marked_context = context.replace(answer, f"<hl> {answer} <hl>", 1)
    if marked_context == context:
        return None

    prompt = f"generate question: {marked_context}"
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4,
            do_sample=False
        )
    question = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    question = clean_text(question)
    if question and not question.endswith("?"):
        question += "?"
    return question or None


def main():
    QA_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(DATASET_PATH)

    usable = []
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        content = clean_text(record.get("content"))
        if len(content) < 250:
            continue
        usable.append({
            "article_id": str(record.get("link") or record.get("id") or f"bbc-news-{i+1}").split("/")[-1],
            "title": clean_text(record.get("title") or f"BBC News article {i+1}"),
            "context": content,
        })
        if len(usable) >= NUM_ARTICLES:
            break

    if not usable:
        raise RuntimeError("No usable articles found in Dataset/articles.json")

    print("Loading spaCy English model...")
    nlp = spacy.load("en_core_web_sm")
    print(f"Loading question-generation model: {QG_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        QG_MODEL_NAME,
        use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(QG_MODEL_NAME)
    model.to("cpu")
    model.eval()

    qa_rows = []
    for article_index, article in enumerate(usable, start=1):
        context = article["context"][:MAX_CONTEXT_CHARS]
        answer_spans = choose_answer_spans(context, nlp, QUESTIONS_PER_ARTICLE * 4)
        accepted = []
        seen_questions = set()

        for answer in answer_spans:
            question = generate_question(context, answer, tokenizer, model)
            if not question:
                continue
            key = question.casefold()
            if key in seen_questions or len(question.split()) < 4:
                continue
            seen_questions.add(key)
            accepted.append({
                "question_id": f"{article['article_id']}_q{len(accepted)+1}",
                "question": question,
                "reference_answer": answer
            })
            if len(accepted) >= QUESTIONS_PER_ARTICLE:
                break

        if not accepted:
            print(f"WARNING: No questions generated for {article['title']}")
        qa_rows.append({
            "article_id": article["article_id"],
            "title": article["title"],
            "context": article["context"],
            "qa_pairs": accepted
        })
        print(f"[{article_index}/{len(usable)}] {article['title']} — {len(accepted)} questions")

    payload = {
        "metadata": {
            "source": "Local BBC News dataset",
            "question_generation_model": QG_MODEL_NAME,
            "num_articles": len(qa_rows),
            "questions_per_article_target": QUESTIONS_PER_ARTICLE,
            "note": (
                "Automatically generated questions and noun-phrase reference answers. "
                "Manually inspect questions and answers before interpreting evaluation scores."
            )
        },
        "articles": qa_rows
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    total_questions = sum(len(row["qa_pairs"]) for row in qa_rows)
    print(f"\nSaved {total_questions} QA pairs to {OUTPUT_PATH}")
    print("Review the questions and reference answers before running the QA experiment.")


if __name__ == "__main__":
    main()
