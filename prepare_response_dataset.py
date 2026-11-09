"""
Prepare multi-turn response-generation examples from the reviewed QA dataset.

Each example includes:
- the full BBC article context (the evidence source)
- earlier QA pairs as conversation history
- the next QA question as the current user message
- the reviewed answer as reference response

This is a BBC-news adaptation of a conversation response-generation task,
not an exact reproduction of the original paper's ShareGPT experiment.
"""
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/qa/qa_dataset.json")
    parser.add_argument("--output", default="results/response_generation_corrected/response_dataset.json")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"QA dataset not found: {source}")

    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("articles"), list):
        raise ValueError("Expected QA JSON with top-level 'articles' list.")

    examples = []
    for article in data["articles"]:
        article_id = article["article_id"]
        title = article.get("title", article_id)
        context = article.get("context", "").strip()
        qa_pairs = article.get("qa_pairs", [])

        if not context:
            raise ValueError(f"Article {article_id} has no context.")
        if len(qa_pairs) < 2:
            continue

        for i in range(1, len(qa_pairs)):
            history = []
            for previous in qa_pairs[:i]:
                history.append({"role": "user", "content": previous["question"]})
                history.append({"role": "assistant", "content": previous["reference_answer"]})

            current = qa_pairs[i]
            examples.append({
                "example_id": f"{article_id}_dialogue_{i}",
                "article_id": article_id,
                "article_title": title,
                "article_context": context,
                "history": history,
                "current_question": current["question"],
                "reference_response": current["reference_answer"],
                "history_turns": i
            })

    if not examples:
        raise ValueError("No response-generation examples were created.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Corrected response-generation dataset created.")
    print(f"Articles processed: {len(data['articles'])}")
    print(f"Dialogue examples: {len(examples)}")
    print("Each example includes its full source article context.")
    print(f"Saved to: {output}")

if __name__ == "__main__":
    main()
