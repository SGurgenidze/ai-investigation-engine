
import json
import re
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path("data/documents.jsonl")
MAX_RECORDS = 60_000


def normalize_text(text: str) -> str:
    """Normalize whitespace and casing for duplicate detection."""
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


def prepare_dataset() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        "hotpotqa/hotpot_qa",
        "distractor",
        split="train",
        streaming=True,
    )

    seen = set()
    saved_count = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for row in dataset:
            titles = row["context"]["title"]
            paragraphs = row["context"]["sentences"]

            for title, sentences in zip(titles, paragraphs):
                text = " ".join(sentences).strip()

                if not text:
                    continue

                normalized = normalize_text(text)
                if normalized in seen:
                    continue

                seen.add(normalized)

                record = {
                    "id": f"hotpot-{saved_count + 1}",
                    "source": "HotpotQA",
                    "title": title,
                    "text": text,
                    "source_question_id": row["id"],
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                saved_count += 1

                if saved_count >= MAX_RECORDS:
                    break

            if saved_count >= MAX_RECORDS:
                break

    print(f"Saved records: {saved_count}")
    print(f"Output file: {OUTPUT_PATH.resolve()}")

    if saved_count < 50_000:
        print(
            "WARNING: Fewer than 50,000 unique records were produced. "
            "More source data or a different deduplication strategy is needed."
        )


if __name__ == "__main__":
    prepare_dataset()
