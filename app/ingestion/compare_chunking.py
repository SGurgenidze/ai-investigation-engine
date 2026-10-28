
import json
from pathlib import Path

from app.ingestion.chunking import (
    fixed_size_chunks,
    sentence_based_chunks,
)


DATA_PATH = Path("data/documents.jsonl")


def main():
    with DATA_PATH.open(encoding="utf-8") as file:
        # Use the same 100 documents for a fair first comparison.
        sample = [
            json.loads(line)
            for _, line in zip(range(100), file)
        ]

    fixed_total = 0
    sentence_total = 0

    for document in sample:
        text = document["text"]

        fixed_chunks = fixed_size_chunks(text)
        sentence_chunks = sentence_based_chunks(text)

        fixed_total += len(fixed_chunks)
        sentence_total += len(sentence_chunks)

    print(f"Documents tested: {len(sample)}")
    print(f"Fixed-size chunks: {fixed_total}")
    print(f"Sentence-based chunks: {sentence_total}")

    if fixed_total > 0 and sentence_total > 0:
        print(
            "Average fixed-size chunks per document:",
            round(fixed_total / len(sample), 2),
        )
        print(
            "Average sentence-based chunks per document:",
            round(sentence_total / len(sample), 2),
        )


if __name__ == "__main__":
    main()
