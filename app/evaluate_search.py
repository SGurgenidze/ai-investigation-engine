
import json
from pathlib import Path
from collections import defaultdict

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "documents.jsonl"
)


def main():
    questions = defaultdict(list)

    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)
            questions[document["source_question_id"]].append(
                document["title"]
            )

    for question_id, titles in list(questions.items())[:3]:
        print(f"\nQuestion ID: {question_id}")
        print("Number of documents:", len(titles))
        for title in titles:
            print("-", title)


if __name__ == "__main__":
    main()
