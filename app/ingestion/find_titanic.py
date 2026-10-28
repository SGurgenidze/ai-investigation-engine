
import json
from pathlib import Path

DATA_PATH = Path("data/documents.jsonl")


def main():
    matches = 0

    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            document = json.loads(line)

            title = document.get("title", "")
            text = document.get("text", "")
            combined = (title + " " + text).casefold()

            if "titanic" in combined:
                matches += 1
                print(f"\nTitle: {title}")
                print(f"Text: {text[:300]}")

    print(f"\nTotal Titanic matches: {matches}")


if __name__ == "__main__":
    main()
