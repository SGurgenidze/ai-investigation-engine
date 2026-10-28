
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "documents.jsonl"

with DATA_PATH.open("r", encoding="utf-8") as file:
    for index, line in enumerate(file):
        document = json.loads(line)

        print(f"\n--- Document {index + 1} ---")
        print("ID:", document.get("id"))
        print("Title:", document.get("title"))
        print("Text:", (document.get("text") or "")[:500])

        if index == 19:
            break
