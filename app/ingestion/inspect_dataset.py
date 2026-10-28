
import json
from pathlib import Path


DATA_PATH = Path("data/documents.jsonl")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run prepare_dataset.py first."
        )

    count = 0

    with DATA_PATH.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            count += 1

            if count == 1:
                print("First record:")
                print(json.dumps(record, indent=2, ensure_ascii=False))

    print(f"\nTotal records: {count}")
    print(f"Meets 50,000-record target: {count >= 50_000}")


if __name__ == "__main__":
    main()
