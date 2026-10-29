
import json
import re
import heapq
from pathlib import Path

from app.ingestion.chunking import (
    fixed_size_chunks,
    sentence_based_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "documents.jsonl"


# Common English stop words
STOP_WORDS = {
    "who", "was", "is", "the", "of", "a", "an",
    "in", "on", "for", "to", "and", "did", "do",
    "does", "what", "which", "that", "by", "with",
    "movie", "film",
}


def tokenize(text: str) -> list[str]:
    """Normalize text and remove common English stop words."""
    tokens = re.findall(r"\b\w+\b", text.casefold())
    return [token for token in tokens if token not in STOP_WORDS]


def load_chunks(
    limit: int | None = None,
    strategy: str = "fixed",
) -> list[dict]:
    """Load documents and split their text into chunks."""

    if strategy == "sentence":
        chunker = sentence_based_chunks
    elif strategy == "fixed":
        chunker = fixed_size_chunks
    else:
        raise ValueError(
            "strategy must be 'sentence' or 'fixed'"
        )

    if not DATA_PATH.exists():
        print(f"Data file not found: {DATA_PATH}", flush=True)
        return []

    chunks = []

    print("Loading document chunks...", flush=True)

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            document = json.loads(line)
            text = document.get("text", "")

            if not text:
                continue

            for index, chunk_text in enumerate(chunker(text)):
                chunks.append({
                    "id": f'{document.get("id", "unknown")}-chunk-{index}',
                    "title": document.get("title", "Untitled"),
                    "text": chunk_text,
                    "source": document.get("source", "unknown"),
                })

                if limit is not None and len(chunks) >= limit:
                    print(
                        f"Loaded {len(chunks)} chunks.",
                        flush=True,
                    )
                    return chunks[:limit]

    print(f"Loaded {len(chunks)} chunks.", flush=True)
    return chunks


class HybridSearch:
    """Lightweight BM25 search without transformer models."""

    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError(
                "At least one document is required"
            )

        from rank_bm25 import BM25Okapi

        self.documents = documents

        print("Building BM25 index...", flush=True)

        tokenized_documents = [
            tokenize(document.get("text", ""))
            for document in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

        # Release the temporary tokenized list.
        del tokenized_documents

        print("BM25 index ready.", flush=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 200,
    ) -> list[dict]:
        """Return the best matching documents using BM25."""

        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")

        query_tokens = tokenize(query)

        if not query_tokens:
            return []

        # Calculate BM25 scores.
        scores = self.bm25.get_scores(query_tokens)

        # Select only the highest-scoring candidates.
        candidate_count = min(
            candidate_k,
            len(self.documents),
        )

        best_indices = heapq.nlargest(
            candidate_count,
            range(len(self.documents)),
            key=lambda index: scores[index],
        )

        results = []

        for rank, index in enumerate(best_indices, start=1):
            bm25_score = float(scores[index])

            # Skip documents with no matching terms.
            if bm25_score <= 0:
                continue

            document = self.documents[index]

            results.append({
                **document,
                "bm25_score": round(bm25_score, 4),
                "bm25_rank": rank,
                "semantic_score": 0.0,
                "semantic_rank": 999,
                "score": round(1.0 / (60 + rank), 6),
                "rerank_score": round(bm25_score, 4),
            })

            if len(results) >= top_k:
                break

        return results
