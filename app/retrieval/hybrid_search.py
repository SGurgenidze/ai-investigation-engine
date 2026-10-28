import json
import re
from pathlib import Path

from app.ingestion.chunking import (
    fixed_size_chunks,
    sentence_based_chunks,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "documents.jsonl"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def tokenize(text: str) -> list[str]:
    """Normalize text and remove common English stop words."""
    stop_words = {
        "who", "was", "is", "the", "of", "a", "an",
        "in", "on", "for", "to", "and", "did", "do",
        "does", "what", "which", "that", "by", "with",
        "movie", "film"
    }

    tokens = re.findall(r"\b\w+\b", text.casefold())
    return [token for token in tokens if token not in stop_words]

print("Project root:", PROJECT_ROOT)
print("Data path:", DATA_PATH)
print("File exists:", DATA_PATH.exists())

def load_chunks(
    limit: int | None = None,
    strategy: str = "sentence",
) -> list[dict]:
    """Load documents and split their text into chunks."""
    if strategy == "sentence":
        chunker = sentence_based_chunks
    elif strategy == "fixed":
        chunker = fixed_size_chunks
    else:
        raise ValueError("strategy must be 'sentence' or 'fixed'")

    chunks = []

    if DATA_PATH.exists():
        with DATA_PATH.open(encoding="utf-8") as file:
            for line in file:
                document = json.loads(line)

                for index, chunk_text in enumerate(
                    chunker(document["text"])
                ):
                    chunks.append(
                        {
                            "id": f'{document["id"]}-chunk-{index}',
                            "title": document.get("title", "Untitled"),
                            "text": chunk_text,
                            "source": document.get("source", "unknown"),
                        }
                    )

                if limit is not None and len(chunks) >= limit:
                    return chunks[:limit]

    return chunks


class HybridSearch:
    def __init__(self, documents: list[dict]):
        if not documents:
            raise ValueError("At least one document is required")

        from rank_bm25 import BM25Okapi

        self.documents = documents

        print("Building BM25 index...", flush=True)
        self.bm25 = BM25Okapi(
            [tokenize(document["text"]) for document in documents]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 200,
    ) -> list[dict]:
        if not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1")

        # 1. Calculate BM25 scores across the full corpus
        keyword_scores = self.bm25.get_scores(tokenize(query))

        # 2. Rank all documents by BM25
        keyword_ranking = sorted(
            range(len(self.documents)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )

        keyword_ranks = {
            index: rank
            for rank, index in enumerate(keyword_ranking, start=1)
        }

        # 3. Select top candidates
        candidate_indices = keyword_ranking[:candidate_k]

        results = []
        for index in candidate_indices:
            bm25_score = float(keyword_scores[index])
            bm25_rank = keyword_ranks[index]

            # მსუბუქი სკორი სემანტიკურის გარეშე (ყველაფერი ეფუძნება BM25-ს და რანგს)
            score = 1.0 / (60 + bm25_rank)

            results.append({
                **self.documents[index],
                "bm25_score": round(bm25_score, 4),
                "bm25_rank": bm25_rank,
                "semantic_score": 0.0,
                "semantic_rank": 999,
                "score": round(score, 6),
                "rerank_score": round(bm25_score, 4), # იმისათვის, რომ evaluation არ დაინგრეს
            })

        # 4. Sort by score
        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]
            reverse=True,
        )

        return results[:top_k]
