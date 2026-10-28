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
        from sentence_transformers import SentenceTransformer, util

        self.util = util
        self.documents = documents

        print("Building BM25 index...", flush=True)
        self.bm25 = BM25Okapi(
            [tokenize(document["text"]) for document in documents]
        )

        print("Loading sentence transformer model...", flush=True)
        self.model = SentenceTransformer(MODEL_NAME)

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

        keyword_scores = self.bm25.get_scores(tokenize(query))

  
        keyword_ranking = sorted(
            range(len(self.documents)),
            key=lambda i: keyword_scores[i],
            reverse=True,
        )

        keyword_ranks = {
            index: rank
            for rank, index in enumerate(keyword_ranking, start=1)
        }

        candidate_indices = keyword_ranking[:candidate_k]

        texts = [query] + [
            self.documents[index]["text"]
            for index in candidate_indices
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        semantic_scores = self.util.cos_sim(
            embeddings[0],
            embeddings[1:],
        )[0].cpu().tolist()

        semantic_order = sorted(
            range(len(candidate_indices)),
            key=lambda i: semantic_scores[i],
            reverse=True,
        )

        semantic_ranks = {
            candidate_indices[position]: rank
            for rank, position in enumerate(semantic_order, start=1)
        }

        rrf_k = 60
        results = []

        for position, index in enumerate(candidate_indices):
            bm25_rank = keyword_ranks[index]
            semantic_rank = semantic_ranks[index]

            rrf_score = (
                1 / (rrf_k + bm25_rank)
                + 1 / (rrf_k + semantic_rank)
            )

            results.append({
                **self.documents[index],
                "bm25_score": round(float(keyword_scores[index]), 4),
                "bm25_rank": bm25_rank,
                "semantic_score": round(
                    float(semantic_scores[position]), 4
                ),
                "semantic_rank": semantic_rank,
                "score": round(rrf_score, 6),
                "rerank_score": round(rrf_score, 6), 
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:top_k]
