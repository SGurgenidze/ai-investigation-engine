
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
        raise ValueError(
            "strategy must be 'sentence' or 'fixed'"
        )

    chunks = []

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
        from sentence_transformers import SentenceTransformer, CrossEncoder, util

        self.util = util
        self.documents = documents

        print("Building BM25 index...", flush=True)
        self.bm25 = BM25Okapi(
            [tokenize(document["text"]) for document in documents]
        )

        print("Loading sentence transformer model...", flush=True)
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading cross-encoder reranker...", flush=True)
        self.reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
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

        # Store each document's BM25 rank for fast lookup
        keyword_ranks = {
            index: rank
            for rank, index in enumerate(keyword_ranking, start=1)
        }

        # 3. Select the top BM25 candidates
        candidate_indices = keyword_ranking[:candidate_k]

        # 4. Encode the query and candidate documents
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

        # 5. Calculate semantic similarity
        semantic_scores = self.util.cos_sim(
            embeddings[0],
            embeddings[1:],
        )[0].cpu().tolist()

        # 6. Rank candidates by semantic similarity
        semantic_order = sorted(
            range(len(candidate_indices)),
            key=lambda i: semantic_scores[i],
            reverse=True,
        )

        semantic_ranks = {
            candidate_indices[position]: rank
            for rank, position in enumerate(semantic_order, start=1)
        }

        # 7. Combine rankings using Reciprocal Rank Fusion (RRF)
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
            })

        # 8. Rerank candidates using a cross-encoder
        pairs = [
            (query, f"{item['title']}. {item['text']}")
            for item in results
        ]

        rerank_scores = self.reranker.predict(pairs)

        for item, rerank_score in zip(results, rerank_scores):
            item["rerank_score"] = float(rerank_score)

        results.sort(
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        return results[:top_k]

def main():
    print("PROGRAM STARTED", flush=True)

    documents = load_chunks()
    print("Loaded chunks:", len(documents), flush=True)

    search_engine = HybridSearch(documents)

    query = input("\nEnter your search query: ").strip()

    # Check whether relevant chunks appear in BM25's top 200
    keyword_scores = search_engine.bm25.get_scores(
        tokenize(query)
    )

    ranking = sorted(
        range(len(documents)),
        key=lambda i: keyword_scores[i],
        reverse=True,
    )

    print("\n--- CHECKING TOP 200 CANDIDATES ---")

    found = False

    for rank, index in enumerate(ranking[:200], start=1):
        document = documents[index]
        text_lower = document["text"].casefold()

        if "james cameron" in text_lower and "titanic" in text_lower:
            print("BM25 rank:", rank)
            print("Title:", document["title"])
            print("Text:", document["text"][:500])
            print()
            found = True

    if not found:
        print("No matching chunk found in top 200.")

    # Run hybrid search
    results = search_engine.search(
        query,
        top_k=5,
        candidate_k=200,
    )

    print("\n--- HYBRID SEARCH RESULTS ---")

    for result in results:
        print("\nTitle:", result["title"])
        print("BM25 rank:", result["bm25_rank"])
        print("Semantic rank:", result["semantic_rank"])
        print("BM25 score:", result["bm25_score"])
        print("Semantic score:", result["semantic_score"])
        print("RRF score:", result["score"])
        print("Text:", result["text"][:500])

    print("\nPROGRAM FINISHED", flush=True)


if __name__ == "__main__":
    main()


