
import json
import uuid
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from app.retrieval.hybrid_search import (
    DATA_PATH,
    HybridSearch,
    load_chunks,
)
from app.rag.answer_generator import generate_answer


app = FastAPI(
    title="AI Investigation Engine",
    description="Evidence-grounded question answering across documents",
    version="0.3.1",
)


# Temporary limit for Render's 512 MB RAM instance.
# This limits loaded chunks, not the number of source documents read.
MAX_SEARCH_CHUNKS = 3000


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class DocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=1, max_length=100000)
    source: str = Field(default="manual", max_length=300)


def read_documents() -> list[dict]:
    """Read source documents from the JSONL file."""
    if not DATA_PATH.exists():
        return []

    documents = []

    with DATA_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                documents.append(json.loads(line))

    return documents


def write_documents(documents: list[dict]) -> None:
    """Save source documents and invalidate the cached search engine."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DATA_PATH.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    get_search_engine.cache_clear()


@lru_cache(maxsize=1)
def get_search_engine() -> HybridSearch:
    """Build and cache a limited-size search index."""
    print(
        f"Loading up to {MAX_SEARCH_CHUNKS} search chunks...",
        flush=True,
    )

    documents = load_chunks(
        limit=MAX_SEARCH_CHUNKS,
        strategy="fixed",
    )

    print(
        f"Loaded {len(documents)} chunks. Building search index...",
        flush=True,
    )

    if not documents:
        raise HTTPException(
            status_code=503,
            detail="No searchable documents are available.",
        )

    search_engine = HybridSearch(documents)

    print("Search index is ready.", flush=True)
    return search_engine


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "AI Investigation Engine",
    }


@app.post("/ask")
def ask_question(request: QuestionRequest):
    print("Received /ask request.", flush=True)

    search_engine = get_search_engine()

    print("Search engine ready. Searching...", flush=True)

    results = search_engine.search(
        request.question,
        top_k=5,
        candidate_k=100,
    )

    print(
        f"Search finished. Results: {len(results)}",
        flush=True,
    )

    if not results or all(
        result.get("bm25_score", 0) <= 0
        for result in results
    ):
        return {
            "question": request.question,
            "answer": None,
            "citations": [],
            "abstained": True,
            "message": "No matching evidence was found.",
        }

    response = generate_answer(request.question, results)

    return {
        "question": request.question,
        **response,
    }


@app.get("/documents")
def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    documents = read_documents()

    return {
        "data_path": str(DATA_PATH.resolve()),
        "file_exists": DATA_PATH.exists(),
        "total": len(documents),
        "limit": limit,
        "offset": offset,
        "documents": documents[offset:offset + limit],
    }


@app.post("/documents", status_code=201)
def add_document(request: DocumentRequest):
    documents = read_documents()

    document = {
        "id": f"manual-{uuid.uuid4().hex[:12]}",
        "title": request.title,
        "text": request.text,
        "source": request.source,
    }

    documents.append(document)
    write_documents(documents)

    return {
        "message": "Document added successfully.",
        "document": document,
    }


@app.put("/documents/{document_id}")
def update_document(
    document_id: str,
    request: DocumentRequest,
):
    documents = read_documents()

    for index, document in enumerate(documents):
        if document.get("id") == document_id:
            updated_document = {
                **document,
                "title": request.title,
                "text": request.text,
                "source": request.source,
            }

            documents[index] = updated_document
            write_documents(documents)

            return {
                "message": "Document updated successfully.",
                "document": updated_document,
            }

    raise HTTPException(
        status_code=404,
        detail="Document not found.",
    )


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    documents = read_documents()

    remaining = [
        document
        for document in documents
        if document.get("id") != document_id
    ]

    if len(remaining) == len(documents):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    write_documents(remaining)

    return {
        "message": "Document deleted successfully.",
        "id": document_id,
    }
