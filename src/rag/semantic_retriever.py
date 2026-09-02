from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.chunking import chunk_knowledge_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"


def _build_chunk_text(
    chunk: dict[str, Any],
) -> str:
    """Build retrieval text from chunk metadata and content."""

    return (
        f"{chunk['section_path']}\n\n"
        f"{chunk['content']}"
    )


def retrieve_chunks(
    query: str,
    top_k: int = 5,
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> pd.DataFrame:
    """Retrieve the most relevant knowledge chunks using TF-IDF."""

    if not query.strip():
        raise ValueError("query must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    chunks = chunk_knowledge_directory(
        knowledge_dir
    )

    chunk_texts = [
        _build_chunk_text(chunk)
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    chunk_matrix = vectorizer.fit_transform(
        chunk_texts
    )

    query_vector = vectorizer.transform(
        [query]
    )

    similarity_scores = cosine_similarity(
        query_vector,
        chunk_matrix,
    ).flatten()

    results = pd.DataFrame(
        {
            "chunk_id": [
                chunk["chunk_id"]
                for chunk in chunks
            ],
            "source": [
                chunk["source"]
                for chunk in chunks
            ],
            "heading": [
                chunk["heading"]
                for chunk in chunks
            ],
            "section_path": [
                chunk["section_path"]
                for chunk in chunks
            ],
            "content": [
                chunk["content"]
                for chunk in chunks
            ],
            "similarity_score": similarity_scores,
        }
    )

    results = (
        results.sort_values(
            "similarity_score",
            ascending=False,
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    return results