"""Hybrid dense/BM25 retrieval, RRF fusion, and optional cross-encoder reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, cast

from sparkdocs.chunking import build_chunks
from sparkdocs.models import Chunk, DocumentIndex, IngestedDocument, Section

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+(?:[.:/-][a-zA-Z0-9_]+)*")
RRF_CONSTANT = 60
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "is",
    "it",
    "of",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class DenseEmbedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...

    def embed_query(self, text: str) -> Sequence[float]: ...


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]: ...


class FastEmbedDense:
    """Lazy local ONNX embedding adapter; model weights download on first construction."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [
            cast(Sequence[float], vector.tolist())
            for vector in self._model.passage_embed(list(texts))
        ]

    def embed_query(self, text: str) -> Sequence[float]:
        vector = next(iter(self._model.query_embed(text)))
        return cast(Sequence[float], vector.tolist())


class FastEmbedReranker:
    """Lazy local cross-encoder adapter for second-stage precision."""

    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        values: Any = self._model.rerank(query, list(documents))
        return [float(value) for value in values]


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.lower()) if token not in STOP_WORDS]


def build_index(document: IngestedDocument, embedder: DenseEmbedder) -> DocumentIndex:
    chunks = build_chunks(document)
    contextual_texts = [
        f"Document: {document.filename}\nLocation: {chunk.locator}\n{chunk.text}"
        for chunk in chunks
    ]
    vectors = embedder.embed_documents(contextual_texts)
    if not chunks or len(vectors) != len(chunks):
        raise ValueError("The embedding provider returned an unexpected vector count.")
    embeddings = tuple(tuple(float(value) for value in vector) for vector in vectors)
    if any(not vector for vector in embeddings) or len({len(vector) for vector in embeddings}) != 1:
        raise ValueError("The embedding provider returned invalid vector dimensions.")
    return DocumentIndex(document.sha256, chunks, embeddings)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Query and document embedding dimensions do not match.")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _dense_ranking(index: DocumentIndex, query_vector: Sequence[float]) -> list[int]:
    scores = [
        (_cosine(query_vector, vector), position)
        for position, vector in enumerate(index.embeddings)
    ]
    return [
        position
        for score, position in sorted(scores, key=lambda item: item[0], reverse=True)
        if score > 0.05
    ]


def _bm25_ranking(chunks: Sequence[Chunk], question: str) -> list[int]:
    query = _tokens(question)
    if not query:
        return []
    documents = [_tokens(chunk.text) for chunk in chunks]
    average_length = sum(map(len, documents)) / max(len(documents), 1)
    document_frequency = Counter(token for token in set(query) for doc in documents if token in doc)
    scored: list[tuple[float, int]] = []
    for position, tokens in enumerate(documents):
        frequencies = Counter(tokens)
        score = 0.0
        for token in set(query):
            frequency = frequencies[token]
            if not frequency:
                continue
            frequency_docs = document_frequency[token]
            inverse_frequency = math.log(
                1 + (len(documents) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + 1.5 * (0.25 + 0.75 * len(tokens) / max(average_length, 1))
            score += inverse_frequency * frequency * 2.5 / denominator
        if score:
            scored.append((score, position))
    return [position for _, position in sorted(scored, key=lambda item: item[0], reverse=True)]


def _rrf(*rankings: Sequence[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, position in enumerate(ranking, start=1):
            scores[position] = scores.get(position, 0.0) + 1 / (RRF_CONSTANT + rank)
    return [
        position for position, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


class HybridRetriever:
    def __init__(self, embedder: DenseEmbedder, reranker: Reranker | None = None) -> None:
        self._embedder = embedder
        self._reranker = reranker

    def retrieve(
        self,
        index: DocumentIndex,
        question: str,
        *,
        candidate_limit: int = 20,
        result_limit: int = 6,
        max_characters: int = 24_000,
    ) -> tuple[Section, ...]:
        if candidate_limit < 1 or result_limit < 1 or max_characters < 1:
            raise ValueError("Retrieval limits must be positive.")
        dense = _dense_ranking(index, self._embedder.embed_query(question))[:candidate_limit]
        keyword = _bm25_ranking(index.chunks, question)[:candidate_limit]
        fused = _rrf(dense, keyword)[:candidate_limit]
        if not fused:
            return ()
        if self._reranker is not None:
            scores = self._reranker.score(
                question, [index.chunks[position].text for position in fused]
            )
            if len(scores) != len(fused):
                raise ValueError("The reranker returned an unexpected score count.")
            fused = [
                position
                for _, position in sorted(
                    zip(scores, fused, strict=True), key=lambda item: item[0], reverse=True
                )
            ]
        selected: list[Section] = []
        remaining = max_characters
        for position in fused[:result_limit]:
            if remaining <= 0:
                break
            chunk = index.chunks[position]
            text = chunk.text[:remaining]
            if text:
                selected.append(Section(chunk.locator, text))
                remaining -= len(text)
        return tuple(selected)
