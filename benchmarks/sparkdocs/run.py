"""Evaluate SparkDocs against the bundled synthetic corpus.

Run from the repository root with uv run --locked python -m benchmarks.sparkdocs.run.
BM25 is the default; --local-models also loads the production FastEmbed models.
--live explicitly opts into a GPT-6 Luna answer pass using OPENAI_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from benchmarks.sparkdocs.fixtures import CORPUS, Evidence, Question
from scripts.openai_probe import read_key
from sparkdocs.chunking import build_chunks
from sparkdocs.errors import QuestionError
from sparkdocs.ingestion import ingest_upload
from sparkdocs.models import Chunk, DocumentIndex, Section
from sparkdocs.prompts import PROMPT_TEMPLATE_VERSION
from sparkdocs.providers import OpenAIProvider, TextProvider
from sparkdocs.retrieval import (
    FastEmbedDense,
    FastEmbedReranker,
    HybridRetriever,
    _bm25_ranking,
    _dense_ranking,
    _rrf,
    build_index,
)
from sparkdocs.service import ABSTENTION_MARKERS, DocumentQuestionService

MODEL = "gpt-6-luna"
REVIEW_FILE = Path(__file__).resolve().parents[2] / "benchmark-results" / "answer-review.json"
CANDIDATES = 20
RESULTS = 6
EVIDENCE_CHARACTERS = 24_000
MEDIA_TYPES = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def matches(section: Chunk | Section, label: Evidence) -> bool:
    same_location = section.locator == label.locator or section.locator.startswith(
        f"{label.locator} · chunk "
    )
    return same_location and normalize(label.text) in normalize(section.text)


def contains_phrase(answer: str, marker: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(normalize(marker))}(?!\w)", answer) is not None


def rank_of(evidence: Sequence[Chunk | Section], label: Evidence) -> int | None:
    return next((rank for rank, item in enumerate(evidence, 1) if matches(item, label)), None)


def select_evidence(chunks: Sequence[Chunk]) -> tuple[Section, ...]:
    evidence = []
    remaining = EVIDENCE_CHARACTERS
    for chunk in chunks[:RESULTS]:
        selected = chunk.text[:remaining]
        if not selected:
            break
        evidence.append(Section(chunk.locator, selected))
        remaining -= len(selected)
    return tuple(evidence)


@dataclass(frozen=True)
class Ranking:
    candidates: tuple[Chunk, ...]
    evidence: tuple[Section, ...]
    milliseconds: float


def rank_passages(
    index: DocumentIndex,
    question: str,
    embedder: FastEmbedDense | None,
    reranker: FastEmbedReranker | None,
) -> dict[str, Ranking]:
    def ranking(positions: list[int], seconds: float) -> Ranking:
        candidates = tuple(index.chunks[position] for position in positions)
        return Ranking(candidates, select_evidence(candidates), round(seconds * 1_000, 3))

    started = perf_counter()
    keyword = _bm25_ranking(index.chunks, question)[:CANDIDATES]
    keyword_seconds = perf_counter() - started
    rankings = {"bm25": ranking(keyword, keyword_seconds)}
    if embedder is None or reranker is None:
        return rankings

    started = perf_counter()
    dense = _dense_ranking(index, embedder.embed_query(question))[:CANDIDATES]
    dense_seconds = perf_counter() - started
    rankings["dense"] = ranking(dense, dense_seconds)

    started = perf_counter()
    fused = _rrf(dense, keyword)[:CANDIDATES]
    fusion_seconds = dense_seconds + keyword_seconds + perf_counter() - started
    rankings["hybrid"] = ranking(fused, fusion_seconds)

    started = perf_counter()
    reranked = fused
    if fused:
        scores = reranker.score(question, [index.chunks[position].text for position in fused])
        reranked = [
            position
            for _, position in sorted(
                zip(scores, fused, strict=True), key=lambda item: item[0], reverse=True
            )
        ]
    rankings["hybrid_rerank"] = ranking(reranked, fusion_seconds + perf_counter() - started)
    return rankings


@dataclass(frozen=True)
class FixedEvidence:
    evidence: tuple[Section, ...]

    def retrieve(
        self, index: DocumentIndex, question: str, *, max_characters: int
    ) -> tuple[Section, ...]:
        # The ranking step already applied the same limits as the answer service.
        return self.evidence


def retrieval_trace(ranking: Ranking, question: Question) -> dict:
    final_ranks = [rank_of(ranking.evidence, label) for label in question.evidence]
    return {
        "candidate_ranks": [rank_of(ranking.candidates, label) for label in question.evidence],
        "evidence_ranks": final_ranks,
        "evidence_locators": [section.locator for section in ranking.evidence],
        "all_required_in_evidence": all(rank is not None for rank in final_ranks)
        if final_ranks
        else None,
        "milliseconds": ranking.milliseconds,
    }


def evaluate_answer(
    provider: TextProvider,
    question: Question,
    index: DocumentIndex,
    evidence: tuple[Section, ...],
    *,
    review_answers: list[dict[str, str]] | None = None,
    stage: str = "retrieved_evidence",
) -> dict:
    if not evidence:
        return {"status": "empty_retrieval", "provider_called": False}
    service = DocumentQuestionService(provider, cast(HybridRetriever, FixedEvidence(evidence)))
    try:
        result = service.answer(index, question.text)
    except QuestionError:
        return {"status": "answer_validation_rejected", "provider_called": True}
    except Exception:
        # SDK exception messages can include request data. Record only the outcome.
        return {"status": "provider_error", "provider_called": True}

    if review_answers is not None:
        review_answers.append({"question_id": question.id, "stage": stage, "answer": result.answer})
    answer = normalize(result.answer)
    cited = set(result.cited_locators)
    cited_evidence = [section for section in evidence if section.locator in cited]
    abstention_marker = any(marker in answer for marker in ABSTENTION_MARKERS)
    outcome = {"status": "accepted_answer", "provider_called": True}
    outcome.update(
        {
            "cited_locators": list(result.cited_locators),
            "abstention_marker_detected": abstention_marker,
            "literal_answer_markers_present": (
                all(contains_phrase(answer, marker) for marker in question.answer_markers)
                if question.answer_markers
                else None
            ),
            "all_labeled_passages_cited": (
                all(rank_of(cited_evidence, label) is not None for label in question.evidence)
                if question.evidence
                else None
            ),
            "cited_without_labeled_support": (
                any(
                    not any(matches(section, label) for label in question.evidence)
                    for section in cited_evidence
                )
                if question.evidence
                else None
            ),
        }
    )
    return outcome


def summarize(rows: list[dict], modes: tuple[str, ...]) -> dict:
    retrieval = {}
    for mode in modes:
        traces = [row["retrieval"][mode] for row in rows]
        scored = [item for item in traces if item["all_required_in_evidence"] is not None]
        first_reciprocals = []
        for item in scored:
            ranks = [rank for rank in item["evidence_ranks"] if rank is not None]
            first_reciprocals.append(1 / min(ranks) if ranks else 0)
        retrieval[mode] = {
            "questions": len(traces),
            "answerable_questions": len(scored),
            "all_required_in_evidence_count": sum(
                item["all_required_in_evidence"] for item in scored
            ),
            "mean_reciprocal_rank_at_6": statistics.mean(first_reciprocals) if scored else None,
            "median_retrieval_ms": statistics.median(item["milliseconds"] for item in traces)
            if traces
            else None,
        }

    generated = [row for row in rows if "answer" in row]
    generation = {}
    if generated:
        generation["outcomes"] = dict(Counter(row["answer"]["status"] for row in generated))
        for answerable in (True, False):
            key = "answerable" if answerable else "unanswerable"
            cohort = [row["answer"] for row in generated if row["answerable"] == answerable]
            generation[key] = {
                "questions": len(cohort),
                "empty_retrieval_count": sum(x["status"] == "empty_retrieval" for x in cohort),
                "abstention_marker_count": sum(
                    x.get("abstention_marker_detected", False) for x in cohort
                ),
                "literal_answer_match_count": sum(
                    x.get("literal_answer_markers_present") is True for x in cohort
                ),
                "labeled_evidence_citation_count": sum(
                    x.get("all_labeled_passages_cited") is True for x in cohort
                ),
            }
    return {"retrieval": retrieval, "generation": generation}


def run(
    args: argparse.Namespace,
    provider: TextProvider | None,
    *,
    review_answers: list[dict[str, str]] | None = None,
) -> dict:
    setup_start = perf_counter()
    embedder = FastEmbedDense() if args.local_models else None
    reranker = FastEmbedReranker() if args.local_models else None
    setup_ms = round((perf_counter() - setup_start) * 1_000, 3)
    production_retriever = HybridRetriever(embedder, reranker) if embedder else None
    source_mode = "hybrid_rerank" if args.local_models else "bm25"
    modes = ("bm25", "dense", "hybrid", "hybrid_rerank") if args.local_models else ("bm25",)
    rows = []
    documents = []
    oracle_inputs = []
    calls = 0
    for fixture in CORPUS:
        if args.split != "all" and fixture.split != args.split:
            continue
        content = fixture.render()
        started = perf_counter()
        document = ingest_upload(
            filename=fixture.filename,
            claimed_media_type=MEDIA_TYPES[fixture.format],
            content=content,
        )
        extraction_ms = round((perf_counter() - started) * 1_000, 3)
        started = perf_counter()
        index = (
            build_index(document, embedder)
            if embedder
            else DocumentIndex(document.sha256, build_chunks(document), ())
        )
        indexing_ms = round((perf_counter() - started) * 1_000, 3)
        documents.append(
            {
                "id": fixture.id,
                "split": fixture.split,
                "format": fixture.format,
                "bytes": len(content),
                "chunks": len(index.chunks),
                "extraction_ms": extraction_ms,
                "indexing_ms": indexing_ms,
            }
        )
        for question in fixture.questions:
            for label in question.evidence:
                if rank_of(document.sections, label) is None:
                    raise ValueError(
                        f"Benchmark label does not match extracted source for {question.id}."
                    )
            rankings = rank_passages(index, question.text, embedder, reranker)
            row = {
                "document_id": fixture.id,
                "id": question.id,
                "split": fixture.split,
                "category": question.category,
                "answerable": question.expected_answer is not None,
                "retrieval": {
                    mode: retrieval_trace(ranking, question) for mode, ranking in rankings.items()
                },
            }
            if provider is not None:
                evidence = rankings[source_mode].evidence
                if production_retriever is not None:
                    production_evidence = production_retriever.retrieve(index, question.text)
                    if production_evidence != evidence:
                        raise ValueError(f"Production retrieval trace diverged for {question.id}.")
                if calls >= args.max_calls and evidence:
                    row["answer"] = {"status": "call_limit", "provider_called": False}
                else:
                    row["answer"] = evaluate_answer(
                        provider, question, index, evidence, review_answers=review_answers
                    )
                    calls += int(row["answer"]["provider_called"])
                if (
                    args.oracle
                    and row["answerable"]
                    and not row["retrieval"][source_mode]["all_required_in_evidence"]
                    and row["answer"]["status"] not in {"provider_error", "call_limit"}
                    and not (
                        row["answer"].get("literal_answer_markers_present") is True
                        and row["answer"].get("all_labeled_passages_cited") is True
                    )
                ):
                    oracle_inputs.append((row, question, index))
            rows.append(row)

    if provider is not None:
        # Finish the baseline pass before using any remaining quota on diagnosis.
        for row, question, index in oracle_inputs:
            if calls >= args.max_calls:
                break
            # Each labeled passage was verified against the extracted text above.
            expected = tuple(Section(label.locator, label.text) for label in question.evidence)
            if (
                not expected
                or len(expected) > RESULTS
                or sum(len(item.text) for item in expected) > EVIDENCE_CHARACTERS
            ):
                raise ValueError(f"Gold evidence exceeds the answer budget for {question.id}.")
            row["gold_evidence_answer"] = evaluate_answer(
                provider,
                question,
                index,
                expected,
                review_answers=review_answers,
                stage="gold_evidence",
            )
            calls += int(row["gold_evidence_answer"]["provider_called"])

    return {
        "schema_version": 1,
        "platform": platform.system(),
        "python": platform.python_version(),
        "source_commit": args.source_commit,
        "split": args.split,
        "generation_model": MODEL if provider is not None else None,
        "generation_evidence_mode": source_mode if provider is not None else None,
        "prompt_template_version": PROMPT_TEMPLATE_VERSION if provider is not None else None,
        "provider_calls": calls,
        "model_setup_ms": setup_ms if args.local_models else None,
        "candidate_limit": CANDIDATES,
        "result_limit": RESULTS,
        "evidence_character_limit": EVIDENCE_CHARACTERS,
        "documents": documents,
        "summary": {
            split: summarize([row for row in rows if row["split"] == split], modes)
            for split in ("development", "holdout")
            if any(row["split"] == split for row in rows)
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("all", "development", "holdout"), default="all")
    parser.add_argument("--local-models", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-file", type=Path, help="Read only OPENAI_API_KEY for a live run.")
    parser.add_argument("--oracle", action="store_true", help="Rerun misses with labeled evidence.")
    parser.add_argument(
        "--review-answers",
        action="store_true",
        help=(
            "Save accepted synthetic answers for local review "
            "in benchmark-results/answer-review.json."
        ),
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=40,
        help="Maximum provider invocations. SDK-level HTTP retries are separate.",
    )
    parser.add_argument("--source-commit", help="Full Git commit for this run's SparkDocs source.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.max_calls < 1:
        parser.error("--max-calls must be positive")
    if args.oracle and not args.live:
        parser.error("--oracle requires --live")
    if args.env_file and not args.live:
        parser.error("--env-file requires --live")
    if args.review_answers and not args.live:
        parser.error("--review-answers requires --live")
    if args.report and args.report.resolve() == REVIEW_FILE.resolve():
        parser.error("--report must use a different path from the local answer review file")
    if args.source_commit and not re.fullmatch("[0-9a-fA-F]{40}", args.source_commit):
        parser.error("--source-commit must be a full 40-character Git commit")
    provider = None
    if args.live:
        try:
            key = read_key(args.env_file) if args.env_file else os.environ.get("OPENAI_API_KEY", "")
        except (OSError, UnicodeError):
            parser.error("Could not read the supplied env file.")
        key = key.strip()
        if not key:
            parser.error("--live requires OPENAI_API_KEY in the environment or supplied env file")
        provider = OpenAIProvider(key, model=MODEL)
    review_answers: list[dict[str, str]] | None = [] if args.review_answers else None
    report = run(args, provider, review_answers=review_answers)
    if args.report:
        report_file = args.report
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if review_answers is not None:
        REVIEW_FILE.parent.mkdir(parents=True, exist_ok=True)
        REVIEW_FILE.write_text(json.dumps(review_answers, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"provider_calls": report["provider_calls"], "summary": report["summary"]}, indent=2
        )
    )
    if args.live:
        provider_failed = any(
            row.get(stage, {}).get("status") == "provider_error"
            for row in report["cases"]
            for stage in ("answer", "gold_evidence_answer")
        )
        if not report["provider_calls"] or provider_failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
