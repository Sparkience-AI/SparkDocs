"""Evaluate evidence selection on a fresh set of synthetic local documents.

Run with `uv run --locked python -m benchmarks.sparkdocs.challenge`.
Add `--local-models` to evaluate the production embedding and reranking models.
The report contains no generated answers and makes no cloud provider calls.
"""

import argparse
import json
from io import BytesIO
from pathlib import Path

from docx import Document

from benchmarks.sparkdocs.fixtures import Evidence, Fixture, Question
from benchmarks.sparkdocs.run import MEDIA_TYPES, rank_of, rank_passages, retrieval_trace, summarize
from sparkdocs.chunking import build_chunks
from sparkdocs.ingestion import ingest_upload
from sparkdocs.models import DocumentIndex
from sparkdocs.retrieval import FastEmbedDense, FastEmbedReranker, HybridRetriever, build_index

PDF_POLICY = """
Seasonal lockers are available near the east entrance. Patrons request an allocation once their
first shift has been confirmed. The sign-out sheet lists lockers that are available and the
end date for each allocation. Ask the facilities desk if the assigned locker is too small.

At shift handover, check that any supplies left in the common cupboard have a name and return
date. Loose supplies go back to the shelf beside the entrance. Mark damaged lockers out of
circulation until the facilities desk clears the inspection. Do not place equipment in them.

Inside the locker area, leave the emergency exit and the marked walkway clear. Visitors wait on
the benches while the shift coordinator checks their passes. Keep the sign-out sheet on the
clipboard and return it to the desk when the building closes. Staff check it each Friday.

The allocation ends after the season schedule is published. Remove personal equipment, tell the
coordinator that the locker is empty, and return issued keys. The equipment team keeps a separate
list of tagged boxes, shared radios, and spare batteries. Report a missing tag to the desk.

A key may carry a faded inventory number. Note the location and record the number if readable.
Check the distribution sheet for spare sets kept in the staff cupboard at the loading entrance.
For a locker key lost during a weekend shift, notify the duty manager within 90 minutes.
The duty manager records the replacement in the staff inventory. Volunteers and staff can
refer to the facilities desk for any subsequent questions about key retention.
"""


def question(question_id: str, text: str, category: str, *evidence: tuple[str, str]) -> Question:
    return Question(
        id=question_id,
        text=text,
        expected_answer=" ".join(span for _, span in evidence) if evidence else None,
        evidence=tuple(Evidence(locator, span) for locator, span in evidence),
        category=category,
        answer_markers=(),
    )


PDF = Fixture(
    id="birch-locker-handbook",
    split="holdout",
    format="pdf",
    passages=(
        "For a locker key lost during a weekday shift, notify the facilities desk within 4 hours.",
        PDF_POLICY,
        "Locker assignment requests are accepted through the sign-out sheet beside the east exit.",
    ),
    questions=(
        question(
            "birch-01",
            "How soon must I report a missing key if it happens on a weekend shift?",
            "chunked_lookalike",
            ("PDF page 2", "notify the duty manager within 90 minutes"),
        ),
        question("birch-02", "What is the locker key replacement fee?", "missing_answer"),
    ),
)

SLIDES = Fixture(
    id="north-bridge-drill",
    split="holdout",
    format="pptx",
    passages=(
        "Drill policy for North Bridge. The regular radio test begins at 10:00 on Wednesdays.",
        "After a missed radio check-in, use backup channel 4 and report the operator's name.",
        "If backup channel 4 still has no response, notify the incident lead after 8 minutes.",
        "After a radio check-in with distorted audio, test channel 6 and record interference.",
    ),
    questions=(
        question(
            "north-01",
            "Which channel should I use after a missed check-in, and when do I escalate if silent?",
            "multi_slide",
            ("PPTX slide 2", "use backup channel 4"),
            ("PPTX slide 3", "notify the incident lead after 8 minutes"),
        ),
        question("north-02", "How long must a radio drill report be retained?", "missing_answer"),
    ),
)

TABLE_QUESTIONS = (
    question(
        "aspen-01",
        "Who can approve an international standard return and how long is the window?",
        "table_lookalike",
        ("DOCX table 1, row 3", "International standard | 14 calendar days | Returns lead"),
    ),
    question(
        "aspen-02",
        "Which form covers a damaged return and where do I record its approval?",
        "paragraph_and_table",
        ("DOCX paragraph 2", "Form ASP-17 is for damaged returns"),
        ("DOCX table 1, row 4", "Damage register"),
    ),
    question("aspen-03", "How much does return shipping cost?", "missing_answer"),
)


def render_return_policy() -> bytes:
    document = Document()
    document.add_paragraph("Aspen returns policy. The windows below start at delivery.")
    document.add_paragraph("Form ASP-17 is for damaged returns. Submit the form for approval.")
    table = document.add_table(rows=0, cols=3)
    for values in (
        ("Return type", "Window", "Approver or record"),
        ("Domestic standard", "30 calendar days", "Service desk"),
        ("International standard", "14 calendar days", "Returns lead"),
        ("Damaged return", "3 business days", "Damage register"),
        ("Damaged delivery", "2 business days", "Courier incident register"),
    ):
        for cell, value in zip(table.add_row().cells, values, strict=True):
            cell.text = value
    destination = BytesIO()
    document.save(destination)
    return destination.getvalue()


def run(*, use_local_models: bool) -> dict:
    embedder = FastEmbedDense() if use_local_models else None
    reranker = FastEmbedReranker() if use_local_models else None
    production = HybridRetriever(embedder, reranker) if embedder else None
    rows = []
    for filename, kind, content, questions in (
        (PDF.filename, PDF.format, PDF.render(), PDF.questions),
        (SLIDES.filename, SLIDES.format, SLIDES.render(), SLIDES.questions),
        ("aspen-returns.docx", "docx", render_return_policy(), TABLE_QUESTIONS),
    ):
        document = ingest_upload(
            filename=filename, claimed_media_type=MEDIA_TYPES[kind], content=content
        )
        index = (
            build_index(document, embedder)
            if embedder
            else DocumentIndex(document.sha256, build_chunks(document), ())
        )
        for case in questions:
            for label in case.evidence:
                if rank_of(index.chunks, label) is None:
                    raise ValueError(f"Labeled span was not preserved by ingestion for {case.id}.")
            rankings = rank_passages(index, case.text, embedder, reranker)
            if (
                production
                and production.retrieve(index, case.text) != rankings["hybrid_rerank"].evidence
            ):
                raise ValueError(f"Benchmark evidence does not match production for {case.id}.")
            rows.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "expected_locators": [label.locator for label in case.evidence],
                    "retrieval": {
                        mode: retrieval_trace(ranking, case) for mode, ranking in rankings.items()
                    },
                }
            )

    retrieval = summarize(rows, tuple(rows[0]["retrieval"]))["retrieval"]
    return {"retrieval": retrieval, "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-models", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(use_local_models=args.local_models)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["retrieval"], indent=2))


if __name__ == "__main__":
    main()
