# When does SparkDocs find the right passage?

SparkDocs sends a language model the evidence it retrieves from one document. I wanted to see whether local semantic search and reranking help it find evidence that keyword search misses, and whether the model can tell when the retrieved text does not answer the question.

## What I measured

The results below are from single-turn queries. The chat interface was added later.

The study uses six fictional documents, two each in PDF, DOCX, and PPTX. Each document has five labeled questions covering an exact identifier, a paraphrase, similar sections with different rules, a question that needs two passages, and a missing answer. The development and holdout splits use different documents. Treat the current holdout as evaluated. If I tune the pipeline using its misses, I will bring fresh documents for the next holdout.

The runner compares BM25, dense retrieval, hybrid retrieval, and hybrid retrieval with the production reranker. The default run measures BM25 using SparkDocs ingestion, chunking, and keyword scoring. The optional model run uses SparkDocs' FastEmbed models. Each configuration uses the same candidate, final passage, and character budgets. I count how often every required passage reaches the final six and report the rank of the first supporting passage and query time after model construction and document indexing. These short fixtures exercise evidence selection, with limited coverage of long documents and candidate cutoffs.

## The baseline miss

BM25 returned every labeled passage for 12 of 12 answerable development questions and 11 of 12 answerable holdout questions. Each split also contains three unanswerable questions. These counts measure evidence retrieval.

The holdout miss asked how much time someone has to submit a power-cell claim for a unit used outdoors. Page 3 of the Maple claims handbook says outdoor battery claims must be filed within seven calendar days of discovery. Page 2 gives indoor devices 30 days. BM25 returned pages 5 and 6, which discuss claims but contain no deadline for outdoor devices. The supporting page never entered the candidate list. The query says *power-cell* and *outdoors*, while the source says *battery* and *outdoor*. The keyword ranker matches literal tokens and does not stem them.

## What GPT-6 Luna did with the evidence

I ran GPT-6 Luna on the 15 holdout questions using BM25 evidence, plus one replay with the labeled evidence. Eleven of 12 answerable questions passed the literal answer and labeled-citation checks. All three unanswerable questions matched an abstention phrase. Citation labels identify the excerpts Luna saw. For an abstention, a citation does not establish that the full document lacks the answer.

On the Maple question, Luna abstained when it saw pages 5 and 6. With page 3 supplied directly, it answered seven calendar days from discovery and cited page 3. The replay shows that supplying the missing passage recovered the answer in this run.

## What dense search and reranking changed

| Retrieval mode | All required passages in final six | MRR@6 | Median query time | Maple page 3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 11/12 | 0.875 | 0.185 ms | Absent |
| Dense | 12/12 | 1.000 | 6.527 ms | 1st |
| Hybrid | 12/12 | 0.944 | 6.739 ms | 3rd |
| Hybrid with reranking | 12/12 | 0.958 | 53.975 ms | 1st |

Dense search recovered page 3. Fusion moved it behind two keyword matches, and reranking moved it back to first. Across the holdout, reranking improved the fused ranking without adding evidence coverage. Dense-only ranked the first expected passage first for every answerable question and had the lowest median query time of the model-assisted modes. These results make dense-only worth comparing on fresh documents before changing the production default.

## What I kept

With production retrieval feeding the same GPT-6 Luna model, I counted 12 supported answers out of 12 answerable holdout questions and three abstentions out of three unanswerable questions. All 15 responses passed service validation. The automated abstention check counted two. The third response said the document did not specify a minimum device weight for pickup, which I counted as an abstention after reading it.

The production evidence included both the seven-day outdoor policy and the 30-day indoor policy. Luna gave the seven-day deadline and cited page 3. That recovered the original miss without supplying the labeled passage by hand.

I kept the production pipeline. Dense-only earned a follow-up on fresh documents because it covered the same evidence, ranked the first supporting passage higher, and queried faster on this holdout. This run measures live answer quality for BM25 and the production pipeline. It leaves dense-only answer quality and behavior on longer documents open.

## Follow-up questions in chat

SparkDocs now starts each turn with a fresh evidence search. It can use up to two recent user
questions, capped at 600 characters in total, as context for retrieval and answer generation.
The answer shows its own expandable sources. Citation validation checks those sources against
the evidence selected for that turn.

In a local retrieval check with the built-in Maple sample, the outdoor battery claim question ranked
page 3 first. The follow-up *What about indoor ones?* ranked page 2 first both with and without the
earlier question as context. The passages specify seven days outdoors and 30 days indoors. This
check confirms the expected evidence was retrieved on both turns. It does not establish a ranking
improvement from context or measure live multi-turn answer quality.

## Reproduce

Run from the SparkDocs repository root.

```powershell
uv run --locked python -m benchmarks.sparkdocs.run --report benchmark-results/bm25.json
uv run --locked python -m benchmarks.sparkdocs.run --local-models --report benchmark-results/retrieval.json
```

For live GPT-6 Luna generation, supply `OPENAI_API_KEY` in the process environment or pass `--env-file` with a local dotenv path. These commands send only the bundled fictional test inputs. The invocation limit excludes SDK retries.

```powershell
uv run --locked python -m benchmarks.sparkdocs.run --live --split holdout --oracle --review-answers --max-calls 16 --report benchmark-results/luna-holdout.json
uv run --locked python -m benchmarks.sparkdocs.run --local-models --live --split holdout --review-answers --max-calls 15 --report benchmark-results/luna-hybrid-holdout.json
```

The optional gold-evidence replay sends labeled passages directly for selected misses. The answer review file is replaced on each live run that enables it. Review each answer against the evidence for its question before treating citation or phrase matches as quality scores. Git ignores the answer review file and raw reports.

The [sanitized evidence summary](retrieval-case-study-evidence.json) records the aggregate scores, the Maple trace, and hashes of the reviewed source files. The original runs did not record a Git commit. To pin a future run, supply the full commit with `--source-commit`.
