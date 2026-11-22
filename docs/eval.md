# Evaluation guide

`seekr eval` measures retrieval at the **file level**: a ranking is the unique,
best-first list of files returned for one query, regardless of how many chunks
from that file matched. These measurements tell you whether the right files are
found, not whether a particular snippet or locator is ideal.

## Metrics

- `recall@5` / `recall@10`: the fraction of relevant files appearing in the
  de-duplicated top 5 / top 10 results, averaged over queries.
- `mrr@10`: reciprocal rank of the first relevant file (`1/rank`, or `0` when
  none is in the top 10), averaged over queries.
- `ndcg@10`: binary-gain nDCG with a `log2(rank + 1)` discount. Each relevant
  file in the top 10 contributes `1 / log2(rank + 1)`; the ideal DCG uses the
  number of relevant files actually available (capped at 10).
- `p50_ms`: median wall-clock time for the full `Searcher.get_hits` call,
  including query embedding and reranking when that mode uses them. Indexing
  and one untimed warm-up per mode are excluded, so lazy model loading is not
  reported as query latency. It is still real end-to-end search work, not a
  retrieval-only microbenchmark.

The harness indexes the supplied root into a throwaway temporary SQLite
database, runs every query through each requested mode with `per_file=1`, and
always requests at least ten files so the fixed `@5`/`@10` metrics are
meaningful. It never writes to your normal seekr index.

## Writing your own queries

A useful set has roughly **50 queries** and mixes two kinds:

1. **Exact / vocabulary** queries use distinctive terms that appear in the
   target file (function names, headings, product names, rare phrases). They
   check keyword and hybrid retrieval.
2. **Semantic / paraphrase** queries ask for the idea without using the file's
   exact words. They show whether vector search earns its place.
3. Add a few **multi-concept** or **near-miss** queries, and a few with more
   than one genuinely relevant file.

Good queries are short tasks or questions you would actually type, such as
"how do I reset the database safely?" or "WAL checkpoint". Avoid file paths,
secret values, or queries so broad that half the corpus is relevant. Judge
relevance by the file's content, not by what a particular mode happened to
return. Use the same query set for every mode; otherwise numbers are not
comparable. Keep queries and relevance labels versioned with the notes they
describe, and re-check labels when the underlying files change.

### Query JSONL format

One JSON object per line, with **relative POSIX paths** from the `--root`
directory (no leading `/`, no `./`, use `/` even on Windows):

```json
{"id": "q001", "query": "how do I reset the index safely?", "relevant": ["notes/seekr.md"]}
{"id": "q002", "query": "WAL checkpoint", "relevant": ["notes/sqlite.md", "notes/seekr.md"]}
```

Paths are compared after converting retrieved absolute paths back to
root-relative POSIX form. Duplicate paths in a ranking count once, at their
best rank. Empty lines are ignored.

## Running

Use real models and a real store through the normal CLI:

```bash
seekr eval \
  --root ~/notes \
  --queries ~/notes/eval-queries.jsonl \
  --modes keyword,vector,hybrid,hybrid+rerank \
  --k 10 \
  --out bench/notes-results.md
```

Omit `--modes` to run all four modes. `--k` only sets the minimum candidate
request depth; reported metrics remain `@5`/`@10`. The markdown table is
printed and, with `--out`, written to a file. A small versioned example corpus
lives in `eval/example`:

```bash
seekr eval \
  --root eval/example/corpus \
  --queries eval/example/queries.jsonl \
  --out bench/example-results.md
```

The example contains no precomputed results. Only commit measured results
produced by a real run on a stated corpus, and keep the command and model
names with them.

## Your own course

To evaluate retrieval across your own course folders, organise notes, slides,
notebooks, and code under a `<module>/weekN/` hierarchy, for example:

```text
~/Documents/ai-course/
├── machine-learning/
│   ├── week1/
│   └── week2/
├── data-structures/
│   ├── week1/
│   └── week2/
└── 4CCS1LGC-logic/
    └── week1/
```

Write your private query set in `eval/personal/queries.jsonl` (git-ignored so
personal course paths and queries never leave your machine), covering queries
across multiple modules and file formats (`.md`, `.pdf`, `.docx`, `.pptx`,
`.ipynb`, and code):

```json
{"id": "ml-01", "query": "bias variance tradeoff", "relevant": ["machine-learning/week2/bias-variance-tradeoff.md"]}
{"id": "ds-01", "query": "Dijkstra shortest paths", "relevant": ["data-structures/week3/shortest-paths.pptx"]}
```

Then run `seekr eval` against your course root and save the private report under
`eval/personal/results.md`:

```bash
seekr eval \
  --root <course> \
  --queries eval/personal/queries.jsonl \
  --out eval/personal/results.md
```

For example, pass `~/Documents/ai-course` as `<course>` when your files live
there. For a ready-made reference set, inspect the packaged demo course's query
file (`src/seekr/demo_course/queries.jsonl`).

## BEIR SciFact

`eval/beir_scifact.py` converts the public BEIR SciFact dataset into the same
layout. It uses the trusted public archive published by the UKP Lab at
TU Darmstadt:

<https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip>

The script is opt-in: importing it or using seekr never downloads anything.
Running it explicitly downloads only when `eval/data/scifact.zip` and the
extracted source files are absent. An existing valid ZIP is reused. The script
validates ZIP member paths (no absolute paths, `..` traversal, backslashes or
NUL bytes), rejects encrypted entries, caps both declared and actual
uncompressed sizes, and selectively writes only `corpus.jsonl`,
`queries.jsonl` and `qrels/test.tsv`. Data stays under the git-ignored
`eval/data/` directory.

```bash
python eval/beir_scifact.py
seekr eval \
  --root eval/data/scifact/corpus \
  --queries eval/data/scifact/queries.jsonl \
  --out bench/scifact-results.md
```

If the primary URL is unavailable, the script prints an actionable manual
fallback. You can also do it directly: open
<https://huggingface.co/datasets/BeIR/scifact/tree/main>, download
`corpus.jsonl`, `queries.jsonl` and `qrels/test.tsv`, place them under
`eval/data/scifact-source/` at those relative names, then run:

```bash
python eval/beir_scifact.py --no-download
```

`--no-download` guarantees no network access; it fails with the same fallback
instructions if the source files are missing. The conversion writes
`eval/data/scifact/corpus/<doc_id>.txt` as `title`, a blank line, then `text`.
Only test-split qrels with a score greater than zero count as relevant, and
queries without at least one relevant document present in the corpus are
skipped.

SciFact is a public benchmark; review its dataset terms at the source before
redistributing data. `eval/data/` is intentionally not committed.
