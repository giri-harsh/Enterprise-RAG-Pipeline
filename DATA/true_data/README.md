# Demo corpus — `DATA/true_data/`

This directory is **tracked in git**. It is the knowledge base the demo runs
against and the corpus all 15 golden evaluation samples are written for.

## Why it is tracked

It is small (~2.9 MB total), it is all original or freely-redistributable content,
and without it the demo produces no useful answers — retrieval returns nothing, and
every question gets "the documentation does not cover this". Tracking it makes a
fresh clone immediately runnable.

## Contents

| File | Format | Size | Topic |
|---|---|---|---|
| `architecture.pptx` | PowerPoint | 2.7 MB | Kubernetes architecture overview — nodes, control plane, scheduler, etcd |
| `pods_autoscale.html` | HTML | 26 KB | Horizontal Pod Autoscaler — metrics, scaling algorithm, behaviour |
| `parallel_work_queue.txt` | Plain text | 5 KB | Running a Redis-backed parallel work queue job in Kubernetes |
| `job_management.html` | HTML | 14 KB | Kubernetes Job and CronJob management — spec fields, completions, parallelism |
| `cronjobs.docx` | Word | 64 KB | CronJob schedule format, timezone handling, concurrency policy |
| `monitor_job.docx` | Word | 25 KB | Monitoring Kubernetes jobs — conditions, status, pod log inspection |

**Total: 6 files, ~2.9 MB.**

## Indexing

```bash
python -m app.ingestion.processor DATA/true_data true --wipe
```

All six files are parsed, chunked, and embedded (768-dim in local mode, 3072-dim
in cloud mode), then upserted into Qdrant tagged `source_type: "true"`. The
`--wipe` flag drops and recreates the collection, which is required on a first run
and whenever you switch between local and cloud embedding models (the vector width
changes from 768 to 3072 and the old vectors cannot be queried).

Indexing takes roughly one minute on first run — sentence-transformers downloads
~420 MB of model weights and FlashRank fetches a small ONNX file. Both are cached
and subsequent runs are fast.

## Relationship to the evaluation suite

`evals/golden_dataset.json` contains 15 question-answer pairs and 6 guardrail
test cases written specifically against this corpus. The RAGAS scores (faithfulness,
relevancy, context precision/recall, correctness) are meaningful only if the index
holds this content.

## Adding documents

Any PDF, HTML, DOCX, PPTX, or TXT file dropped here and re-indexed will be
retrievable. Re-run ingestion **without** `--wipe` to add to the existing index,
or **with** `--wipe` to rebuild from scratch:

```bash
# Add to existing index
python -m app.ingestion.processor DATA/true_data true

# Rebuild from scratch
python -m app.ingestion.processor DATA/true_data true --wipe
```
