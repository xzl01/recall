# Architecture

The executable configures paths before importing the engine. Runtime data lives outside the source checkout. `sources.py` reads six agent history formats into one record shape; `engine.py` indexes those records and selected text files; `support.py` provides token windows, lexical terms, locks and publication; `embedding.py` supplies fixed BGE CLS vectors; `daemon.py` optionally reuses a resident service running whichever engine the platform selected — MLX on Apple Silicon, OpenVINO elsewhere, or GGUF when explicitly requested. `identity.py` derives the digest binding an index to the weights that produced it.

## Retrieval

1. The query is bounded for the BGE tokenizer; lexical terms retain up to 128 distinct terms, using both ends when needed.
2. Normalized query and stored vectors produce cosine ranks. SQLite FTS5/BM25 supplies lexical candidates from the same corpus filter.
3. Reciprocal rank fusion uses semantic weight 0.6, lexical weight 0.4, and constant 60. Candidate protection reserves up to `max(1, k//3)` slots for lexical hits matching at least two query terms (one for a one-term query), plus a semantic candidate when `k > 1`. Remaining slots follow fusion order with no duplicate anchors.
4. The chosen anchor is extended by neighboring windows, preserving message boundaries and removing overlapping text. Character clipping centers on the anchor.

Returned anchors are ordered by fusion score, but protected candidates may have a lower original fusion rank. `selection_reason` and `fusion_rank` make this explicit. No universal similarity filter is applied by the CLI; callers should inspect sources, dates, matching terms and truncation.

## Storage

`index/meta.db` stores source, role/time, original parent identity, window offsets and stable keys. `index/vectors.npy` stores contiguous, normalized float32 rows. Numeric chunk ID equals vector row index + 1. The FTS table has one row per window.

A writer holds an exclusive process lock, constructs a staging snapshot, validates row IDs, dimensions, FTS row count and finite normalized vectors, then publishes the paired files with a recovery journal. Readers hold a shared lock. Interrupted publication restores both files from the paired rollback snapshot on the next operation.

Session sources use relative paths to avoid same-name collisions within the selected session root, and each agent gets its own corpus (`sessions:pi`, `sessions:dsh`, …) so one harness cannot shadow another. Directory corpus IDs include the resolved root's path hash; `dir:project` remains a usable prefix filter. Rebuilds replace only that exact selected corpus. Indexed material is retained until a corresponding forced rebuild; this is not a live filesystem search.

## Embedding service

Unix-socket NDJSON requests are local, serialized, size bounded, and served through a socket with mode 0600. A lock prevents duplicate daemons; regular files are never replaced as sockets. Clients check device and model directory before reusing a service. The client imports no inference runtime — numpy and tokenizers only — so a base-only install can use a daemon running in another environment; direct in-process inference is the fallback when no service is available. The model is fixed to 512-dimensional BGE-small-zh-v1.5 and CLS pooling, not a general embedding-provider plugin system.
