---
name: recall
description: Retrieve indexed local Pi history and text/code sources with recall when past decisions, actions or discussions matter. Use source snippets to establish context; exact file symbols and literal errors can also be located with rg.
---

# recall

Use the installed `recall` executable. This is retrieval of original material, not a verified conclusion store. If a separate reviewed memory store exists, consult it first when appropriate.

## Source coverage

Only indexed Pi JSONL sessions and explicitly selected text directories are included. Sharing the executable with another harness does not automatically index that harness's history. Missing snippets do not prove an action never happened. Model installation and indexing are separate from installing this skill.

```bash
recall --chat --json "previous decision about gesture configuration"
recall --code --json --full "identifier and error message"
recall search "query" --corpus sessions -k 10 --json
recall --read STABLE_ID --neighbors 3 --json
recall --semantic --neighbors 0 --json "natural-language description"
recall --status
```

Natural-language descriptions help semantic retrieval; names, identifiers and literal errors help keyword retrieval. Precise function/file searches may be better served by `rg`.

## Evidence and context

- `id` is a stable window key for reading; numeric row IDs can change after rebuilds.
- `score` is cosine similarity, `rank_score` is rank fusion, and neither is a confidence probability. `semantic_rank`, `lexical_rank`, `matched_terms`, `selection_reason` and `fusion_rank` explain candidate selection.
- `parent/offset_start/offset_end` identify the anchor window. `context_records` identifies each added window's record, time, role and position; do not attribute all neighboring content to the anchor's author or time.
- `truncated/text_chars` indicate character clipping. Search adds one neighboring window per side by default, staying within the same source. `--neighbors 0` returns the raw hit. `--full` removes the character budget.
- `--read` returns neighboring windows from the same original record (two per side by default), not an entire session or file.

Read more context or revise the query when evidence is incomplete. Check dates, roles and source versions when facts conflict; current code and reviewed conclusions may supersede old snippets. Do not treat a model-generated summary as original evidence. Avoid reusing a universal similarity cutoff across languages or splitting schemes.

## Operation

Daily indexing/search is local and needs no answering-model credentials. Model setup may download public weights only when invoked. Configuration comes from `RECALL_HOME`, `RECALL_MODEL_DIR`, `RECALL_SESSIONS_DIR`, `RECALL_DEVICE` and `RECALL_SOCKET`, or their CLI overrides.

Capture stderr and respect nonzero exit codes; errors are not JSON results. Preserve the paired database/vector files. Incremental `recall index sessions` ingests new records; `--force` rebuilds the selected corpus. Publishing indexes requires separate judgment about the actual contents; the source repository contains no real histories.
