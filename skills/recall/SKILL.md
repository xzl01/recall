---
name: recall
description: Retrieve indexed local agent history (Pi, DeepSeek Harness, Codex, opencode, CodeBuddy, WorkBuddy, Claude Code) and selected text/code directories with recall when past decisions, actions or discussions matter. Snippets are original material, not verified conclusions; use rg for exact symbols.
---

# recall

Use the installed `recall` executable to retrieve original material: a decision already
made, an error already diagnosed, a configuration already settled. It is a retrieval
index, not a verified conclusion store. If a separate reviewed memory store exists,
consult it first when appropriate.

## What is searchable

Indexing is explicit. Nothing is indexed merely because the executable is shared with a
harness.

- `sessions:*` — one corpus per agent: `pi`, `dsh`, `codex`, `opencode`, `codebuddy`,
  `workbuddy`, `claude`. Only conversation text (user/assistant) is indexed, never tool calls or
  tool outputs, and not every record survives the size caps.
- `dir:*` — text directories added with `recall index dir PATH`.

A missing snippet does not prove something never happened: that agent may not be indexed,
or the record may have been skipped for size. Note that the CodeBuddy **IDE** keeps
conversations server-side, so `sessions:codebuddy` stays empty on an IDE-only install.

```bash
recall "natural-language description" --chat --json        # all agent histories
recall "identifier or error text" --code --json --full     # indexed text directories
recall search "query" --corpus sessions:codex -k 10 --json
recall search "query" --corpus sessions -k 10 --json       # prefix: every agent
recall read STABLE_ID --neighbors 3 --json
recall search "query" --mode semantic --neighbors 0 --json
recall stats
```

Natural-language descriptions suit the semantic half of hybrid retrieval; names,
identifiers and literal error strings suit the keyword half. For exact symbols or paths,
`rg` is usually the better tool.

## Evidence

- `id` is a stable window key for reading; numeric `chunk_id` changes on rebuild.
- `score` is cosine similarity and `rank_score` is rank fusion. Neither is a confidence
  probability. `semantic_rank`, `lexical_rank`, `matched_terms`, `selection_reason` and
  `fusion_rank` explain why a candidate was selected.
- **`search` results** carry `parent` and `offset_start`/`offset_end`. Two hits sharing a
  `parent` are two windows of *one* message — windows overlap by 64 tokens, so near-identical
  text is one record, not independent corroboration. `context_records` reports each
  neighbouring window's record, time, role and position; do not attribute all neighbouring
  text to the anchor's author or timestamp.
- `truncated` / `text_chars` report clipping. Search adds one neighbouring window per side
  within the same record; `--neighbors 0` returns the raw hit; `--full` lifts the character
  budget.
- **`read` returns a different, smaller shape**: `text` may already include neighbouring
  windows merged in, and `window_ids` lists which windows contributed. It has no
  `context_records` and no `parent`. A short message is a single window, so `--neighbors`
  has nothing to add there and `window_ids` has length 1.

Read more context or rephrase when evidence is incomplete. Check dates, roles and source
versions when facts conflict: current code and reviewed conclusions may supersede old
snippets. Do not present a model-generated summary as original evidence. Avoid carrying
one similarity cutoff across languages or chunking schemes.

## Operation

Daily indexing and search are local and need no answering-model credentials. Embedding
weights are a separate install — `recall setup`, or explicitly `--mlx` / `--openvino` /
`--gguf`; `RECALL_BACKEND` pins one engine. Weights download only when setup is invoked.

`recall index sessions` is incremental and ingests new records; `recall index <agent>`
does a single source; `--force` rebuilds the selected corpus. Publishing an index is a
separate judgement about its contents — this repository ships no real histories.

Keep `meta.db` and `vectors.npy` together. Errors go to stderr with a nonzero exit code
and are not JSON results.

Configuration comes from `RECALL_HOME`, `RECALL_MODEL_DIR` (or `RECALL_MLX_MODEL_DIR` /
`RECALL_GGUF_MODEL_DIR`), `RECALL_BACKEND`, `RECALL_DEVICE`, `RECALL_SOCKET`, and the
per-agent source overrides `RECALL_SESSIONS_DIR`, `RECALL_DSH_SESSIONS_DIR`,
`RECALL_CODEX_SESSIONS_DIR`, `RECALL_OPENCODE_DB`, `RECALL_CODEBUDDY_SESSIONS_DIR` and
`RECALL_WORKBUDDY_SESSIONS_DIR`. CLI flags override them.

If `recall` is not found at all, that is a setup gap rather than an empty history: the
executable may live in a checkout's virtualenv instead of on `PATH`. Report it as such —
do not conclude that nothing was recorded. A forwarding script for `PATH` comes from
`skills/install.sh --shim` in the same repository.
