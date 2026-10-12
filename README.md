# recall

Local retrieval for an agent's history and source code. Recall indexes the session history of six agents — Pi, DeepSeek Harness, Codex, opencode, CodeBuddy and WorkBuddy — plus selected text directories, combines BGE embeddings with SQLite FTS5, and returns source snippets with stable IDs and neighboring context.

**Retrieval runs locally.** No LLM provider or API key is needed. Model setup downloads public weights when you request it. Your index stays outside the repository.

[中文说明](README.zh-CN.md) · [Agent skill](skills/recall/SKILL.md) · [Architecture](docs/architecture.md) · [Evaluation](docs/evaluation.md) · [Backends](docs/backends.md)

## Install

Python 3.10+ on Linux/POSIX. Windows is currently unsupported because indexing uses POSIX locks and the daemon uses Unix sockets.

### Backends: one project, three embedding engines

The retrieval pipeline is engine-agnostic; the embedding layer has interchangeable implementations. The first two are picked by platform, GGUF is opt-in:

| Platform | Default backend | Install |
| --- | --- | --- |
| Apple Silicon (darwin / arm64) | **MLX** (Metal) | `pip install '.[mlx]'` |
| Linux, Intel macOS, other | **OpenVINO** | `pip install '.[openvino]'` |
| any platform, explicit only | **GGUF** (llama.cpp) | `pip install '.[gguf]'` |

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[mlx]'      # or '.[openvino]' / '.[gguf]'
.venv/bin/recall setup                  # installs the model for your platform's backend
```

`recall setup` follows the platform default; pass `--mlx`, `--openvino` or `--gguf` to force one. `RECALL_BACKEND=auto` (the default) takes the platform default and falls back to the other fp32 engine if its dependency is missing. An explicit `RECALL_BACKEND` never falls back — it errors instead of silently switching.

**GGUF is never selected automatically.** It is usually a *quantised* artifact, so its vectors live in a different space from the fp32 weights — recall therefore binds every index to the model identity it was built from and refuses to search or append with a different one. Switching engines (or quantisation levels) means rebuilding the index. `llama-cpp-python` ships no wheels, so `.[gguf]` compiles on install and needs a C++ toolchain and cmake.

Both engines load the **same** `BAAI/bge-small-zh-v1.5` weights and keep the same semantics: CLS pooling, L2 normalisation, 512 dimensions, and the same Chinese query prefix, so an index built by one engine can be read by the other. The index now records the **model identity** it was built from (the source `model.safetensors` digest, which both engines share) and refuses to search or append with a different one — swapping weights for another 512-dimension BERT used to silently return wrong results.

Measured on one Apple Silicon machine against a `transformers` reference on the same `model.safetensors`: MLX differs by 8.0e-07, while OpenVINO differs by 1.4e-03 (its own graph-optimisation numerics, long-standing upstream behaviour — not introduced here). The practical effect is nil: over a 6-document corpus and 5 mixed-language queries the two backends produced **identical top-k rankings**, with scores differing in the fourth decimal. See [docs/backends.md](docs/backends.md) for the numbers and caveats.

```bash
git clone https://github.com/xzl01/recall.git
cd recall
python3 -m venv .venv
. .venv/bin/activate
pip install '.[setup]'
recall setup
```

Setup exports the pinned [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5) revision to OpenVINO. Its 512-dimensional CLS embeddings and Chinese query prefix are fixed; arbitrary replacement models are not supported. The model is optimized for Chinese, so English results should be checked against the source.

For an existing export with `tokenizer.json` and `ov/openvino_model.{xml,bin}`, install `.[openvino]` and use:

```bash
recall setup --from /path/to/bge-small-zh-v1.5
```

Model weights are downloaded/imported into your state directory, not bundled with the code. Export follows the [Optimum Intel API](https://huggingface.co/docs/optimum-intel/openvino/reference). See [third-party notices](THIRD_PARTY_NOTICES.md).

## Index and retrieve

```bash
# Explicitly index agent history, or a text directory.
recall index sessions                  # every agent it can find
recall index codex                     # or just one: pi | dsh | codex | opencode
recall index dir ./my-project --ext .py,.md,.ts

recall "How did we configure the touch gestures?" --chat --json
recall "timeout handler" --code --json --full
recall search "exact_identifier error message" --json -k 6
recall search "query" --mode semantic --neighbors 0 --json
recall read STABLE_ID --neighbors 2 --json
recall stats
```

`--code` searches all indexed `dir:` corpora, `--chat` searches every agent's history. The explicit search command takes `--corpus sessions:dsh` or `--corpus dir:my-project`.

### Agent history

`recall index sessions` reads the history of six agents, each into its own corpus (`sessions:pi`, `sessions:dsh`, …); path overrides are environment variables:

| agent | location | format | override |
| --- | --- | --- | --- |
| Pi | `~/.pi/agent/sessions` | plain JSONL | `RECALL_SESSIONS_DIR` |
| DeepSeek Harness | `~/.dsh/sessions` | zstd JSONL event stream | `RECALL_DSH_SESSIONS_DIR` |
| Codex CLI | `~/.codex/sessions` | plain JSONL | `RECALL_CODEX_SESSIONS_DIR` |
| opencode | `~/.local/share/opencode/opencode.db` | SQLite | `RECALL_OPENCODE_DB` |
| CodeBuddy | `~/.codebuddy/projects` | plain JSONL | `RECALL_CODEBUDDY_SESSIONS_DIR` |
| WorkBuddy | `~/.workbuddy/projects` | plain JSONL | `RECALL_WORKBUDDY_SESSIONS_DIR` |

A source whose directory does not exist is reported and skipped, so a machine that never ran an agent needs no setup. CodeBuddy and WorkBuddy share one reader — Tencent's two products write the same local session format — but note that the **CodeBuddy IDE keeps conversations server-side** and leaves nothing indexable on disk; `~/.codebuddy/projects` is what the CodeBuddy CLI writes.

Only **conversation text** (user/assistant messages) is indexed — tool calls and tool outputs carry little retrieval value relative to their size, so they are skipped. Records longer than 200,000 characters are skipped, and JSONL lines larger than 1 MiB are dropped before parsing; both counts are reported. That keeps reading bounded: it used to peak at 2 GB (a single Codex session is 986 MB), and now peaks near 100 MB — a full run of 22k records → 63k windows peaks around 670 MB. See [docs/agent-history.md](docs/agent-history.md) for the per-agent event mapping and the measurements behind those limits.

- Long records use 448-token windows with 64-token overlap.
- Hybrid retrieval reserves candidates for strong keyword matches, retaining hits that pure rank fusion could bury.
- Queries keep up to 128 distinct lexical terms, retaining both ends for longer inputs.
- Search adds one neighboring window on each side by default. Session records retain visible time/role boundaries; file context stays within its original record. Use `--neighbors 0` for raw hits.
- JSON includes `id`, `score`, `semantic_rank`, `lexical_rank`, `selection_reason`, `fusion_rank`, `context_records`, `text_chars` and `truncated`. Similarity/rank scores are evidence, not confidence probabilities. The default 4000-character budget preserves the anchor; `--full` removes that budget.

Writes use a staged index, process locks and a paired database/vector recovery journal. Keep `meta.db` and `vectors.npy` together. Indexing is incremental; `--force` rebuilds only the selected corpus. Directory IDs include a path hash so roots with the same basename cannot overwrite each other. Deleted source records are retained until that corpus is rebuilt. Exclusions are configurable through the index command. Text files larger than 1.2 MB and binary files are skipped.

## Paths and resident service

| Environment variable | Default |
| --- | --- |
| `RECALL_HOME` | `~/.local/share/recall` |
| `RECALL_MODEL_DIR` | `$RECALL_HOME/models/bge-small-zh-v1.5` |
| `RECALL_SESSIONS_DIR` | `~/.pi/agent/sessions` |
| `RECALL_DEVICE` | `CPU` |
| `RECALL_SOCKET` | `$RECALL_HOME/embed.sock` |

CLI overrides: `--home`, `--model-dir`, `--sessions-dir`, `--device`, `--batch`. A daemon can avoid per-query model compilation:

```bash
recall daemon --device CPU --idle-exit 0
# In another terminal, using the same configuration:
recall "query" --device CPU --json
```

The socket accepts only local clients and has mode 0600. The client verifies the model directory and device before reuse; otherwise it falls back to direct inference. [Example systemd user units](examples/) assume `recall` is installed at `~/.local/bin/recall`; adjust `ExecStart` for your environment. [Migration notes](docs/migration.md) describe existing npulab indexes.

## Agent skill

Copy `skills/recall/` to the skill directory of your harness, such as `~/.agents/skills/recall/`. `skills/install.sh` does that for the common locations (`~/.agents`, `~/.codex`, `~/.claude`, `~/.workbuddy`, `~/.codebuddy`) — it skips anything that already exists unless you pass `--force`, and `--list` shows the targets without writing. It copies rather than symlinks, because a checkout on a removable volume would leave dangling links; re-run with `--force` after the skill changes. The skill guides retrieval and evidence checking; it does not install the executable or silently ingest history.

## Develop

```bash
pip install '.[dev]'
pytest
ruff check .
python -m build
```

Unit tests use synthetic text, deterministic fake embeddings, and a locally trained tokenizer. CI needs neither model downloads nor private sessions. Real OpenVINO integration is a separate Linux check; see [validation notes](docs/validation.md). Small-sample MemoryAgentBench results are documented with strict versus format-normalized scoring, and do not claim leaderboard performance.

## License

MIT. Model and dependency licenses remain their respective authors' licenses.
