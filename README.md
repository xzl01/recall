# recall

Local retrieval for an agent's history and source code. Recall indexes Pi JSONL sessions and text directories, combines BGE embeddings with SQLite FTS5, and returns source snippets with stable IDs and neighboring context.

**Retrieval runs locally.** No LLM provider or API key is needed. Model setup downloads public weights when you request it. Your index stays outside the repository.

[中文说明](README.zh-CN.md) · [Agent skill](skills/recall/SKILL.md) · [Architecture](docs/architecture.md) · [Evaluation](docs/evaluation.md)

## Install

Python 3.10+ on Linux/POSIX. Real CPU/GPU inference is tested on Linux x86_64 with OpenVINO; GPU/NPU availability depends on hardware and drivers. Portable unit tests also run on macOS, which is not a claim of verified model inference there. Windows is currently unsupported because indexing uses POSIX locks and the daemon uses Unix sockets.

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
# Explicitly index Pi history, or a text directory.
recall index sessions
recall index dir ./my-project --ext .py,.md,.ts

recall "How did we configure the touch gestures?" --chat --json
recall "timeout handler" --code --json --full
recall search "exact_identifier error message" --json -k 6
recall search "query" --mode semantic --neighbors 0 --json
recall read STABLE_ID --neighbors 2 --json
recall stats
```

`--code` searches all indexed `dir:` corpora. The explicit search command supports `--corpus sessions` or `--corpus dir:my-project`. Only indexed Pi history and selected text files are searchable; other agent histories are not automatically ingested.

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

Copy `skills/recall/` to the skill directory of your harness, such as `~/.agents/skills/recall/`. The skill guides retrieval and evidence checking; it does not install the executable or silently ingest history.

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
