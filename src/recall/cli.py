"""Installed entry point, including the original agent-friendly CLI flags."""

import argparse
import os
import sys

from . import __version__

HELP = """recall — local agent history and code retrieval

Usage:
  recall setup [--from EXISTING_MODEL_DIR] [--mlx|--openvino]
  recall index sessions [--force]
  recall index dir PATH [--ext .py,.md]
  recall search "query" [--json] [-k 6] [--neighbors 1]
  recall "query" [--chat|--code] [--json] [--full]
  recall read ID [--json] [--neighbors 2]
  recall daemon [--idle-exit 0]
  recall stats
  recall migrate

Global options (also accepted after a command):
  --home PATH           State directory (RECALL_HOME)
  --model-dir PATH      BGE tokenizer + ov/ model directory (RECALL_MODEL_DIR)
  --sessions-dir PATH   Pi JSONL source (RECALL_SESSIONS_DIR)
  --device CPU|GPU|NPU  OpenVINO device (RECALL_DEVICE, default CPU)
  --batch N             Embedding batch size (default 8)
  --version

Backends (one project, three engines; the first two are picked by platform):
  Apple Silicon (darwin/arm64)   MLX       pip install '.[mlx]'
  Intel / Linux / other          OpenVINO  pip install '.[openvino]'
  any platform (explicit only)   GGUF      pip install '.[gguf]'
  RECALL_BACKEND        auto (default) | openvino | mlx | gguf. `auto` takes the
                        platform default and falls back if its deps are absent.
                        gguf is never picked automatically: it is usually
                        quantised, so its vectors are a different space.
  RECALL_MLX_MODEL_DIR  MLX safetensors directory
                        (default $RECALL_HOME/models/bge-small-zh-v1.5-mlx).
  RECALL_GGUF_MODEL_DIR GGUF directory: one .gguf plus tokenizer.json
                        (default $RECALL_HOME/models/bge-small-zh-v1.5-gguf).
  `recall setup` installs the model for the platform's default backend;
  use --mlx / --openvino / --gguf to force one.

Original flags remain available: --status, --index, --read ID,
--semantic, --neighbors, --chars, --full, --chat and --code.
Use `recall search --help` for retrieval options.
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--version"]:
        print(__version__)
        return
    if not argv or argv in (["--help"], ["-h"]):
        print(HELP)
        return
    pre = argparse.ArgumentParser(add_help=False)
    for name in ("home", "model-dir", "sessions-dir", "device"):
        pre.add_argument("--" + name)
    pre.add_argument("--batch", type=int, default=8)
    config, rest = pre.parse_known_args(argv)
    for attr, env in (
        ("home", "RECALL_HOME"),
        ("model_dir", "RECALL_MODEL_DIR"),
        ("sessions_dir", "RECALL_SESSIONS_DIR"),
        ("device", "RECALL_DEVICE"),
    ):
        value = getattr(config, attr)
        if value is not None:
            os.environ[env] = value
    os.umask(0o077)
    if not rest:
        pre.error("a command or query is required")
    command = rest[0]
    try:
        if command == "setup":
            from .setup import main as setup_main

            return setup_main(rest[1:])
        if command == "daemon":
            from .daemon import main as daemon_main

            return daemon_main(["--batch", str(config.batch), *rest[1:]])
        from . import engine

        if command in ("--status", "-s"):
            rest = ["stats", *rest[1:]]
        elif command in ("--index", "-i"):
            rest = ["index", "sessions", *rest[1:]]
        elif command == "migrate":
            rest = ["rechunk", *rest[1:]]
        elif command not in ("search", "index", "stats", "read", "rechunk"):
            return legacy_search(engine, rest)
        if rest[0] in ("search", "read"):
            # --full is a convenient alias for the engine's character budget.
            if "--full" in rest:
                rest.remove("--full")
                rest.extend(["--chars", "0"])
        return engine.main(["--batch", str(config.batch), *rest])
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"recall: {error}", file=sys.stderr)
        raise SystemExit(1) from None


def legacy_search(engine, args):
    parser = argparse.ArgumentParser(prog="recall", description=HELP)
    corpus = parser.add_mutually_exclusive_group()
    corpus.add_argument("--chat", action="store_true")
    corpus.add_argument("--code", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--full", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--semantic", action="store_true")
    mode.add_argument("--hybrid", action="store_true")
    parser.add_argument("--read", dest="id")
    parser.add_argument("--neighbors", type=int)
    parser.add_argument("--chars", type=int)
    parser.add_argument("-k", type=int, default=6)
    parser.add_argument("query", nargs="*")
    a = parser.parse_args(args)
    if a.id:
        if a.query:
            parser.error("--read does not accept a query")
        a.neighbors = 2 if a.neighbors is None else a.neighbors
        a.chars = 0 if a.full else (12000 if a.chars is None else a.chars)
        return engine.cmd_read(a)
    if not a.query:
        parser.error("a query is required")
    a.query = " ".join(a.query)
    a.corpus = "sessions" if a.chat else ("dir:" if a.code else None)
    a.neighbors = 1 if a.neighbors is None else a.neighbors
    a.chars = 0 if a.full else (4000 if a.chars is None else a.chars)
    a.mode = "semantic" if a.semantic else "hybrid"
    a.device = engine.DEFAULT_DEVICE
    a.width = 220
    return engine.cmd_search(a)
