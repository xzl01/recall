"""Paths are independent of a checkout or a particular machine."""

import os
from pathlib import Path

DATA_DIR = (
    Path(os.environ.get("RECALL_HOME", str(Path.home() / ".local/share/recall")))
    .expanduser()
    .resolve()
)
MODEL_DIR = (
    Path(os.environ.get("RECALL_MODEL_DIR", str(DATA_DIR / "models/bge-small-zh-v1.5")))
    .expanduser()
    .resolve()
)
# MLX 后端用的模型目录。和 OpenVINO 的 IR 目录分开，权重是同一份 BGE。
MLX_MODEL_DIR = (
    Path(os.environ.get("RECALL_MLX_MODEL_DIR", str(DATA_DIR / "models/bge-small-zh-v1.5-mlx")))
    .expanduser()
    .resolve()
)
# GGUF 后端的模型目录：放一个 .gguf 加一份 tokenizer.json（分块要用）。
# GGUF 是另一份产物（通常还是量化过的），所以它有独立的索引身份。
GGUF_MODEL_DIR = (
    Path(os.environ.get("RECALL_GGUF_MODEL_DIR", str(DATA_DIR / "models/bge-small-zh-v1.5-gguf")))
    .expanduser()
    .resolve()
)
SESSIONS_DIR = (
    Path(os.environ.get("RECALL_SESSIONS_DIR", str(Path.home() / ".pi/agent/sessions")))
    .expanduser()
    .resolve()
)
# 其它 agent 的历史位置。各自的落盘格式不同，读取器在 sources.py。
DSH_SESSIONS_DIR = (
    Path(os.environ.get("RECALL_DSH_SESSIONS_DIR", str(Path.home() / ".dsh/sessions")))
    .expanduser()
    .resolve()
)
CODEX_SESSIONS_DIR = (
    Path(os.environ.get("RECALL_CODEX_SESSIONS_DIR", str(Path.home() / ".codex/sessions")))
    .expanduser()
    .resolve()
)
# opencode 已经迁到 SQLite：旧的 storage/ 目录是空的，历史都在这个库里。
OPENCODE_DB = (
    Path(
        os.environ.get("RECALL_OPENCODE_DB", str(Path.home() / ".local/share/opencode/opencode.db"))
    )
    .expanduser()
    .resolve()
)
SOCKET_PATH = (
    Path(os.environ.get("RECALL_SOCKET", str(DATA_DIR / "embed.sock"))).expanduser().resolve()
)
DEFAULT_DEVICE = os.environ.get("RECALL_DEVICE", "CPU")
# auto（默认：优先 OpenVINO，缺失则回退 MLX）/ openvino / mlx / gguf
BACKEND = os.environ.get("RECALL_BACKEND", "auto")
