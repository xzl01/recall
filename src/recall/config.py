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
SESSIONS_DIR = (
    Path(os.environ.get("RECALL_SESSIONS_DIR", str(Path.home() / ".pi/agent/sessions")))
    .expanduser()
    .resolve()
)
SOCKET_PATH = (
    Path(os.environ.get("RECALL_SOCKET", str(DATA_DIR / "embed.sock"))).expanduser().resolve()
)
DEFAULT_DEVICE = os.environ.get("RECALL_DEVICE", "CPU")
