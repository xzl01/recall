import json
import socket
import threading
import tempfile
from pathlib import Path
import time

import pytest

from helpers import FakeEmbedding
from recall import daemon


def request(path, body):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(3)
        s.connect(str(path))
        s.sendall(json.dumps(body).encode() + b"\n")
        return json.loads(s.makefile("rb").readline())


@pytest.fixture
def short_path():
    # macOS Unix sockets have a shorter pathname limit than pytest temp roots.
    with tempfile.TemporaryDirectory(prefix="recall-", dir="/tmp") as temp:
        yield Path(temp)


def test_daemon_survives_bad_requests_and_matches_the_model(short_path, monkeypatch):
    tmp_path = short_path
    model = tmp_path / "model"
    monkeypatch.setattr(FakeEmbedding, "model_dir", model)
    monkeypatch.setattr(daemon, "Embedder", FakeEmbedding)
    monkeypatch.setattr(daemon, "MODEL_DIR", model)
    path = tmp_path / "embed.sock"
    errors = []

    def serve():
        try:
            daemon.run_server(str(path), "CPU", 2, 8)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while not path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert path.exists(), errors
        assert path.stat().st_mode & 0o777 == 0o600
        assert not request(path, [])["ok"]
        assert not request(path, {"cmd": "embed", "texts": [42]})["ok"]
        assert request(path, {"cmd": "ping"})["ok"]
        client = daemon.EmbClient(str(path))
        client.connect()
        assert client.supports("CPU")
        assert client.encode("synthetic fact").shape == (1, 512)
        assert client.encode(["synthetic fact"] * 257).shape == (257, 512)
        monkeypatch.setattr(daemon, "MODEL_DIR", tmp_path / "different-model")
        assert not client.supports("CPU")
        with pytest.raises(RuntimeError, match="already running"):
            daemon.run_server(str(path), "CPU", 2, 8)
    finally:
        if worker.is_alive() and path.exists():
            request(path, {"cmd": "shutdown"})
        worker.join(timeout=3)
    assert not worker.is_alive() and not errors
    assert not path.exists()


def test_daemon_does_not_replace_regular_files(tmp_path):
    path = tmp_path / "embed.sock"
    path.write_text("keep me")
    with pytest.raises(RuntimeError):
        daemon.run_server(str(path), "CPU", 1, 8)
    assert path.read_text() == "keep me"
