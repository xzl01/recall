"""Explicit Linux real-model smoke test; never included in default pytest runs."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--device", default="CPU")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export local safetensors weights instead of importing IR",
    )
    a = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="recall-integration-") as temp:
        root = Path(temp)
        home = root / "state"
        sessions = root / "sessions"
        sessions.mkdir()
        fixtures = [
            ("user", "ZephyrChronicle uses Novara9 at address 0x12."),
            ("assistant", "Novara9 timeout is 250 ms. Check the device log before retrying."),
        ]
        path = sessions / "demo.jsonl"
        path.write_text(
            "".join(
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": f"2026-01-01T10:0{i}:00",
                        "message": {"role": role, "content": [{"type": "text", "text": text}]},
                    }
                )
                + "\n"
                for i, (role, text) in enumerate(fixtures)
            )
        )
        base = [
            sys.executable,
            "-m",
            "recall",
            "--home",
            str(home),
            "--sessions-dir",
            str(sessions),
            "--device",
            a.device,
        ]

        def run(*args):
            p = subprocess.run([*base, *args], capture_output=True, text=True, timeout=180)
            assert p.returncode == 0, (args, p.stderr)
            return p.stdout

        run("setup", "--source" if a.export else "--from", str(a.model_dir))
        run("index", "sessions")
        status = run("stats")
        hits = json.loads(run("--chat", "--json", "--full", "Novara9 ZephyrChronicle"))
        assert hits and any("Novara9" in h["text"] for h in hits)
        read = json.loads(run("--read", hits[0]["id"], "--json", "--full"))
        assert hits[0]["id"] in read["window_ids"]
        raw = json.loads(run("search", "Novara9", "--json", "--neighbors", "0"))
        assert all(len(h["context_window_ids"]) == 1 for h in raw)
        demo = root / "project"
        demo.mkdir()
        (demo / "README.md").write_text(
            "The synthetic Amber protocol uses the 250 ms timeout. " * 5
        )
        run("index", "dir", str(demo), "--ext", ".md")
        code = json.loads(run("--code", "--json", "Amber protocol"))
        assert code and all(h["corpus"].startswith("dir:") for h in code)
        with (root / "daemon.log").open("w") as log:
            daemon = subprocess.Popen([*base, "daemon", "--idle-exit", "0"], stdout=log, stderr=log)
            try:
                sock = home / "embed.sock"
                deadline = time.monotonic() + 30
                while not sock.exists() and daemon.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert sock.exists(), (root / "daemon.log").read_text()
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(sock))
                    client.sendall(b'{"cmd":"whoami"}\n')
                    info = json.loads(client.makefile("rb").readline())
                assert info["device"] == a.device and info["model_dir"] == str(
                    (home / "models/bge-small-zh-v1.5").resolve()
                )
                env = {**os.environ, "RECALL_HOME": str(home), "RECALL_DEVICE": a.device}
                probe = (
                    "import sys;from recall import daemon;v=daemon.resolve_embedder('"
                    + a.device
                    + "',1).encode_query('Novara9');assert v.shape==(1,512);assert 'openvino' not in sys.modules"
                )
                p = subprocess.run(
                    [sys.executable, "-c", probe],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                assert p.returncode == 0, p.stderr
                resident = json.loads(run("--chat", "--json", "Novara9"))
                assert resident
            finally:
                daemon.terminate()
                daemon.wait(timeout=20)
        print(
            json.dumps(
                dict(
                    device=a.device,
                    model_export=a.export,
                    synthetic_sessions=True,
                    setup_index_search_read=True,
                    directory_filter=True,
                    daemon_and_lazy_client=True,
                    private_history_used=False,
                )
            )
        )


if __name__ == "__main__":
    main()
