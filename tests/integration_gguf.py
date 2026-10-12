"""Explicit GGUF (llama.cpp) real-model smoke test; never included in default pytest runs.

对齐 `integration_mlx.py` / `integration_openvino.py`，但走 GGUF 后端。

    python tests/integration_gguf.py /path/to/dir-with-gguf-and-tokenizer

模型目录里需要一个 .gguf 和一份 tokenizer.json，可以用 `recall setup --gguf` 生成。
没装 llama-cpp-python 时以退出码 0 跳过（打印 skipped），这样在没有 C++ 工具链的
环境里也不会把 CI 弄红。
"""

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
    parser.add_argument("--device", default="CPU", help="仅作记录，llama.cpp 自己选后端")
    a = parser.parse_args()

    import importlib.util

    if importlib.util.find_spec("llama_cpp") is None:
        print(json.dumps({"backend": "gguf", "skipped": "llama-cpp-python not installed"}))
        return

    with tempfile.TemporaryDirectory(prefix="recall-gguf-") as temp:
        root = Path(temp)
        home = root / "state"
        sessions = root / "sessions"
        sessions.mkdir()
        fixtures = [
            ("user", "ZephyrChronicle uses Novara9 at address 0x12."),
            ("assistant", "Novara9 timeout is 250 ms. Check the device log before retrying."),
        ]
        (sessions / "demo.jsonl").write_text(
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
        env = {
            **os.environ,
            "RECALL_HOME": str(home),
            "RECALL_BACKEND": "gguf",
            "RECALL_GGUF_MODEL_DIR": str(home / "models/bge-small-zh-v1.5-gguf"),
            "RECALL_DEVICE": a.device,
            # 同上：把其余 agent 指到不存在的位置，别去索引真实历史。
            "RECALL_DSH_SESSIONS_DIR": str(root / "dsh"),
            "RECALL_CODEX_SESSIONS_DIR": str(root / "codex"),
            "RECALL_OPENCODE_DB": str(root / "opencode.db"),
            "RECALL_CODEBUDDY_SESSIONS_DIR": str(root / "codebuddy"),
            "RECALL_WORKBUDDY_SESSIONS_DIR": str(root / "workbuddy"),
        }
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
            p = subprocess.run([*base, *args], capture_output=True, text=True, timeout=300, env=env)
            assert p.returncode == 0, (args, p.stderr)
            return p.stdout

        run("setup", "--gguf", "--from", str(a.model_dir))
        run("index", "sessions")
        run("stats")
        hits = json.loads(run("--chat", "--json", "--full", "Novara9 ZephyrChronicle"))
        assert hits and any("Novara9" in h["text"] for h in hits), hits
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

        paraphrase = json.loads(run("--code", "--json", "--full", "how long before giving up"))
        assert paraphrase, "semantic retrieval returned nothing"

        # 身份校验：同一个索引换成另一份模型必须被拒绝，而不是静默给错结果。
        weights = sorted(a.model_dir.glob("*.gguf"))[0]
        other = root / "other-model"
        other.mkdir()
        for name in (weights.name, "tokenizer.json"):
            (other / name).write_bytes((a.model_dir / name).read_bytes())
        tampered = other / weights.name
        tampered.write_bytes(tampered.read_bytes() + b"\0")  # 改动权重产物 -> 身份必变
        swapped = {**env, "RECALL_GGUF_MODEL_DIR": str(other)}
        p = subprocess.run(
            [*base, "search", "Novara9", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            env=swapped,
        )
        assert p.returncode != 0 and "另一份嵌入模型" in p.stderr, (p.returncode, p.stderr)

        with (root / "daemon.log").open("w") as log:
            daemon = subprocess.Popen(
                [*base, "daemon", "--idle-exit", "0"], stdout=log, stderr=log, env=env
            )
            try:
                sock = home / "embed.sock"
                deadline = time.monotonic() + 90
                while not sock.exists() and daemon.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert sock.exists(), (root / "daemon.log").read_text()
                with socket.socket(socket.AF_UNIX) as client:
                    client.connect(str(sock))
                    client.sendall(b'{"cmd":"whoami"}\n')
                    info = json.loads(client.makefile("rb").readline())
                assert info["backend"] == "gguf", info
                assert info["dim"] == 512, info
                resident = json.loads(run("--chat", "--json", "Novara9"))
                assert resident
            finally:
                daemon.terminate()
                daemon.wait(timeout=20)

        print(
            json.dumps(
                dict(
                    backend="gguf",
                    device=a.device,
                    model_identity_enforced=True,
                    setup_index_search_read=True,
                    directory_filter=True,
                    daemon_and_lazy_client=True,
                    private_history_used=False,
                )
            )
        )


if __name__ == "__main__":
    main()
