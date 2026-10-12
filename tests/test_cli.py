import os
import subprocess
import sys


def call(*args):
    return subprocess.run([sys.executable, "-m", "recall", *args], capture_output=True, text=True)


def test_help_and_version():
    assert call("--version").stdout.strip() == "0.1.0"
    assert "recall index sessions" in call("--help").stdout
    assert "--neighbors" in call("search", "--help").stdout


def test_configuration_does_not_use_the_live_index(tmp_path):
    p = call("--home", str(tmp_path / "empty"), "--status")
    assert p.returncode != 0 and not p.stdout
    assert not (tmp_path / "empty/index/meta.db").exists()


def test_json_errors_stay_on_stderr(tmp_path):
    p = call("--home", str(tmp_path), "--json", "--neighbors", "9", "test")
    assert p.returncode != 0 and not p.stdout and p.stderr


def test_code_flag_selects_all_directory_corpora(monkeypatch):
    from recall import cli, engine

    captured = []
    monkeypatch.setattr(engine, "cmd_search", lambda args: captured.append(args))
    cli.legacy_search(engine, ["--code", "--json", "identifier"])
    assert captured[0].corpus == "dir:"
    assert captured[0].neighbors == 1


def test_read_compatibility_defaults(monkeypatch):
    from recall import cli, engine

    captured = []
    monkeypatch.setattr(engine, "cmd_read", lambda args: captured.append(args))
    cli.legacy_search(engine, ["--read", "stable-id", "--json", "--full"])
    assert captured[0].neighbors == 2 and captured[0].chars == 0


def test_nested_session_sources_remain_distinct(tmp_path):
    from recall import engine as e
    from helpers import install_fake
    from types import SimpleNamespace
    import json

    restore = install_fake()
    try:
        e.INDEX = tmp_path / "index"
        e.DB = e.INDEX / "meta.db"
        e.VEC = e.INDEX / "vectors.npy"
        e.SESSIONS = tmp_path / "sessions"
        # `index sessions` 会遍历所有 agent；其余三个指向不存在的位置，
        # 否则测试会去索引真实的 Codex/opencode 历史。
        e.DSH_SESSIONS = tmp_path / "dsh"
        e.CODEX_SESSIONS = tmp_path / "codex"
        e.OPENCODE_PATH = tmp_path / "opencode.db"
        for name in ["a", "b"]:
            p = e.SESSIONS / name / "same.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text(
                json.dumps(
                    {
                        "type": "message",
                        "timestamp": "2026-01-01T00:00:00",
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": "An identical remembered fact."}],
                        },
                    }
                )
                + "\n"
            )
        e.cmd_index(SimpleNamespace(what="sessions", force=False, device="CPU", batch=8))
        con = e.readonly()
        assert {r[0] for r in con.execute("SELECT source FROM chunks")} == {
            "a/same.jsonl",
            "b/same.jsonl",
        }
        con.close()
    finally:
        restore()


def test_same_named_directories_do_not_replace_each_other(tmp_path):
    from recall import engine as e
    from helpers import install_fake
    from types import SimpleNamespace

    restore = install_fake()
    try:
        e.INDEX = tmp_path / "index"
        e.DB = e.INDEX / "meta.db"
        e.VEC = e.INDEX / "vectors.npy"
        roots = [tmp_path / "one/project", tmp_path / "two/project"]
        for i, r in enumerate(roots):
            (r / "docs").mkdir(parents=True)
            (r / "docs/notes.md").write_text(
                f"An independent documented decision for root {i}. " * 3
            )
            e.cmd_index(
                SimpleNamespace(
                    what="dir",
                    path=str(r),
                    ext=".md",
                    include_vendor=False,
                    exclude="",
                    force=False,
                    device="CPU",
                    batch=8,
                )
            )
        con = e.readonly()
        before = set(r[0] for r in con.execute("SELECT corpus FROM chunks"))
        con.close()
        assert len(before) == 2
        e.cmd_index(
            SimpleNamespace(
                what="dir",
                path=str(roots[0]),
                ext=".md",
                include_vendor=False,
                exclude="",
                force=True,
                device="CPU",
                batch=8,
            )
        )
        con = e.readonly()
        assert set(r[0] for r in con.execute("SELECT corpus FROM chunks")) == before
        con.close()
    finally:
        restore()
