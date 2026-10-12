"""四个 agent 的会话读取器：格式解析、过滤规则与边界。"""

import json
import pathlib
import sqlite3

import pytest

from recall import sources


# ── pi：纯文本 JSONL，type=message ─────────────────────────────────


def _pi_line(role, text, ts="2026-01-01T00:00:00"):
    return json.dumps(
        {
            "type": "message",
            "timestamp": ts,
            "message": {"role": role, "content": [{"type": "text", "text": text}]},
        }
    )


def test_pi_reader_reads_message_text(tmp_path):
    path = tmp_path / "abc12345-def.jsonl"
    path.write_text(
        "\n".join(
            [
                _pi_line("user", "Remember the Amber protocol."),
                json.dumps({"type": "other", "timestamp": "x"}),  # 非 message
                _pi_line("assistant", "ok", ts="2026-01-01T00:00:01"),  # 太短
                "not json at all",
            ]
        )
        + "\n"
    )
    rows = list(sources.iter_pi_file(path))
    assert len(rows) == 1
    role, ts, text = rows[0]
    assert role == "user" and ts == "2026-01-01T00:00:00"
    assert text.startswith("[abc12345] ") and "Amber protocol" in text


# ── dsh：zstd 压缩的事件流 ─────────────────────────────────────────


def _write_streamed_zstd(path: pathlib.Path, payload: bytes):
    """按 DSH 的写法落盘：流式压缩，帧头里**没有** content size。"""
    import zstandard

    with path.open("wb") as handle, zstandard.ZstdCompressor().stream_writer(handle) as writer:
        writer.write(payload)


def _dsh_event(kind, data, ms):
    return json.dumps({"type": kind, "seq": 1, "time": ms, "data": data})


def test_dsh_reader_handles_streamed_frames(tmp_path):
    """回归：流式 zstd 帧没有 content size，用 decompress() 会直接失败。"""
    payload = "\n".join(
        [
            _dsh_event(
                "user/message",
                {"role": "user", "content": [{"type": "text", "text": "看下我的 npm 环境变量"}]},
                1790752622209,
            ),
            _dsh_event(
                "assistant/message",
                {
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "先看 PATH 和 npm 的配置"}],
                    }
                },
                1790752623000,
            ),
            _dsh_event("tool/call", {"name": "bash", "arguments": "{}"}, 1790752624000),
            _dsh_event("tool/result", {"message": {"content": []}}, 1790752625000),
        ]
    ).encode()
    path = tmp_path / "session.v4.jsonl.zstd"
    _write_streamed_zstd(path, payload)

    rows = list(sources.iter_dsh_file(path))
    assert [r[0] for r in rows] == ["user", "assistant"]  # 工具事件被跳过
    assert "npm" in rows[0][2] and "PATH" in rows[1][2]
    # epoch 毫秒要转成和别的 agent 一致的 ISO 字符串
    assert rows[0][1].startswith("2026-") and len(rows[0][1]) == 19


def test_dsh_reader_reports_a_missing_zstd(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in ("compression.zstd", "compression", "zstandard"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="需要 zstd"):
        sources._decompress(b"\x28\xb5\x2f\xfd")


# ── codex：纯文本 JSONL，response_item ────────────────────────────


def _codex_line(kind, payload, ts="2026-05-19T15:41:06.123Z"):
    return json.dumps({"type": kind, "timestamp": ts, "ordinal": 1, "payload": payload})


def test_codex_reader_takes_user_and_assistant_only(tmp_path):
    path = tmp_path / "rollout-2026-05-19.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session_meta", "payload": {"cwd": "/x"}}),
                _codex_line(
                    "response_item",
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Fix the timeout."}],
                    },
                ),
                _codex_line(
                    "response_item",
                    {
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Bumped it to 30s."}],
                    },
                ),
                # developer 提示与推理内容不索引
                _codex_line(
                    "response_item",
                    {
                        "role": "developer",
                        "content": [{"type": "input_text", "text": "system prompt"}],
                    },
                ),
                _codex_line(
                    "response_item",
                    {"role": "reasoning", "content": [{"type": "text", "text": "thinking"}]},
                ),
                # 工具调用没有 role，也不索引
                _codex_line(
                    "response_item", {"type": "function_call", "name": "shell", "arguments": "{}"}
                ),
                _codex_line("event_msg", {"type": "token_count"}),
            ]
        )
        + "\n"
    )
    rows = list(sources.iter_codex_file(path))
    assert [r[0] for r in rows] == ["user", "assistant"]
    assert rows[0][1] == "2026-05-19T15:41:06"
    assert "timeout" in rows[0][2] and "30s" in rows[1][2]


# ── opencode：SQLite ──────────────────────────────────────────────


def _opencode_db(path: pathlib.Path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE session(id TEXT, slug TEXT)")
    con.execute("CREATE TABLE message(id TEXT, session_id TEXT, data TEXT)")
    con.execute(
        "CREATE TABLE part(id TEXT, message_id TEXT, session_id TEXT,"
        " time_created INTEGER, data TEXT)"
    )
    con.execute("INSERT INTO session VALUES('ses_1','lucky-comet')")
    con.execute(
        "INSERT INTO message VALUES('msg_1','ses_1',?)",
        (json.dumps({"role": "user"}),),
    )
    con.execute(
        "INSERT INTO message VALUES('msg_2','ses_1',?)",
        (json.dumps({"role": "assistant"}),),
    )
    parts = [
        ("prt_1", "msg_1", "text", "ping the gateway twice", 1777000406987),
        ("prt_2", "msg_2", "text", "the gateway answered", 1777000407987),
        ("prt_3", "msg_2", "tool", None, 1777000408987),  # 非 text 不索引
        ("prt_4", "msg_2", "text", "ok", 1777000409987),  # 太短
    ]
    for pid, mid, kind, text, created in parts:
        data = {"type": kind}
        if text is not None:
            data["text"] = text
        con.execute(
            "INSERT INTO part VALUES(?,?,?,?,?)",
            (pid, mid, "ses_1", created, json.dumps(data)),
        )
    con.commit()
    con.close()


def test_opencode_reader_joins_parts_with_message_roles(tmp_path):
    db = tmp_path / "opencode.db"
    _opencode_db(db)
    rows = list(sources.iter_opencode_db(db))
    assert [r[0] for r in rows] == ["lucky-comet", "lucky-comet"]
    assert [r[2] for r in rows] == ["user", "assistant"]  # role 来自 message
    assert "gateway twice" in rows[0][3]
    assert rows[0][1].startswith("20") and len(rows[0][1]) == 19  # 毫秒 → ISO


# ── 统一入口 ──────────────────────────────────────────────────────


def test_iter_records_names_sources_per_agent(tmp_path):
    (tmp_path / "sub").mkdir()
    _opencode_db(tmp_path / "x.db")
    pi = tmp_path / "sub" / "abc.jsonl"
    pi.write_text(_pi_line("user", "a remembered pi fact") + "\n")

    pi_rows = list(sources.iter_records("pi", tmp_path))
    assert pi_rows and pi_rows[0][0] == "sub/abc.jsonl"

    db_rows = list(sources.iter_records("opencode", tmp_path / "x.db"))
    assert db_rows and db_rows[0][0] == "x:lucky-comet"  # 库名 + 会话 slug


def test_agent_registry_covers_the_documented_set():
    assert sources.AGENTS == ("pi", "dsh", "codex", "opencode", "codebuddy", "workbuddy")
    assert all(sources.LABELS[a] for a in sources.AGENTS)


# ── codebuddy / workbuddy：两个产品共用一套写入器 ──────────────────


def _cb_line(kind, **fields):
    return json.dumps({"type": kind, "sessionId": "ses_1", "cwd": "/tmp/x", **fields})


def test_codebuddy_reader_takes_top_level_role_and_content(tmp_path):
    """与 pi 不同：正文在**顶层** content，时间戳是 epoch 毫秒。"""
    path = tmp_path / "0dd227b4-a66a-4630-ab87-ec994dd57d47.jsonl"
    path.write_text(
        "\n".join(
            [
                _cb_line(
                    "message",
                    role="user",
                    timestamp=1789543534454,
                    content=[{"type": "input_text", "text": "把网关的超时改成 30 秒"}],
                ),
                _cb_line(
                    "message",
                    role="assistant",
                    timestamp=1789543589746,
                    content=[{"type": "output_text", "text": "已经改好，并补了重试退避。"}],
                ),
                # 这些都不是对话正文，必须跳过
                _cb_line("function_call", name="bash", arguments="{}"),
                _cb_line("function_call_result", output="huge tool output"),
                _cb_line("reasoning", content=[{"type": "text", "text": "thinking"}]),
                _cb_line("file-history-snapshot", isSnapshotUpdate=True),
                _cb_line("ai-title", aiTitle="网关超时"),
                # 短消息丢弃
                _cb_line("message", role="assistant", timestamp=1789543589000, content="ok"),
                # 非 user/assistant 的 message
                _cb_line("message", role="system", timestamp=1789543589000, content="prompt"),
            ]
        )
        + "\n"
    )
    rows = list(sources.iter_codebuddy_file(path))
    assert [r[0] for r in rows] == ["user", "assistant"]
    assert "30 秒" in rows[0][2] and "退避" in rows[1][2]
    # epoch 毫秒 → 和别的 agent 一致的 ISO 字符串
    assert rows[0][1].startswith("20") and len(rows[0][1]) == 19


def test_both_tencent_agents_share_one_reader(tmp_path):
    """codebuddy 与 workbuddy 只是根目录不同，读取器必须是同一个。"""
    assert sources.AGENTS.index("codebuddy") + 1 == sources.AGENTS.index("workbuddy")
    for agent in ("codebuddy", "workbuddy"):
        root = tmp_path / agent
        root.mkdir()
        (root / "p").mkdir()
        (root / "p" / "s.jsonl").write_text(
            _cb_line(
                "message",
                role="user",
                timestamp=1789543534454,
                content=[{"type": "input_text", "text": f"{agent} 的历史正文"}],
            )
            + "\n"
        )
        rows = list(sources.iter_records(agent, root))
        assert rows and rows[0][0] == "p/s.jsonl"
        assert agent in rows[0][3]


def test_workbuddy_is_skipped_cleanly_when_absent(tmp_path):
    """目录不存在时 available() 为 False —— 只有过 IDE 的机器就是这种情况。"""
    missing = tmp_path / "nope"
    assert not missing.exists()
    import recall.sources as s

    original = s.source_path
    s.source_path = lambda agent: missing  # type: ignore[assignment]
    try:
        assert s.available("workbuddy") is False
    finally:
        s.source_path = original  # type: ignore[assignment]


def test_dsh_glob_matches_the_zstd_extension(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "session.v4.jsonl.zstd").write_bytes(b"")
    (tmp_path / "a" / "notes.jsonl").write_bytes(b"")  # 非 DSH 会话
    found = sources._jsonl_files(tmp_path, sources.DSH_SESSION_GLOB)
    assert [p.name for p in found] == ["session.v4.jsonl.zstd"]


# ── 内存有界读取（回归）────────────────────────────────────────────


def test_oversized_lines_are_dropped_without_breaking_the_stream(tmp_path):
    """回归：一个 986 MB 的 Codex 会话曾让读取阶段 RSS 冲到 990 MB。

    文本模式逐行迭代会按文件大小吃内存；改成二进制分块之后，超长行必须被就地
    丢弃，而且不能影响它前后的行（包括没有结尾换行的最后一行）。
    """
    path = tmp_path / "rollout.jsonl"
    with path.open("wb") as handle:
        handle.write(b"first\n")
        handle.write(b"y" * (sources.MAX_LINE_BYTES + 7) + b"\n")
        handle.write(b"second\n")
        handle.write(b"tail-without-newline")

    stats: dict = {}
    with path.open("rb") as handle:
        lines = list(sources._bounded_lines(handle, sources.MAX_LINE_BYTES, stats))
    assert lines == [b"first", b"second", b"tail-without-newline"]
    assert stats == {"skipped_large_lines": 1}


def test_oversized_line_at_end_of_file_is_counted(tmp_path):
    path = tmp_path / "rollout.jsonl"
    with path.open("wb") as handle:
        handle.write(b"ok\n")
        handle.write(b"z" * (sources.MAX_LINE_BYTES + 1))
    stats: dict = {}
    with path.open("rb") as handle:
        lines = list(sources._bounded_lines(handle, sources.MAX_LINE_BYTES, stats))
    assert lines == [b"ok"]
    assert stats == {"skipped_large_lines": 1}


def test_readers_skip_a_giant_line_but_keep_the_rest(tmp_path):
    """端到端：中间夹一条超长记录时，前后的正常记录都要留下来。"""
    path = tmp_path / "rollout.jsonl"
    good = _codex_line(
        "response_item",
        {"role": "user", "content": [{"type": "input_text", "text": "keep this one"}]},
    )
    huge = json.dumps(
        {
            "type": "response_item",
            "timestamp": "2026-05-19T15:41:07.000Z",
            "payload": {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "x" * 40}],
            },
            "blob": "z" * (sources.MAX_LINE_BYTES + 10),
        }
    )
    after = _codex_line(
        "response_item",
        {"role": "assistant", "content": [{"type": "output_text", "text": "and this one"}]},
    )
    path.write_text("\n".join([good, huge, after]) + "\n")

    stats: dict = {}
    rows = list(sources.iter_codex_file(path, stats))
    assert [r[2] for r in rows] == ["keep this one", "and this one"]
    assert stats == {"skipped_large_lines": 1}
