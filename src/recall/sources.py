"""各 agent 的会话读取器：pi / dsh / codex / opencode / codebuddy / workbuddy。

六个 agent 的落盘格式各不相同，这里统一成同一种记录
``(source, ts, role, text)``，再交给 engine 分块与索引。

只取**对话正文**（user / assistant 的文本），不索引工具调用与工具输出：
那些又大又是噪声，会让索引迅速膨胀。各 agent 的判定方式：

| agent | 位置 | 格式 | 取哪些 |
| --- | --- | --- | --- |
| pi | `~/.pi/agent/sessions` | 纯文本 JSONL | `type=message` 里的 text 块 |
| dsh | `~/.dsh/sessions` | zstd 压缩 JSONL 事件流 | `user/message`、`assistant/message` |
| codex | `~/.codex/sessions` | 纯文本 JSONL | `response_item` 中 `payload.role` 为 user/assistant |
| opencode | `~/.local/share/opencode/opencode.db` | SQLite | `part` 表中 `type=text` 的条目 |
| codebuddy | `~/.codebuddy/projects` | 纯文本 JSONL | `type=message` 的 `role`/`content` |
| workbuddy | `~/.workbuddy/projects` | 纯文本 JSONL | 同上（两个产品共用一套写入器） |
| claude | `~/.claude/projects` | 纯文本 JSONL | `type` 是 user/assistant，正文在 `message.content` |

DSH 的会话是 **zstd 压缩**的：Python 3.14+ 自带 ``compression.zstd``，更早的版本需要
``pip install '.[dsh]'``（会装 ``zstandard``）。缺这个库时 DSH 会被跳过并给出提示，
不影响其它 agent。
"""

from __future__ import annotations

import io
import json
import pathlib
import sqlite3
from datetime import datetime, timezone
from typing import Iterator

from .config import (
    CLAUDE_SESSIONS_DIR,
    CODEBUDDY_SESSIONS_DIR,
    CODEX_SESSIONS_DIR,
    DSH_SESSIONS_DIR,
    OPENCODE_DB,
    SESSIONS_DIR,
    WORKBUDDY_SESSIONS_DIR,
)

#: 支持的 agent，顺序即 `recall index sessions` 的处理顺序。
AGENTS = ("pi", "dsh", "codex", "opencode", "codebuddy", "workbuddy", "claude")

LABELS = {
    "pi": "Pi",
    "dsh": "DeepSeek Harness",
    "codex": "Codex CLI",
    "opencode": "opencode",
    "codebuddy": "CodeBuddy",
    "workbuddy": "WorkBuddy",
    "claude": "Claude Code",
}

#: 单条记录（一次发言）的字符上限。超过就跳过 —— 一次粘贴几 MB 的日志进索引
#: 会产出上千个窗口，把索引撑爆而检索价值很低。
MAX_RECORD_CHARS = 200_000

#: 单行 JSON 的字节上限，**在 json.loads 之前**就按它跳过。
#:
#: 这不是可选优化：Codex 历史实测 3.77 GB / 31 万行，其中 507 行超过 1 MB、合计
#: 2.09 GB（最大单行 13.5 MB —— 工具输出、图片 base64、加密推理块）。把这种行整个
#: 解析成对象图会让峰值内存冲到 2 GB 以上，而它们要么根本不在我们读取的字段里，
#: 要么正文超过 MAX_RECORD_CHARS 本来就会被丢弃。
#:
#: 1 MiB 的依据：能留下记录的正文 ≤ MAX_RECORD_CHARS 字符，UTF-8 最坏 4 字节/字符
#: 再加 JSON 转义也放得下，所以「以文本为主」的记录不会因为这条被砍掉。
MAX_LINE_BYTES = 1 << 20

#: 太短的记录没有检索价值（与 engine 的 MIN_CHARS 一致）。
MIN_CHARS = 8

#: DSH 会话文件名。外层扩展名是 .zstd，所以 `*.jsonl` 匹配不到。
DSH_SESSION_GLOB = "**/*.jsonl.zstd"


def source_path(agent: str) -> pathlib.Path:
    """该 agent 的历史位置（目录或 SQLite 文件）。"""
    return {
        "pi": SESSIONS_DIR,
        "dsh": DSH_SESSIONS_DIR,
        "codex": CODEX_SESSIONS_DIR,
        "opencode": OPENCODE_DB,
        "codebuddy": CODEBUDDY_SESSIONS_DIR,
        "workbuddy": WORKBUDDY_SESSIONS_DIR,
        "claude": CLAUDE_SESSIONS_DIR,
    }[agent]


def available(agent: str) -> bool:
    return source_path(agent).exists()


def _iso_from_ms(value) -> str:
    """epoch 毫秒 → 和别的 agent 一致的 ISO 字符串（本地时区，精确到秒）。"""
    try:
        stamp = float(value) / 1000.0
    except (TypeError, ValueError):
        return ""
    try:
        return datetime.fromtimestamp(stamp).astimezone().strftime("%Y-%m-%dT%H:%M:%S")
    except (OverflowError, OSError, ValueError):
        return ""


def _blocks_to_text(content) -> str:
    """把各家的 content 结构拍平成纯文本。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(p for p in parts if p)


def _note(stats, key: str, amount: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + amount


def _bounded_lines(handle, limit: int, stats=None):
    """按行读二进制流，**绝不把整个文件读进内存**。

    这是必须的，不是优化：实测一个 986 MB 的 Codex 会话文件，用 Python 文本模式
    ``for line in fh`` 迭代会让 RSS 冲到 990 MB（和文件大小几乎 1:1），换成二进制
    分块 + 自己按 ``\\n`` 切分只有 21 MB。

    超过 ``limit`` 的行就地丢弃：已经缓冲的部分直接扔掉，再一直读到下一个换行为止，
    全程不累积。
    """
    pending = b""
    skipping = False
    while True:
        chunk = handle.read(1 << 20)
        if not chunk:
            break
        if skipping:
            cut = chunk.find(b"\n")
            if cut < 0:
                continue
            skipping = False
            chunk = chunk[cut + 1 :]
        pending += chunk
        while True:
            cut = pending.find(b"\n")
            if cut < 0:
                break
            line, pending = pending[:cut], pending[cut + 1 :]
            if len(line) > limit:
                _note(stats, "skipped_large_lines")
                continue
            yield line
        if len(pending) > limit:
            _note(stats, "skipped_large_lines")
            pending = b""
            skipping = True
    if pending:
        if len(pending) <= limit:
            yield pending
        else:
            _note(stats, "skipped_large_lines")


def _lines(path: pathlib.Path, stats=None):
    """逐行给 JSON dict；坏行直接跳过（会话文件可能是追加中的半截）。

    超大行在 ``json.loads`` **之前**就丢掉，见 :data:`MAX_LINE_BYTES`；
    读取本身走 :func:`_bounded_lines`，内存与文件大小无关。
    """
    try:
        handle = path.open("rb")
    except OSError:
        return
    with handle:
        for raw in _bounded_lines(handle, MAX_LINE_BYTES, stats):
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                _note(stats, "skipped_bad_lines")
                continue
            if isinstance(record, dict):
                yield record


def _decompress(data: bytes) -> bytes:
    """解 zstd。

    DSH 的会话是**流式**写出来的，帧头里往往没有 content size ——
    那种帧用要求已知大小的 ``decompress()`` 会直接抛
    ``could not determine content size in frame header``，必须走流式接口。
    """
    try:
        from compression.zstd import ZstdDecompressor  # Python 3.14+
    except ImportError:
        pass
    else:
        return ZstdDecompressor().decompressobj().decompress(data)
    try:
        import zstandard
    except ImportError:
        raise RuntimeError(
            "读取 DSH 会话需要 zstd 解压：Python 3.14+ 自带 compression.zstd，"
            "更早的版本请 pip install '.[dsh]'"
        ) from None
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)) as reader:
        return reader.read()


# ── 各 agent 的读取器 ────────────────────────────────────────────────


def iter_pi_file(path: pathlib.Path, stats=None) -> Iterator[tuple[str, str, str]]:
    """Pi：``type=message`` + ``message.content`` 的 text 块。"""
    sid = path.stem
    for record in _lines(path, stats):
        if record.get("type") != "message":
            continue
        message = record.get("message") or {}
        if not isinstance(message, dict):
            continue
        text = _blocks_to_text(message.get("content")).strip()
        if len(text) < MIN_CHARS:
            continue
        ts = (record.get("timestamp") or "")[:19]
        yield message.get("role", "?"), ts, f"[{sid[:8]}] {text}"


def iter_dsh_file(path: pathlib.Path, stats=None) -> Iterator[tuple[str, str, str]]:
    """DSH：zstd 压缩的事件流；只取 user/assistant 的 message 事件。

    解压后**不**再 ``decode()`` 出一份 str 再 ``splitlines()`` 出一份行列表 ——
    那对一个几十 MB 的会话就是三份副本。直接在解压后的字节上按 ``BytesIO`` 迭代。
    """
    raw = _decompress(path.read_bytes())
    for line in io.BytesIO(raw):
        line = line.strip()
        if not line:
            continue
        if len(line) > MAX_LINE_BYTES:
            _note(stats, "skipped_large_lines")
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        data = record.get("data") if isinstance(record.get("data"), dict) else {}
        if kind == "user/message":
            text = _blocks_to_text(data.get("content"))
            role = data.get("role") or "user"
        elif kind == "assistant/message":
            message = data.get("message") if isinstance(data.get("message"), dict) else {}
            text = _blocks_to_text(message.get("content"))
            role = message.get("role") or "assistant"
        else:
            continue
        text = text.strip()
        if len(text) < MIN_CHARS:
            continue
        yield role, _iso_from_ms(record.get("time")), text


def iter_codex_file(path: pathlib.Path, stats=None) -> Iterator[tuple[str, str, str]]:
    """Codex：``response_item`` 里 payload.role 为 user/assistant 的正文。"""
    for record in _lines(path, stats):
        if record.get("type") != "response_item":
            continue
        payload = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        role = payload.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _blocks_to_text(payload.get("content")).strip()
        if len(text) < MIN_CHARS:
            continue
        yield role, (record.get("timestamp") or "")[:19], text


def iter_codebuddy_file(path: pathlib.Path, stats=None) -> Iterator[tuple[str, str, str]]:
    """CodeBuddy / WorkBuddy：``type=message`` 行的顶层 ``role`` 与 ``content``。

    两个腾讯 agent 共用同一套本地写入器（会话文件里带 ``__codebuddyLocal`` 字段），
    所以共用一个读取器。同一个文件里还有 ``function_call`` / ``function_call_result``
    / ``reasoning`` / ``file-history-snapshot`` / ``ai-title``，都跳过。

    与 pi 的区别：正文不在 ``message.content`` 里，而是**顶层**的 ``content``；
    ``timestamp`` 是 epoch 毫秒。
    """
    for record in _lines(path, stats):
        if record.get("type") != "message":
            continue
        role = record.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _blocks_to_text(record.get("content")).strip()
        if len(text) < MIN_CHARS:
            continue
        yield role, _iso_from_ms(record.get("timestamp")), text


def iter_claude_file(  # noqa: D401
    path: pathlib.Path, stats=None
) -> Iterator[tuple[str, str, str]]:
    """Claude Code：``type`` 直接是 ``user``/``assistant``，正文在 ``message.content``。

    与 pi 同源（pi 的格式就派生自这套），区别是这里 ``type`` 本身就是角色，而不是
    ``"message"``，且 ``content`` 可能是纯字符串。

    ``thinking`` / ``tool_use`` / ``tool_result`` 块都跳过 —— 前者的文本在
    ``thinking`` 字段而不是 ``text``，后两者根本没有 ``text`` 字段，所以
    :func:`_blocks_to_text` 天然只收到正文。``mode`` / ``permission-mode`` /
    ``ai-title`` / ``last-prompt`` / ``file-history-snapshot`` 这类行同样跳过。

    ``isSidechain`` 的子代理会话**保留**：它们也是真实对话，来源路径已经能区分。
    """
    for record in _lines(path, stats):
        role = record.get("type")
        if role not in ("user", "assistant"):
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        text = _blocks_to_text(message.get("content")).strip()
        if len(text) < MIN_CHARS:
            continue
        yield role, (record.get("timestamp") or "")[:19], text


def iter_opencode_db(path: pathlib.Path) -> Iterator[tuple[str, str, str]]:
    """opencode：SQLite 里 ``part`` 表中 type=text 的条目，靠 message 拿 role。"""
    uri = f"file:{path}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        rows = con.execute(
            """
            SELECT COALESCE(s.slug, s.id) AS source,
                   json_extract(m.data, '$.role') AS role,
                   json_extract(p.data, '$.text') AS text,
                   p.time_created
            FROM part p
            JOIN message m ON m.id = p.message_id
            JOIN session s ON s.id = p.session_id
            WHERE json_extract(p.data, '$.type') = 'text'
            ORDER BY p.session_id, p.time_created, p.id
            """
        )
        for source, role, text, created in rows:
            if not isinstance(text, str):
                continue
            text = text.strip()
            if len(text) < MIN_CHARS:
                continue
            yield str(source or "opencode"), _iso_from_ms(created), role or "?", text
    finally:
        con.close()


# ── 统一入口 ─────────────────────────────────────────────────────────


def _jsonl_files(root: pathlib.Path, pattern: str) -> list[pathlib.Path]:
    if root.is_file():
        return [root]
    return sorted(root.glob(pattern))


def iter_records(
    agent: str, root: pathlib.Path, stats: dict | None = None
) -> Iterator[tuple[str, str, str, str]]:
    """yield ``(source, ts, role, text)``；``source`` 在 agent 内唯一且稳定。

    ``stats`` 是可选的可变计数器，用来回报被跳过的行（超大行 / 坏行）。
    """
    if agent == "opencode":
        for source, ts, role, text in iter_opencode_db(root):
            yield _session_source(root, source), ts, role, text
        return

    readers = {
        "pi": iter_pi_file,
        "dsh": iter_dsh_file,
        "codex": iter_codex_file,
        # 腾讯这两个产品共用一套写入器，所以是同一个读取器、两个根目录。
        "codebuddy": iter_codebuddy_file,
        "workbuddy": iter_codebuddy_file,
        "claude": iter_claude_file,
    }
    reader = readers[agent]
    # 只有 opencode 走 SQLite，其余都是文件型读取器，都回报跳过量。
    takes_stats = agent != "opencode"

    if agent == "dsh":
        files = _jsonl_files(root, DSH_SESSION_GLOB)
    else:
        files = _jsonl_files(root, "**/*.jsonl")

    for path in files:
        try:
            relative = path.relative_to(root).as_posix() if root.is_dir() else path.name
        except ValueError:
            relative = path.name
        produced = reader(path, stats) if takes_stats else reader(path)
        for role, ts, text in produced:
            yield relative, ts, role, text


def _session_source(root: pathlib.Path, source: str) -> str:
    """opencode 没有文件路径，用（库名 + ）会话 slug 作为稳定来源标识。"""
    if root.is_file():
        return f"{root.stem}:{source}"
    return source
