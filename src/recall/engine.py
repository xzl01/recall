"""Local history and text indexing, hybrid retrieval, and neighbor reads."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import pathlib
import sqlite3
import sys
import time

import numpy as np

from .embedding import Embedder
from .daemon import resolve_embedder
from .identity import model_identity
from . import sources
from .sources import iter_pi_file
from .support import (
    token_windows,
    compact_query,
    query_terms,
    terms,
    fts_text,
    locked,
    transactional,
    WINDOW_TOKENS,
    OVERLAP_TOKENS,
)
from .config import (
    CLAUDE_SESSIONS_DIR,
    CODEBUDDY_SESSIONS_DIR,
    CODEX_SESSIONS_DIR,
    DATA_DIR,
    DEFAULT_DEVICE,
    DSH_SESSIONS_DIR,
    OPENCODE_DB,
    SESSIONS_DIR,
    WORKBUDDY_SESSIONS_DIR,
)

HOME = pathlib.Path.home()
LAB = DATA_DIR
INDEX = LAB / "index"
VEC = INDEX / "vectors.npy"
DB = INDEX / "meta.db"

# 各 agent 的历史位置。都放在模块级，方便按 agent 覆盖，也便于测试隔离。
SESSIONS = SESSIONS_DIR
DSH_SESSIONS = DSH_SESSIONS_DIR
CODEX_SESSIONS = CODEX_SESSIONS_DIR
OPENCODE_PATH = OPENCODE_DB
CODEBUDDY_SESSIONS = CODEBUDDY_SESSIONS_DIR
CLAUDE_SESSIONS = CLAUDE_SESSIONS_DIR
WORKBUDDY_SESSIONS = WORKBUDDY_SESSIONS_DIR

SESSION_ROOTS = {
    "pi": lambda: SESSIONS,
    "dsh": lambda: DSH_SESSIONS,
    "codex": lambda: CODEX_SESSIONS,
    "opencode": lambda: OPENCODE_PATH,
    "codebuddy": lambda: CODEBUDDY_SESSIONS,
    "workbuddy": lambda: WORKBUDDY_SESSIONS,
    "claude": lambda: CLAUDE_SESSIONS,
}

DIM = 512
MIN_CHARS = 8

# 默认排除的目录名（构建产物 / 依赖 / 版本库 / 第三方 vendor SDK）
EXCLUDE_DIRS = {
    "build",
    "target",
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "out",
    ".next",
    ".cache",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "obj",
    "cmake-build-debug",
    "public",
    "_site",
    ".dart_tool",
}
# 默认跳过的顶级目录（体积巨大且属于第三方，需要时用 --include-vendor 打开）
VENDOR_TOPS = {"vendor", "third_party", "external"}
MAX_FILE_BYTES = 1_200_000  # 超过就跳过（日志/生成的巨型文件）


# ── 存储 ──────────────────────────────────────────────────────
def chunk_key(source: str, ts: str, role: str, text: str) -> str:
    """片段级去重键。必须是「只由落库的字段算出来」的纯函数，这样老记录能就地回填。

    为什么不能用文件级判断：pi 的会话 jsonl 是**持续追加**的，同一个文件今天写、
    明天接着写。只要文件里已有任一片段就跳过整个文件，尾巴就永远进不来。
    """
    raw = "\x00".join((source or "", ts or "", role or "", text or ""))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


IDENTITY_KEY = "embedding_id"


def recorded_identity(con) -> str | None:
    """索引里记录的嵌入模型身份；旧索引没有 meta 表时返回 None。"""
    has_meta = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if not has_meta:
        return None
    found = con.execute("SELECT value FROM meta WHERE key=?", (IDENTITY_KEY,)).fetchone()
    return found[0] if found else None


def stamp_identity(con, value: str) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (IDENTITY_KEY, value))
    globals()["_INDEX_DIRTY"] = True


def guard_identity(con, embedder, *, stamp: bool) -> str | None:
    """索引记录的模型身份必须和当前要用的模型一致。

    换一份权重（哪怕是同为 512 维的 BERT）会整体改变向量空间，混进同一个索引会
    让检索**静默**给出错误结果。所以这里硬性拦下，要求重建，而不是照常返回。
    """
    model_dir = getattr(embedder, "model_dir", None)
    if model_dir is None:
        return None
    current = model_identity(model_dir)
    if current is None:
        # 目录里没有可识别的权重（测试替身/自定义布局）：宁可跳过校验也不拒绝服务。
        # 真实后端不可能走到这里，因为它们本来就要求加载标准权重文件。
        return None
    recorded = recorded_identity(con)
    if recorded is None:
        if stamp:
            stamp_identity(con, current)
        else:
            print(
                "提示：索引未记录嵌入模型指纹（旧索引）。跑一次 recall index 会补记录，"
                "之后才能校验模型有没有被换过。",
                file=sys.stderr,
            )
        return current
    if recorded != current:
        sys.exit(
            "索引是用另一份嵌入模型建立的，继续使用会让检索结果静默变错：\n"
            f"  索引记录: {recorded}\n"
            f"  当前模型: {current}\n"
            f"  模型目录: {model_dir}\n"
            "请用新的 RECALL_HOME 重建索引，或换回原来那份模型。"
        )
    return current


def db_connect():
    INDEX.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS chunks(
        id INTEGER PRIMARY KEY, corpus TEXT, source TEXT, ts TEXT,
        role TEXT, text TEXT, key TEXT, parent TEXT, start INTEGER, end INTEGER)""")
    columns = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    for name, kind in [
        ("key", "TEXT"),
        ("parent", "TEXT"),
        ("start", "INTEGER"),
        ("end", "INTEGER"),
    ]:
        if name not in columns:
            con.execute(f"ALTER TABLE chunks ADD COLUMN {name} {kind}")
            globals()["_INDEX_DIRTY"] = True
    missing = con.execute("SELECT id,source,ts,role,text FROM chunks WHERE key IS NULL").fetchall()
    if missing:
        con.executemany(
            "UPDATE chunks SET key=? WHERE id=?",
            [(chunk_key(s, t, r, x), i) for i, s, t, r, x in missing],
        )
    for name in ["corpus", "key", "parent"]:
        con.execute(f"CREATE INDEX IF NOT EXISTS i_{name} ON chunks({name})")
    exists = con.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone()
    con.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='unicode61')"
    )
    if not exists:
        con.executemany(
            "INSERT INTO chunks_fts(rowid,text) VALUES(?,?)",
            [(i, fts_text(x)) for i, x in con.execute("SELECT id,text FROM chunks")],
        )
        globals()["_INDEX_DIRTY"] = True
    con.commit()
    return con


def readonly():
    if not DB.exists():
        sys.exit("索引是空的，先跑 recall --index")
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def sync_fts(con):
    con.execute("DELETE FROM chunks_fts")
    con.executemany(
        "INSERT INTO chunks_fts(rowid,text) VALUES(?,?)",
        [(i, fts_text(x)) for i, x in con.execute("SELECT id,text FROM chunks")],
    )


def write_vectors(mat: np.ndarray):
    # 注意：不能用 np.save(路径) —— 它会给不以 .npy 结尾的名字再补一个 .npy
    tmp = INDEX / "vectors.new.npy"
    with open(tmp, "wb") as fh:
        np.save(fh, np.asarray(mat, dtype=np.float32))
    tmp.replace(VEC)


def renumber_ids(con):
    """把 chunks.id 重排为连续的 1..N。

    search 依赖「id = 向量行号 + 1」这个不变式。因为 id 升序与向量行序一致，
    重排只是把 id 改成它在升序里的排名 —— **向量矩阵不需要动**。
    先整体挪到高位，避免主键冲突。
    """
    ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
    if ids == list(range(1, len(ids) + 1)):
        return
    con.execute("UPDATE chunks SET id = id + 1000000000")
    con.executemany(
        "UPDATE chunks SET id = ? WHERE id = ?",
        [(new, old + 1000000000) for new, old in enumerate(ids, start=1)],
    )


def purge_corpus(con, corpus: str) -> int:
    """删掉某语料的全部行，并从向量矩阵里摘掉对应行（维持 id = 行号+1）。"""
    ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
    drop = {r[0] for r in con.execute("SELECT id FROM chunks WHERE corpus = ?", (corpus,))}
    if not ids or not drop:
        return 0
    v = np.asarray(load_vectors())
    if v.shape[0] != len(ids):
        sys.exit(f"索引不一致：chunks {len(ids)} 行 vs vectors {v.shape[0]} 行；删掉 {INDEX} 重建")
    keep = np.array([i not in drop for i in ids])
    write_vectors(v[keep])
    con.execute("DELETE FROM chunks WHERE corpus = ?", (corpus,))
    renumber_ids(con)
    sync_fts(con)
    con.commit()
    return len(drop)


def load_vectors():
    if not VEC.exists():
        return np.zeros((0, DIM), dtype=np.float32)
    return np.load(VEC, mmap_mode="r")


def append_vectors(new: np.ndarray):
    old = load_vectors()
    n_old = old.shape[0]
    merged = (
        np.vstack([np.asarray(old), new]).astype(np.float32) if n_old else new.astype(np.float32)
    )
    # 注意：不能用 np.save(路径) —— 它会给不以 .npy 结尾的名字再补一个 .npy
    tmp = INDEX / "vectors.new.npy"
    with open(tmp, "wb") as fh:
        np.save(fh, merged)
    tmp.replace(VEC)
    return merged.shape[0]


# ── 语料读取 ──────────────────────────────────────────────────
def iter_session_chunks(path: pathlib.Path):
    """pi 会话 jsonl → (role, ts, text)"""
    sid = path.stem
    with path.open(errors="replace") as handle:
        for ln in handle:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:
                continue
            if d.get("type") != "message":
                continue
            msg = d.get("message") or {}
            role = msg.get("role", "?")
            parts = []
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") in ("text", "input_text", "output_text"):
                    parts.append(c.get("text", ""))
                elif isinstance(c, str):
                    parts.append(c)
            text = "\n".join(p for p in parts if p).strip()
            if len(text) < MIN_CHARS:
                continue
            ts = (d.get("timestamp") or "")[:19]
            yield role, ts, f"[{sid[:8]}] {text}"


def iter_file_chunks(path: pathlib.Path, chunk_lines=60, overlap=10):
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return
        text = path.read_text(errors="replace")
    except Exception:
        return
    # 跳过二进制/生成物（大量不可打印字符）
    sample = text[:4000]
    if (
        sample
        and sum(1 for c in sample if not c.isprintable() and c not in "\n\r\t") / len(sample) > 0.05
    ):
        return
    lines = text.splitlines()
    step = max(1, chunk_lines - overlap)
    for i in range(0, len(lines), step):
        block = "\n".join(lines[i : i + chunk_lines]).strip()
        if len(block) < 40:
            continue
        yield "file", "", f"[{path}] 行 {i + 1}-{min(i + chunk_lines, len(lines))}\n{block}"


# ── 索引 ──────────────────────────────────────────────────────
#: 兼容旧名字：Pi 的读取器以前叫 iter_session_chunks，定义在本模块里。
iter_session_chunks = iter_pi_file


def session_root(name: str) -> pathlib.Path:
    """某个 agent 的历史位置。

    走模块级的 ``SESSION_ROOTS``（而不是 sources 里那份常量），这样覆盖
    ``e.SESSIONS`` 之类的老做法和测试隔离都继续生效。
    """
    return SESSION_ROOTS[name]()


def _skip_note(stats: dict) -> str:
    """把读取器回报的跳过量整理成一行提示。"""
    bits = []
    if stats.get("skipped_large_lines"):
        bits.append(
            f"{stats['skipped_large_lines']} 行超长数据（>{sources.MAX_LINE_BYTES // 1024}KB）"
        )
    if stats.get("skipped_bad_lines"):
        bits.append(f"{stats['skipped_bad_lines']} 行坏 JSON")
    return "，跳过 " + "、".join(bits) if bits else ""


def migrate_legacy_sessions(con, corpora: set[str]) -> None:
    """把老版本堆在裸 ``sessions`` 语料里的 Pi 会话改名成 ``sessions:pi``。

    老版本只有一个 ``sessions`` 语料，现在是一个 agent 一个。就地改名而不是重索引，
    既有索引不必重跑；只有本次确实要索引 Pi 时才动它。
    """
    if "sessions:pi" not in corpora:
        return
    moved = con.execute(
        "UPDATE chunks SET corpus=? WHERE corpus=?", ("sessions:pi", "sessions")
    ).rowcount
    if moved:
        globals()["_INDEX_DIRTY"] = True
        print(f"  已把 {moved} 条旧的 sessions 记录迁移为 sessions:pi")


@locked(exclusive=True)
@transactional
def cmd_index(a):
    con = db_connect()
    items: list[tuple[str, str, str, str, str]] = []  # corpus, source, ts, role, text
    corpora: set[str] = set()
    t0 = time.perf_counter()

    if a.what == "dir":
        root = pathlib.Path(a.path).expanduser().resolve()
        corpus_name = f"dir:{root.name}:" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        corpora.add(corpus_name)
        exts = tuple(e.strip() for e in a.ext.split(",") if e.strip())
        skip_dirs = set(EXCLUDE_DIRS)
        if not a.include_vendor:
            skip_dirs |= VENDOR_TOPS
        skip_dirs |= {d.strip() for d in (a.exclude or "").split(",") if d.strip()}
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel = pathlib.Path(dirpath).relative_to(root)
            # 顶级 vendor 目录整体跳过
            if rel.parts and rel.parts[0] in skip_dirs:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip_dirs and not d.startswith(".")]
            for fn in filenames:
                if exts and not fn.endswith(exts):
                    continue
                files.append(pathlib.Path(dirpath) / fn)
        print(f"目录 {root} 匹配 {len(files)} 个文件")
        print(f"  已排除目录: {sorted(skip_dirs)}")
        for f in files:
            for role, ts, text in iter_file_chunks(f):
                items.append((corpus_name, str(f), ts, role, text))
    elif a.what == "sessions" or a.what in sources.AGENTS:
        # 每个 agent 一个独立语料（sessions:pi / sessions:dsh / ...），互不干扰；
        # `--chat` 用的是 `sessions%` 前缀，所以四个语料都能一起检索。
        names = list(sources.AGENTS) if a.what == "sessions" else [a.what]
        long_records = 0
        for name in names:
            root = session_root(name)
            if not root.exists():
                print(f"  {name}（{sources.LABELS[name]}）: 未找到 {root}，跳过")
                continue
            corpus = f"sessions:{name}"
            corpora.add(corpus)
            count = 0
            stats: dict[str, int] = {}
            try:
                for source, ts, role, text in sources.iter_records(name, root, stats):
                    if len(text) > sources.MAX_RECORD_CHARS:
                        long_records += 1
                        continue
                    items.append((corpus, source, ts, role, text))
                    count += 1
            except (RuntimeError, sqlite3.Error) as error:
                # 例如没装 zstd，或者 opencode 的库正在被写。跳过这个 agent，
                # 不能因为它把其它 agent 的索引一起弄失败。
                print(f"  {name}（{sources.LABELS[name]}）: {error}，跳过")
                continue
            print(f"  {name}（{sources.LABELS[name]}）: {count} 条记录{_skip_note(stats)}")
        if long_records:
            print(f"  跳过 {long_records} 条超长记录（>{sources.MAX_RECORD_CHARS} 字符）")
    else:
        sys.exit(f"未知语料: {a.what}")

    migrate_legacy_sessions(con, corpora)

    if a.force:
        for corpus in sorted(corpora):
            n = purge_corpus(con, corpus)
            if n:
                print(f"--force：已清除语料 {corpus} 的 {n} 条旧记录，重新索引")

    # 去重必须按 (corpus, parent) 作用域：parent 键不含 corpus，全局去重会让
    # 「先索引整个项目、再单独索引其中的子目录」这类重叠语料被整段吞掉——
    # 命令成功退出，却一条都没进来，按子语料检索时报「没有内容」。
    seen = {(c, p or k) for c, p, k in con.execute("SELECT corpus,parent,key FROM chunks")}
    fresh = []
    skipped = 0
    for c, source, ts, role, text in items:
        parent = chunk_key(source, ts, role, text)
        if (c, parent) in seen:
            skipped += 1
            continue
        seen.add((c, parent))
        for fragment, start, end in token_windows(text):
            fresh.append((c, source, ts, role, fragment, parent, start, end))
    print(f"待嵌入 {len(fresh)} 个窗口（跳过已索引 {skipped} 条原始记录）")
    if fresh:
        store_items(con, fresh, a.device, a.batch)
    elif a.force:
        globals()["_INDEX_DIRTY"] = True
    con.close()
    cmd_stats(a)


def store_items(con, items, device, batch):
    globals()["_INDEX_DIRTY"] = True
    emb = resolve_embedder(device, batch=batch, verbose=True)
    guard_identity(con, emb, stamp=True)
    next_id = con.execute("SELECT COALESCE(MAX(id),0) FROM chunks").fetchone()[0]
    # 预分配整块向量矩阵：历史语料动辄几万个窗口，先存 list 再 vstack 会同时
    # 驻留两份（几万个 × 512 × 4 字节）。
    matrix = np.empty((len(items), DIM), dtype=np.float32)
    for offset in range(0, len(items), 256):
        group = items[offset : offset + 256]
        vectors = emb.encode([it[4] for it in group])
        matrix[offset : offset + len(group)] = vectors
        rows = []
        for j, (c, source, ts, role, text, parent, start, end) in enumerate(group):
            key = hashlib.sha1(f"{parent}\0{start}\0{end}\0{text}".encode()).hexdigest()[:24]
            rows.append((next_id + j + 1, c, source, ts, role, text, key, parent, start, end))
        con.executemany(
            "INSERT INTO chunks(id,corpus,source,ts,role,text,key,parent,start,end) VALUES(?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        con.executemany(
            "INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", [(r[0], fts_text(r[5])) for r in rows]
        )
        con.commit()
        next_id += len(rows)
        print(f"  已索引 {min(offset + 256, len(items))}/{len(items)}", flush=True)
    if len(items):
        append_vectors(matrix)


@locked(exclusive=True)
@transactional
def cmd_rechunk(a):
    con = db_connect()
    rows = con.execute(
        "SELECT corpus,source,ts,role,text,key,parent,start,end FROM chunks ORDER BY id"
    ).fetchall()
    legacy = sum(not r[6] for r in rows)
    if not legacy:
        print("索引已使用 token 窗口，无需迁移。")
        con.close()
        return
    items = []
    for c, source, ts, role, text, key, parent, start, end in rows:
        if parent:
            items.append((c, source, ts, role, text, parent, start, end))
        else:
            parent = chunk_key(source, ts, role, text)
            items.extend(
                (c, source, ts, role, x, parent, lo, hi) for x, lo, hi in token_windows(text)
            )
    print(f"将 {legacy} 条旧片段迁移为 {len(items)} 个窗口；保留所有语料。", flush=True)
    con.execute("DELETE FROM chunks")
    con.execute("DELETE FROM chunks_fts")
    con.commit()
    write_vectors(np.zeros((0, DIM), dtype=np.float32))
    store_items(con, items, a.device, a.batch)
    con.close()
    cmd_stats(a)


@locked()
def cmd_stats(_a=None):
    con = readonly()
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"索引片段总数: {n}")
    for corpus, count in con.execute(
        "SELECT corpus,COUNT(*) FROM chunks GROUP BY corpus ORDER BY 2 DESC LIMIT 10"
    ):
        print(f"  {corpus:<24} {count}")
    columns = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    legacy = (
        con.execute("SELECT COUNT(*) FROM chunks WHERE parent IS NULL").fetchone()[0]
        if "parent" in columns
        else n
    )
    print(f"分块: {WINDOW_TOKENS} BGE tokens / 重叠 {OVERLAP_TOKENS}；待迁移旧片段: {legacy}")
    v = load_vectors()
    print(f"向量矩阵: {v.shape}  (约 {v.shape[0] * v.shape[1] * 4 / 1e6:.0f} MB)")
    if v.shape != (n, DIM):
        sys.exit("索引行数与向量不一致；从备份恢复，不要单独删数据库行。")
    con.close()


# ── 搜索 ──────────────────────────────────────────────────────
def search_context(con, row, fields, neighbors, cache):
    """Bounded neighbors in the same source; keep record boundaries visible."""
    if not neighbors or not row[7]:
        return row[5], 0, len(row[5]), [row]
    scope = (row[1], row[2])
    if scope not in cache:
        group = con.execute(
            "SELECT " + fields + " FROM chunks WHERE corpus=? AND source=? ORDER BY ts,id", scope
        ).fetchall()
        # Stable parent order for records sharing a timestamp, then source offsets.
        first = {}
        for r in group:
            first.setdefault(r[7] or r[6], r[0])
        group.sort(key=lambda r: (r[3], first[r[7] or r[6]], r[8] or 0, r[0]))
        cache[scope] = (group, {r[0]: i for i, r in enumerate(group)})
    group, positions = cache[scope]
    pos = positions[row[0]]
    selected = group[max(0, pos - neighbors) : pos + neighbors + 1]
    # Files have no chronological record metadata; do not cross their parents.
    if row[1] != "sessions":
        selected = [r for r in selected if r[7] == row[7]]
    text = ""
    last_parent = None
    end = 0
    anchor_start = anchor_end = 0
    for r in selected:
        start = r[8] or 0
        if r[7] != last_parent:
            if text:
                text += "\n\n"
            text += f"[{r[3]} {r[4]}]\n" if r[1] == "sessions" else ""
            end = start
        overlap = max(0, end - start) if r[7] == last_parent else 0
        if r[0] == row[0]:
            anchor_start = len(text) - overlap
        text += r[5][overlap:]
        if r[0] == row[0]:
            anchor_end = len(text)
        end = r[9] if r[9] is not None else start + len(r[5])
        last_parent = r[7]
    return text, anchor_start, anchor_end, selected


def bounded_context(text, anchor_start, anchor_end, limit):
    """Clip around the hit, so a long predecessor cannot hide the anchor."""
    if not limit or len(text) <= limit:
        return text, False
    anchor_size = anchor_end - anchor_start
    if anchor_size >= limit:
        return text[anchor_start : anchor_start + limit], True
    lo = max(0, anchor_start - (limit - anchor_size) // 2)
    hi = min(len(text), lo + limit)
    lo = max(0, hi - limit)
    return text[lo:hi], True


@locked()
def cmd_search(a):
    if a.k < 1:
        sys.exit("-k 必须大于 0")
    if getattr(a, "chars", 4000) < 0:
        sys.exit("--chars 必须 >= 0")
    neighbors = getattr(a, "neighbors", 1)
    if not 0 <= neighbors <= 8:
        sys.exit("--neighbors 范围为 0..8")
    con = readonly()
    vecs = load_vectors()
    n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not n:
        sys.exit("索引是空的，先跑 recall --index")
    if vecs.shape != (n, DIM):
        sys.exit("索引行数与向量不一致，请恢复备份。")
    query, trimmed = compact_query(a.query)
    emb = resolve_embedder(a.device, batch=1)
    guard_identity(con, emb, stamp=False)
    q = emb.encode_query(query)[0]
    scores = np.asarray(vecs) @ q
    where = ""
    params = []
    if a.corpus:
        pats = [c.strip() + "%" for c in a.corpus.split(",") if c.strip()]
        where = " AND (" + " OR ".join("c.corpus LIKE ?" for _ in pats) + ")"
        params = pats
        ids = [r[0] for r in con.execute("SELECT c.id FROM chunks c WHERE 1=1" + where, params)]
    else:
        ids = list(range(1, n + 1))
    if not ids:
        sys.exit(f"语料 {a.corpus} 里没有内容")
    candidate_ids = np.asarray(ids, dtype=np.int64)
    candidate_scores = scores[candidate_ids - 1]
    selected = np.argsort(-candidate_scores, kind="stable")
    ranked = candidate_ids[selected].tolist()
    sem_rank = {i: r + 1 for r, i in enumerate(ranked)}
    qterms = query_terms(query)
    lex_rank = {}
    fts = con.execute("SELECT 1 FROM sqlite_master WHERE name='chunks_fts'").fetchone()
    mode = getattr(a, "mode", "hybrid")
    if mode == "hybrid" and fts and qterms:
        expression = " OR ".join('"' + t + '"' for t in qterms)
        sql = (
            "SELECT c.id FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.rowid WHERE chunks_fts MATCH ?"
            + where
            + " ORDER BY bm25(chunks_fts) LIMIT ?"
        )
        found = con.execute(sql, [expression, *params, max(64, a.k)]).fetchall()
        lex_rank = {r[0]: j + 1 for j, r in enumerate(found)}
    candidates = set(ranked[: max(64, a.k)]) | set(lex_rank)
    fused = {
        i: 0.6 / (60 + sem_rank[i]) + (0.4 / (60 + lex_rank[i]) if i in lex_rank else 0)
        for i in candidates
    }
    fusion_order = sorted(fused, key=lambda i: (-fused[i], i))
    fusion_rank = {i: r + 1 for r, i in enumerate(fusion_order)}
    reasons = {}
    top = ranked[: a.k]
    if mode == "hybrid" and lex_rank:
        top = []
        # Reserve a third of the slots for strong lexical evidence. Otherwise
        # RRF's two-route agreement can bury even BM25's best exact-name hit.
        needed = min(2, len(qterms))
        lex_slots = max(1, a.k // 3)
        texts = dict(
            con.execute(
                "SELECT id,text FROM chunks WHERE id IN (" + ",".join("?" * len(lex_rank)) + ")",
                list(lex_rank),
            )
        )
        for i in lex_rank:
            if len(set(qterms) & set(terms(texts[i]))) >= needed:
                top.append(i)
                reasons[i] = "lexical_guard"
                if len(top) >= lex_slots:
                    break
        if a.k > 1 and ranked[0] not in top:
            top.append(ranked[0])
            reasons[ranked[0]] = "semantic_guard"
        for i in fusion_order:
            if len(top) >= a.k:
                break
            if i not in top:
                top.append(i)
                reasons[i] = "fusion"
        top.sort(key=lambda i: (-fused[i], i))
    columns = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    tail = "parent,start,end" if "parent" in columns else "NULL,NULL,NULL"
    fields = "id,corpus,source,ts,role,text,key," + tail
    rows = {
        r[0]: r
        for r in con.execute(
            "SELECT " + fields + " FROM chunks WHERE id IN (" + ",".join("?" * len(top)) + ")", top
        )
    }
    out = []
    limit = getattr(a, "chars", 4000)
    context_cache = {}
    for i in top:
        r = rows[i]
        matches = sorted(set(qterms) & set(terms(r[5])))
        text, lo, hi, group = search_context(con, r, fields, neighbors, context_cache)
        body, truncated = bounded_context(text, lo, hi, limit)
        out.append(
            dict(
                id=r[6],
                chunk_id=i,
                score=float(scores[i - 1]),
                rank_score=fused.get(i, 0),
                semantic_rank=sem_rank.get(i),
                lexical_rank=lex_rank.get(i),
                matched_terms=matches,
                retrieval_mode=mode,
                corpus=r[1],
                source=r[2],
                ts=r[3],
                role=r[4],
                parent=r[7],
                offset_start=r[8],
                offset_end=r[9],
                text=body,
                selection_reason=reasons.get(i, "semantic"),
                fusion_rank=fusion_rank.get(i),
                context_window_ids=[x[6] for x in group],
                context_records=[
                    dict(
                        id=x[6], parent=x[7], ts=x[3], role=x[4], offset_start=x[8], offset_end=x[9]
                    )
                    for x in group
                ],
                neighbors=neighbors,
                anchor_text_chars=len(r[5]),
                text_chars=len(text),
                truncated=truncated,
                query_compacted=trimmed,
            )
        )
    con.close()
    if a.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return
    for hit in out:
        lexical = " +关键词" if hit["lexical_rank"] else ""
        print(
            f"{hit['score']:+.3f}{lexical} {hit['corpus']} {hit['ts']} {hit['source']} id={hit['id']}"
        )
        print("  " + hit["text"].replace("\n", " ")[: a.width])


@locked()
def cmd_read(a):
    if not 0 <= a.neighbors <= 8:
        sys.exit("--neighbors 范围为 0..8")
    if a.chars < 0:
        sys.exit("--chars 必须 >= 0")
    con = readonly()
    columns = {r[1] for r in con.execute("PRAGMA table_info(chunks)")}
    tail = "parent,start,end" if "parent" in columns else "NULL,NULL,NULL"
    fields = "id,corpus,source,ts,role,text,key," + tail
    row = con.execute("SELECT " + fields + " FROM chunks WHERE key=?", (str(a.id),)).fetchone()
    if row is None and str(a.id).isdigit():
        row = con.execute("SELECT " + fields + " FROM chunks WHERE id=?", (str(a.id),)).fetchone()
    if row is None:
        sys.exit("未找到窗口；索引可能已更新，请重新检索。")
    group = [row]
    if row[7]:
        # 同一个文件可能既属于包含它的目录语料、又属于它自己的语料，parent 会重复；
        # 兄弟窗口必须限定在同一个语料里，否则会把两份副本拼进同一段上下文。
        siblings = con.execute(
            "SELECT " + fields + " FROM chunks WHERE parent=? AND corpus=? ORDER BY start,id",
            (row[7], row[1]),
        ).fetchall()
        position = next(j for j, r in enumerate(siblings) if r[0] == row[0])
        group = siblings[max(0, position - a.neighbors) : position + a.neighbors + 1]
    text = ""
    end = group[0][8] or 0
    for r in group:
        start = r[8] or 0
        text += r[5][max(0, end - start) :]
        end = r[9] if r[9] is not None else end + len(r[5])
    body = text if not a.chars else text[: a.chars]
    result = dict(
        id=row[6],
        corpus=row[1],
        source=row[2],
        ts=row[3],
        role=row[4],
        window_ids=[r[6] for r in group],
        text=body,
        text_chars=len(text),
        truncated=len(body) < len(text),
    )
    con.close()
    print(json.dumps(result, ensure_ascii=False, indent=2) if a.json else body)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="OpenVINO device: CPU / GPU / NPU (RECALL_DEVICE, default CPU)",
    )
    ap.add_argument("--batch", type=int, default=8)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("index")
    p.add_argument("what", choices=["sessions", "dir", *sources.AGENTS])
    p.add_argument("path", nargs="?", default=".")
    p.add_argument(
        "--ext",
        default=".py,.md,.kdl,.sh,.js,.ts,.tsx,.jsx,.json,.toml,"
        ".yaml,.yml,.c,.h,.cpp,.rs,.go,.qml,.conf,.css,.html",
    )
    p.add_argument(
        "--force", action="store_true", help="清除本次语料的旧记录后重建（不再重复追加）"
    )
    p.add_argument(
        "--include-vendor", action="store_true", help="连第三方大目录（v821-tina/bui/docs）一起索引"
    )
    p.add_argument("--exclude", default="", help="额外排除的目录名，逗号分隔")
    p.set_defaults(fn=cmd_index)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=6)
    p.add_argument(
        "--corpus", help="Corpus prefixes, e.g. sessions or dir:project (comma separated)"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--width", type=int, default=260)
    p.add_argument("--mode", choices=["hybrid", "semantic"], default="hybrid")
    p.add_argument("--chars", type=int, default=4000, help="每条返回字符上限，0=完整窗口")
    p.add_argument(
        "--neighbors", type=int, default=1, help="搜索自动补充前后窗口，0=只返回命中窗口，范围 0..8"
    )
    p.set_defaults(fn=cmd_search)

    sub.add_parser("stats").set_defaults(fn=cmd_stats)
    sub.add_parser("rechunk", help="备份并迁移全部旧片段为 token 窗口").set_defaults(fn=cmd_rechunk)
    p = sub.add_parser("read")
    p.add_argument("id", help="搜索返回的稳定窗口 id，或当前数字行号")
    p.add_argument("--neighbors", type=int, default=2)
    p.add_argument("--chars", type=int, default=12000)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_read)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
