"""Token windows, local lexical search and consistent index publication."""

import contextlib, fcntl, functools, inspect, json, os, pathlib, re, shutil, tempfile, threading
from tokenizers import Tokenizer
from .config import MODEL_DIR

WINDOW_TOKENS = 448
OVERLAP_TOKENS = 64
_tokenizer = None
_local = threading.local()
STOP = set(
    "a an the is are was were be been being do does did what which who when where why how and or of in on at to for from with by as it its this that these those i you we they he she his her their my your our can could would should will may have has had based provided knowledge pool question answer context example output story text label only all now".split()
)
STOP.update(["什么", "怎么", "如何", "这个", "那个", "之前", "上次", "为什么"])


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        _tokenizer.no_truncation()
    return _tokenizer


def token_windows(text):
    enc = tokenizer().encode(text, add_special_tokens=False)
    if not enc.ids:
        if text:
            yield text, 0, len(text)
        return
    for start in range(0, len(enc.ids), WINDOW_TOKENS - OVERLAP_TOKENS):
        end = min(start + WINDOW_TOKENS, len(enc.ids))
        lo = 0 if start == 0 else enc.offsets[start][0]
        hi = len(text) if end == len(enc.ids) else enc.offsets[end - 1][1]
        yield text[lo:hi], lo, hi
        if end == len(enc.ids):
            break


def compact_query(text):
    enc = tokenizer().encode(text, add_special_tokens=False)
    if len(enc.ids) <= WINDOW_TOKENS:
        return text, False
    # Keep context plus the tail (e.g. the actual error/question after a long paste).
    return text[: enc.offsets[95][1]] + "\n...\n" + text[enc.offsets[-348][0] :], True


def terms(text):
    out = []
    for token in re.findall(r"[a-zA-Z0-9_]+|[\u3400-\u9fff]+", text.lower()):
        if re.fullmatch(r"[\u3400-\u9fff]+", token):
            out.extend(
                token[i : i + 2] for i in range(len(token) - 1) if token[i : i + 2] not in STOP
            )
        elif token not in STOP and (len(token) > 1 or token.isdigit()):
            out.append(token)
    return out


def query_terms(text):
    unique = list(dict.fromkeys(terms(text)))
    # A long question/paste can introduce its actual identifiers near the end.
    # Keep ordinary queries intact; bound large FTS expressions at both ends.
    return unique if len(unique) <= 128 else unique[:64] + unique[-64:]


def fts_text(text):
    return " ".join(terms(text))


def recover(index):
    journal = index / "transaction.json"
    if not journal.exists():
        return
    state = json.loads(journal.read_text())
    if set(state) != {"meta.db", "vectors.npy"} or any(type(v) is not bool for v in state.values()):
        raise RuntimeError("Invalid recall transaction journal; restore the saved backup.")
    for name in ["meta.db", "vectors.npy"]:
        target = index / name
        if state[name]:
            backup = index / (".rollback-" + name)
            temp = index / (".recover-" + name)
            shutil.copy2(backup, temp)
            os.replace(temp, target)
        else:
            target.unlink(missing_ok=True)
    journal.unlink()


def locked(exclusive=False):
    def decorate(fn):
        @functools.wraps(fn)
        def call(*args, **kwargs):
            if getattr(_local, "depth", 0):
                return fn(*args, **kwargs)
            g = inspect.unwrap(fn).__globals__
            index = g.get("_ACTIVE_INDEX") or g["INDEX"]
            index.parent.mkdir(parents=True, exist_ok=True)
            lock = index.parent / ("." + index.name + ".lock")
            with lock.open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
                if (index / "transaction.json").exists():
                    fcntl.flock(f, fcntl.LOCK_EX)
                    recover(index)
                    if not exclusive:
                        fcntl.flock(f, fcntl.LOCK_SH)
                _local.depth = 1
                try:
                    return fn(*args, **kwargs)
                finally:
                    _local.depth = 0

        return call

    return decorate


def publish(stage, index):
    index.mkdir(parents=True, exist_ok=True)
    state = {name: (index / name).exists() for name in ["meta.db", "vectors.npy"]}
    for name, exists in state.items():
        if exists:
            shutil.copy2(index / name, index / (".rollback-" + name))
    journal = index / "transaction.json"
    with journal.open("w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    try:
        for name in state:
            os.replace(stage / name, index / name)
        fd = os.open(index, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        journal.unlink()
    except BaseException:
        recover(index)
        raise


def transactional(fn):
    @functools.wraps(fn)
    def call(*args, **kwargs):
        g = fn.__globals__
        index = g["INDEX"]
        db = g["DB"]
        vec = g["VEC"]
        index.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".recall-stage-", dir=index.parent) as tmp:
            stage = pathlib.Path(tmp)
            for name in ["meta.db", "vectors.npy"]:
                if (index / name).exists():
                    shutil.copy2(index / name, stage / name)
            g["_ACTIVE_INDEX"] = index
            g["INDEX"] = stage
            g["DB"] = stage / "meta.db"
            g["VEC"] = stage / "vectors.npy"
            g["_INDEX_DIRTY"] = False
            try:
                result = fn(*args, **kwargs)
                if g["_INDEX_DIRTY"]:
                    if not g["VEC"].exists():
                        g["write_vectors"](g["np"].zeros((0, g["DIM"]), dtype=g["np"].float32))
                    con = g["sqlite3"].connect(g["DB"])
                    count = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                    ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
                    fts_count = con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
                    con.close()
                    matrix = g["load_vectors"]()
                    assert matrix.shape == (count, g["DIM"]) and ids == list(range(1, count + 1))
                    assert fts_count == count
                    assert g["np"].isfinite(matrix).all()
                    assert g["np"].allclose(g["np"].linalg.norm(matrix, axis=1), 1, atol=0.005)
                    publish(stage, index)
                return result
            finally:
                g["INDEX"] = index
                g["DB"] = db
                g["VEC"] = vec
                g["_ACTIVE_INDEX"] = None

    return call
