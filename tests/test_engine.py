import contextlib, hashlib, io, json, pathlib, sqlite3, subprocess, sys, tempfile, types, unittest

ROOT = pathlib.Path(__file__).resolve().parent
from recall import engine as e
from recall import support
from helpers import install_fake


def run(fn, args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(args)
    return out.getvalue()


def index_args(force=False):
    return types.SimpleNamespace(what="sessions", force=force, device="GPU", batch=8)


def search(query, k=6, mode="hybrid"):
    return json.loads(
        run(
            e.cmd_search,
            types.SimpleNamespace(
                query=query,
                k=k,
                mode=mode,
                device="GPU",
                corpus="sessions",
                json=True,
                width=220,
                chars=0,
            ),
        )
    )


def hashes():
    return [hashlib.sha256(p.read_bytes()).hexdigest() for p in [e.DB, e.VEC]]


def message(text, ts="2026-01-01T00:00:00"):
    return (
        json.dumps(
            {
                "type": "message",
                "timestamp": ts,
                "message": {"role": "user", "content": [{"type": "text", "text": text}]},
            }
        )
        + "\n"
    )


class RecallTests(unittest.TestCase):
    def test_strong_keyword_hit_survives_low_semantic_rank(self):
        from unittest import mock
        import numpy as np

        con = e.db_connect()
        vectors = []
        for i in range(1, 101):
            text = (
                "Orion engineer dismissed the coachman."
                if i == 100
                else f"Orion engineer described unrelated scenery number {i}."
            )
            score = 0.2 if i == 100 else 0.9 - i * 0.001
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = score
            vector[1] = (1 - score * score) ** 0.5
            vectors.append(vector)
            con.execute(
                "INSERT INTO chunks(id,corpus,source,ts,role,text,key,parent,start,end) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    i,
                    "sessions",
                    "fixture.jsonl",
                    "2026-01-01T00:00:00",
                    "user",
                    text,
                    str(i) + "key",
                    str(i) + "parent",
                    0,
                    len(text),
                ),
            )
            con.execute(
                "INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (i, support.fts_text(text))
            )
        con.commit()
        con.close()
        e.write_vectors(np.array(vectors))

        class Query:
            def encode_query(self, text):
                v = np.zeros((1, 512), dtype=np.float32)
                v[0, 0] = 1
                return v

        with mock.patch.object(e, "resolve_embedder", return_value=Query()):
            hits = search("Orion dismissed coachman", k=6)
            winner = next(h for h in hits if h["chunk_id"] == 100)
            self.assertEqual(winner["lexical_rank"], 1)
            self.assertEqual(winner["semantic_rank"], 100)
            self.assertEqual(winner["selection_reason"], "lexical_guard")
            self.assertEqual(len(hits), 6)
            self.assertEqual(len({h["id"] for h in hits}), 6)
            long_query = (
                " ".join(f"unknownword{n}" for n in range(180)) + " Orion dismissed coachman"
            )
            longer = search(long_query, k=6)
            late = next(h for h in longer if h["chunk_id"] == 100)
            self.assertEqual(late["lexical_rank"], 1)
            self.assertIn("coachman", late["matched_terms"])
            semantic = search("Orion dismissed coachman", k=6, mode="semantic")
            self.assertNotIn(100, [h["chunk_id"] for h in semantic])

    def test_neighbors_cross_session_record_boundary_but_not_source(self):
        con = e.db_connect()
        fixtures = [
            ("fixture.jsonl", "2026-01-01T00:00:01", "user", "first", "The first event."),
            ("fixture.jsonl", "2026-01-01T00:00:02", "assistant", "second", "The next event."),
            ("other.jsonl", "2026-01-01T00:00:01", "user", "other", "UNRELATED_SOURCE_SECRET"),
        ]
        for i, (source, ts, role, parent, text) in enumerate(fixtures, 1):
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)",
                (i, "sessions", source, ts, role, text, str(i) + "key", parent, 0, len(text)),
            )
        con.commit()
        fields = "id,corpus,source,ts,role,text,key,parent,start,end"
        row = con.execute("SELECT " + fields + " FROM chunks WHERE id=2").fetchone()
        text, lo, hi, group = e.search_context(con, row, fields, 1, {})
        self.assertEqual(text[lo:hi], "The next event.")
        self.assertLess(text.index("The first event."), text.index("The next event."))
        self.assertIn("assistant", text)
        self.assertNotIn("UNRELATED_SOURCE_SECRET", text)
        self.assertEqual([r[0] for r in group], [1, 2])
        raw, lo, hi, group = e.search_context(con, row, fields, 0, {})
        self.assertEqual(raw, "The next event.")
        self.assertEqual(len(group), 1)
        con.close()

    def test_context_truncation_keeps_the_anchor_and_removes_overlap(self):
        text = "PREVIOUS " * 900 + "ANCHOR_PHRASE" + " NEXT" * 900
        lo = text.index("ANCHOR_PHRASE")
        hi = lo + len("ANCHOR_PHRASE")
        body, truncated = e.bounded_context(text, lo, hi, 100)
        self.assertTrue(truncated)
        self.assertEqual(len(body), 100)
        self.assertIn("ANCHOR_PHRASE", body)
        con = e.db_connect()
        raw = "abcdefghijklmno"
        for i, (lo, hi) in enumerate([(0, 10), (7, 15)], 1):
            con.execute(
                "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)",
                (i, "dir:Dev", "/doc", "", "file", raw[lo:hi], str(i) + "key", "parent", lo, hi),
            )
        con.commit()
        fields = "id,corpus,source,ts,role,text,key,parent,start,end"
        row = con.execute("SELECT " + fields + " FROM chunks WHERE id=2").fetchone()
        text, lo, hi, group = e.search_context(con, row, fields, 1, {})
        self.assertEqual(text, raw)
        self.assertEqual(text[lo:hi], raw[7:15])
        con.close()

    def setUp(self):
        self.restore = install_fake()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        e.INDEX = self.root / "index"
        e.DB = e.INDEX / "meta.db"
        e.VEC = e.INDEX / "vectors.npy"
        e.SESSIONS = self.root / "sessions"
        e.SESSIONS.mkdir()
        # `index sessions` 会遍历所有 agent；其余三个指向不存在的位置，避免测试
        # 去索引真实的 Codex/opencode/DSH 历史（那会非常慢，还会读用户数据）。
        e.DSH_SESSIONS = self.root / "dsh"
        e.CODEX_SESSIONS = self.root / "codex"
        e.OPENCODE_PATH = self.root / "opencode.db"
        e.CODEBUDDY_SESSIONS = self.root / "codebuddy"
        e.WORKBUDDY_SESSIONS = self.root / "workbuddy"
        e.CLAUDE_SESSIONS = self.root / "claude"

    def tearDown(self):
        self.tmp.cleanup()
        self.restore()

    def write(self, text):
        path = e.SESSIONS / "fixture.jsonl"
        path.write_text(message(text))
        return path

    def check(self):
        con = sqlite3.connect(e.DB)
        ids = [r[0] for r in con.execute("SELECT id FROM chunks ORDER BY id")]
        self.assertEqual(ids, list(range(1, len(ids) + 1)))
        self.assertEqual(e.load_vectors().shape, (len(ids), 512))
        self.assertEqual(con.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0], len(ids))
        self.assertEqual(
            con.execute("SELECT COUNT(*) FROM chunks WHERE parent IS NULL").fetchone()[0], 0
        )
        con.close()
        return len(ids)

    def test_windows_and_long_query(self):
        text = ("Long code alpha = 43; 中文混合，检查参数。\n" * 500) + "TRAILING_UNIQUE_NEEDLE"
        chunks = list(support.token_windows(text))
        self.assertGreater(len(chunks), 5)
        covered = set()
        for fragment, start, end in chunks:
            self.assertEqual(fragment, text[start:end])
            self.assertLessEqual(len(support.tokenizer().encode(fragment).ids), 512)
            covered.update(range(start, end))
        self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))
        compact, trimmed = support.compact_query(text)
        self.assertTrue(trimmed)
        self.assertIn("TRAILING_UNIQUE_NEEDLE", compact)
        self.assertLessEqual(len(support.tokenizer().encode(compact).ids), 512)

    def test_ignored_boundary_characters_are_preserved(self):
        for text in ["\u0000" + ("alpha beta 中文 " * 400) + "\u200b🎉", "\u200b" * 30]:
            chunks = list(support.token_windows(text))
            self.assertTrue(chunks)
            merged = ""
            end = 0
            for fragment, lo, hi in chunks:
                merged += fragment[max(0, end - lo) :]
                end = hi
                self.assertEqual(fragment, text[lo:hi])
                self.assertLessEqual(len(support.tokenizer().encode(fragment).ids), 512)
            self.assertEqual(merged, text)

    def test_import_does_not_load_openvino(self):
        code = "import sys;from recall import engine;assert 'openvino' not in sys.modules"
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_tail_retrieval_read_incremental_and_force(self):
        text = (
            "The generic log describes old routine data.\n" * 700
        ) + "\nThe Xyrolith device uses protocol Novara9 at address 0x12.\n"
        path = self.write(text)
        run(e.cmd_index, index_args())
        count = self.check()
        before = hashes()
        run(e.cmd_index, index_args())
        self.assertEqual(before, hashes())
        hits = search("Xyrolith Novara9 protocol")
        self.assertTrue(any("Novara9" in h["text"] for h in hits))
        self.assertTrue(all(not h["truncated"] for h in hits))
        h = next(h for h in hits if "Novara9" in h["text"])
        read = json.loads(
            run(e.cmd_read, types.SimpleNamespace(id=h["id"], neighbors=2, chars=0, json=True))
        )
        self.assertIn("Novara9", read["text"])
        with path.open("a") as f:
            f.write(
                message(
                    "A second independent remembered decision: use the Zirconia firmware.",
                    ts="2026-01-01T00:00:01",
                )
            )
        run(e.cmd_index, index_args())
        self.assertGreater(self.check(), count)
        after = self.check()
        run(e.cmd_index, index_args(force=True))
        self.assertEqual(after, self.check())
        self.assertTrue(any("Zirconia" in h["text"] for h in search("Zirconia firmware")))

    def test_numeric_stable_key_is_not_confused_with_row_id(self):
        self.write("Remember the numeric window identifier edge case.")
        run(e.cmd_index, index_args())
        con = sqlite3.connect(e.DB)
        con.execute("UPDATE chunks SET key=? WHERE id=1", ("1" * 24,))
        con.commit()
        con.close()
        value = json.loads(
            run(e.cmd_read, types.SimpleNamespace(id="1" * 24, neighbors=0, chars=0, json=True))
        )
        self.assertEqual(value["id"], "1" * 24)
        self.assertIn("numeric window identifier", value["text"])
        numeric = json.loads(
            run(e.cmd_read, types.SimpleNamespace(id="1", neighbors=0, chars=0, json=True))
        )
        self.assertEqual(value, numeric)

    def test_failure_does_not_publish_partial_update(self):
        path = self.write("An original remembered fact about valid indexing.")
        run(e.cmd_index, index_args())
        before = hashes()
        with path.open("a") as f:
            f.write(
                message(
                    "A new fact that must survive a failed attempted indexing.",
                    ts="2026-01-01T00:00:01",
                )
            )
        resolver = e.resolve_embedder

        class Broken:
            def encode(self, texts):
                raise RuntimeError("Injected embedding failure")

        e.resolve_embedder = lambda *a, **k: Broken()
        try:
            with self.assertRaises(RuntimeError):
                run(e.cmd_index, index_args())
        finally:
            e.resolve_embedder = resolver
        self.assertEqual(before, hashes())
        self.check()
        run(e.cmd_index, index_args())
        self.assertNotEqual(before, hashes())
        self.check()

    def test_legacy_migration_dedup_and_idempotence(self):
        self.write(
            ("Original history before long-message migration.\n" * 700) + "MigrationTailFact"
        )
        con = e.db_connect()
        raw = list(e.iter_session_chunks(e.SESSIONS / "fixture.jsonl"))[0]
        role, ts, text = raw
        source = "fixture.jsonl"
        con.execute(
            "INSERT INTO chunks(id,corpus,source,ts,role,text,key) VALUES(1,?,?,?,?,?,?)",
            ("sessions", source, ts, role, text, e.chunk_key(source, ts, role, text)),
        )
        con.execute("INSERT INTO chunks_fts(rowid,text) VALUES(1,?)", (support.fts_text(text),))
        con.commit()
        con.close()
        e.write_vectors(e.resolve_embedder("GPU", batch=8).encode([text]))
        run(e.cmd_rechunk, types.SimpleNamespace(device="GPU", batch=8))
        self.assertGreater(self.check(), 1)
        after = hashes()
        run(e.cmd_index, index_args())
        self.assertEqual(after, hashes())
        run(e.cmd_rechunk, types.SimpleNamespace(device="GPU", batch=8))
        self.assertEqual(after, hashes())
        self.assertTrue(any("MigrationTailFact" in h["text"] for h in search("MigrationTailFact")))

    def test_mixed_corpora_migrate_and_force_preserves_other_corpus(self):
        self.write(("Long historical context should become token windows. " * 400))
        run(e.cmd_index, index_args())
        con = e.db_connect()
        n = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        text = "Legacy directory text contains metadata for ZephyrChronicle. " * 300
        key = e.chunk_key("/fixture/doc.md", "", "file", text)
        con.execute(
            "INSERT INTO chunks(id,corpus,source,ts,role,text,key) VALUES(?,?,?,?,?,?,?)",
            (n + 1, "dir:Dev", "/fixture/doc.md", "", "file", text, key),
        )
        con.execute(
            "INSERT INTO chunks_fts(rowid,text) VALUES(?,?)", (n + 1, support.fts_text(text))
        )
        con.commit()
        con.close()
        e.append_vectors(e.resolve_embedder("GPU", batch=8).encode([text]))
        run(e.cmd_rechunk, types.SimpleNamespace(device="GPU", batch=8))
        self.check()
        con = sqlite3.connect(e.DB)
        count = con.execute("SELECT COUNT(*) FROM chunks WHERE corpus='dir:Dev'").fetchone()[0]
        con.close()
        self.assertGreater(count, 1)
        run(e.cmd_index, index_args(force=True))
        self.check()
        con = sqlite3.connect(e.DB)
        self.assertEqual(
            count, con.execute("SELECT COUNT(*) FROM chunks WHERE corpus='dir:Dev'").fetchone()[0]
        )
        con.close()
        hits = json.loads(
            run(
                e.cmd_search,
                types.SimpleNamespace(
                    query="ZephyrChronicle",
                    k=3,
                    mode="hybrid",
                    device="GPU",
                    corpus="dir:Dev",
                    json=True,
                    width=220,
                    chars=0,
                ),
            )
        )
        self.assertTrue(hits)
        self.assertTrue(
            all(h["corpus"] == "dir:Dev" and "ZephyrChronicle" in h["text"] for h in hits)
        )

    def test_invalid_vectors_do_not_replace_live_index(self):
        import numpy as np

        path = self.write("Initial valid remembered data for vector validation.")
        run(e.cmd_index, index_args())
        before = hashes()
        with path.open("a") as f:
            f.write(
                message(
                    "A new remembered item with an invalid embedding.", ts="2026-01-01T00:00:01"
                )
            )
        resolver = e.resolve_embedder

        class Broken:
            def encode(self, texts):
                return np.full((len(texts), 512), np.nan, dtype=np.float32)

        e.resolve_embedder = lambda *a, **k: Broken()
        try:
            with self.assertRaises(AssertionError):
                run(e.cmd_index, index_args())
        finally:
            e.resolve_embedder = resolver
        self.assertEqual(before, hashes())
        self.check()

    def test_interrupted_swap_recovery(self):
        self.write("A robust remembered fact about transaction recovery.")
        run(e.cmd_index, index_args())
        before = hashes()
        for name in ["meta.db", "vectors.npy"]:
            (e.INDEX / (".rollback-" + name)).write_bytes((e.INDEX / name).read_bytes())
        (e.INDEX / "transaction.json").write_text(
            json.dumps({"meta.db": True, "vectors.npy": True})
        )
        e.VEC.write_bytes(b"partially published invalid array")
        run(e.cmd_stats, types.SimpleNamespace())
        self.assertEqual(before, hashes())
        self.assertFalse((e.INDEX / "transaction.json").exists())

    def test_overlapping_directory_corpora_are_not_deduplicated_away(self):
        """先索引整个项目、再索引其中的子目录，子语料必须真的建起来。

        parent 去重键不含 corpus，所以全局去重会把重叠语料整段吞掉：命令成功退出，
        却一条都没进来，按子语料检索时报「没有内容」。
        """
        project = self.root / "project"
        docs = project / "docs"
        docs.mkdir(parents=True)
        (docs / "readme.md").write_text(
            "The Amber protocol reaches the gateway over a retained connection. " * 3
        )

        def index_dir(path, force=False):
            run(
                e.cmd_index,
                types.SimpleNamespace(
                    what="dir",
                    path=str(path),
                    force=force,
                    device="GPU",
                    batch=8,
                    ext=".md",
                    exclude="",
                    include_vendor=False,
                ),
            )

        index_dir(project)
        index_dir(docs, force=True)
        con = sqlite3.connect(e.DB)
        corpora = {r[0] for r in con.execute("SELECT corpus FROM chunks")}
        con.close()
        self.assertTrue(
            any(c.startswith("dir:docs:") for c in corpora), f"子目录语料没建起来: {corpora}"
        )
        self.assertTrue(any(c.startswith("dir:project:") for c in corpora), corpora)

    def test_nested_corpus_read_does_not_splice_the_other_copy(self):
        """同一文件同时属于两个语料时，read 的兄弟窗口必须限定在同一语料内。

        两份副本的 parent 与 start/end 完全相同，跨语料取兄弟会把整段正文拼两遍。
        """
        project = self.root / "project"
        docs = project / "docs"
        docs.mkdir(parents=True)
        # 需要多个窗口，否则没有「兄弟」可拼。
        body = "The Amber protocol reaches the gateway over a retained connection. " * 90
        (docs / "readme.md").write_text(body + "\nDISTINCT_TAIL_MARKER\n")

        def index_dir(path):
            run(
                e.cmd_index,
                types.SimpleNamespace(
                    what="dir",
                    path=str(path),
                    force=False,
                    device="GPU",
                    batch=8,
                    ext=".md",
                    exclude="",
                    include_vendor=False,
                ),
            )

        index_dir(project)
        index_dir(docs)
        hits = json.loads(
            run(
                e.cmd_search,
                types.SimpleNamespace(
                    query="Amber protocol",
                    k=3,
                    mode="semantic",
                    device="GPU",
                    corpus="dir:docs",
                    json=True,
                    width=220,
                    chars=0,
                    neighbors=0,
                ),
            )
        )
        self.assertTrue(hits)
        value = json.loads(
            run(
                e.cmd_read,
                types.SimpleNamespace(id=hits[0]["id"], neighbors=8, chars=0, json=True),
            )
        )
        self.assertEqual(value["text"].count("DISTINCT_TAIL_MARKER"), 1, value["text"][-400:])

    def test_index_refuses_a_different_embedding_model(self):
        """换一份权重会整体改变向量空间，必须拦下而不是静默给出错误结果。"""
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights-A")

        class ModelEmbedding:
            model_dir = model

            def encode(self, texts, is_query=False, normalize=True):
                import numpy as np

                out = np.zeros((len(texts), 512), dtype=np.float32)
                out[:, 0] = 1.0
                return out

            def encode_query(self, text, **kw):
                return self.encode([text], is_query=True)

        original = e.resolve_embedder
        e.resolve_embedder = lambda *a, **k: ModelEmbedding()
        try:
            self.write("Remember the Amber protocol for the gateway.")
            run(e.cmd_index, index_args())
            con = sqlite3.connect(e.DB)
            recorded = con.execute("SELECT value FROM meta WHERE key='embedding_id'").fetchone()
            con.close()
            self.assertIsNotNone(recorded, "索引应当记录嵌入模型指纹")

            (model / "model.safetensors").write_bytes(b"weights-B")
            with self.assertRaises(SystemExit) as caught:
                search("Amber protocol")
            self.assertIn("另一份嵌入模型", str(caught.exception))
        finally:
            e.resolve_embedder = original

    def test_journal_is_published_atomically(self):
        """发布日志必须先写完整再原子改名，不能留下半截 JSON 卡死整个索引。"""
        self.write("A robust remembered fact about journal atomicity.")
        run(e.cmd_index, index_args())
        before = hashes()
        stage = self.root / "stage"
        stage.mkdir()
        for name in ("meta.db", "vectors.npy"):
            (stage / name).write_bytes(b"new-pair")

        real_dump = support.json.dump

        def broken_dump(state, handle, *a, **k):
            handle.write('{"meta.db":')
            handle.flush()
            raise KeyboardInterrupt

        support.json.dump = broken_dump
        try:
            with self.assertRaises(KeyboardInterrupt):
                support.publish(stage, e.INDEX)
        finally:
            support.json.dump = real_dump

        self.assertFalse((e.INDEX / "transaction.json").exists(), "不能留下半截日志")
        self.assertFalse((e.INDEX / ".transaction.new").exists())
        self.assertEqual(before, hashes(), "两侧文件都应保持原样")
        run(e.cmd_stats, types.SimpleNamespace())  # 索引仍然可用

    def test_malformed_journal_reports_an_actionable_error(self):
        self.write("A robust remembered fact about damaged journals.")
        run(e.cmd_index, index_args())
        (e.INDEX / "transaction.json").write_text('{"meta.db":')
        with self.assertRaises(RuntimeError) as caught:
            run(e.cmd_stats, types.SimpleNamespace())
        self.assertIn("事务日志损坏", str(caught.exception))

    def test_concurrent_cli_readers_and_writer(self):
        path = self.write("Original concurrent lookup for the Amber protocol.")
        run(e.cmd_index, index_args())
        with path.open("a") as f:
            f.write(
                message(
                    ("Concurrent history entry for Amber updates. " * 400), ts="2026-01-01T00:00:01"
                )
            )
        script = "import sys,pathlib,types;sys.path.insert(0,sys.argv[1]);from recall import engine as e;from helpers import install_fake;install_fake();r=pathlib.Path(sys.argv[2]);e.INDEX=r/'index';e.DB=e.INDEX/'meta.db';e.VEC=e.INDEX/'vectors.npy';e.SESSIONS=r/'sessions';e.DSH_SESSIONS=r/'dsh';e.CODEX_SESSIONS=r/'codex';e.OPENCODE_PATH=r/'opencode.db';e.CODEBUDDY_SESSIONS=r/'codebuddy';e.WORKBUDDY_SESSIONS=r/'workbuddy';e.CLAUDE_SESSIONS=r/'claude';"
        writer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script
                + "e.cmd_index(types.SimpleNamespace(what='sessions',force=False,device='GPU',batch=8))",
                str(ROOT),
                str(self.root),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        readers = [
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script
                    + "e.cmd_search(types.SimpleNamespace(query='Amber protocol',k=3,device='GPU',corpus='sessions',json=True,width=220,chars=0))",
                    str(ROOT),
                    str(self.root),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(4)
        ]
        for child in readers:
            out, err = child.communicate(timeout=60)
            self.assertEqual(child.returncode, 0, err)
            self.assertTrue(json.loads(out))
        _, err = writer.communicate(timeout=60)
        self.assertEqual(writer.returncode, 0, err)
        self.check()


if __name__ == "__main__":
    unittest.main(verbosity=2)
