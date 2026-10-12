"""后端选择：平台默认、依赖回退、显式指定的边界。"""

import subprocess
import sys
import types

import pytest

from recall import backend


@pytest.fixture
def fake_install(monkeypatch):
    """用集合模拟「装了哪些后端」，避免依赖测试机上真实装了什么。"""
    installed: set[str] = set()
    monkeypatch.setattr(backend, "runtime_available", installed.__contains__)
    return installed


def _pretend_platform(monkeypatch, plat, machine):
    monkeypatch.setattr(backend, "sys", types.SimpleNamespace(platform=plat))
    monkeypatch.setattr(backend, "platform", types.SimpleNamespace(machine=lambda: machine))


def test_apple_silicon_defaults_to_mlx(monkeypatch):
    _pretend_platform(monkeypatch, "darwin", "arm64")
    assert backend.is_apple_silicon()
    assert backend.platform_default() == "mlx"


@pytest.mark.parametrize(
    "plat,machine",
    [
        ("darwin", "x86_64"),  # Intel Mac
        ("linux", "x86_64"),
        ("linux", "aarch64"),  # ARM Linux 没有 MLX，仍然走 OpenVINO
        ("win32", "AMD64"),
    ],
)
def test_everything_else_defaults_to_openvino(monkeypatch, plat, machine):
    _pretend_platform(monkeypatch, plat, machine)
    assert not backend.is_apple_silicon()
    assert backend.platform_default() == "openvino"


def test_auto_takes_the_platform_default(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "auto")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)
    fake_install.update({"mlx", "openvino"})  # 两个都装了，也要按平台选
    assert backend.active_backend() == "mlx"

    monkeypatch.setattr(backend, "is_apple_silicon", lambda: False)
    assert backend.active_backend() == "openvino"


def test_auto_falls_back_when_the_platform_default_is_absent(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "auto")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)
    fake_install.add("openvino")  # MLX 没装
    assert backend.active_backend() == "openvino"

    fake_install.clear()
    fake_install.add("mlx")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: False)
    assert backend.active_backend() == "mlx"


def test_auto_without_any_backend_says_how_to_fix(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "auto")
    with pytest.raises(RuntimeError, match="没有可用的嵌入后端") as excinfo:
        backend.active_backend()
    assert ".[mlx]" in str(excinfo.value) and ".[openvino]" in str(excinfo.value)


def test_explicit_backend_requires_its_own_dependency(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "mlx")
    with pytest.raises(RuntimeError, match="没装 MLX"):
        backend.active_backend()
    fake_install.add("mlx")
    assert backend.active_backend() == "mlx"


def test_explicit_openvino_is_never_silently_replaced(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "openvino")
    fake_install.add("mlx")  # 另一个后端可用也不能顶替
    with pytest.raises(RuntimeError, match="没装 OpenVINO"):
        backend.active_backend()


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(backend, "BACKEND", "bogus")
    with pytest.raises(ValueError, match="RECALL_BACKEND"):
        backend.active_backend()


def test_model_dir_follows_the_backend(monkeypatch):
    monkeypatch.setattr(backend, "active_backend", lambda: "mlx")
    assert backend.active_model_dir() == backend.MLX_MODEL_DIR
    monkeypatch.setattr(backend, "active_backend", lambda: "openvino")
    assert backend.active_model_dir() == backend.MODEL_DIR


def test_each_backend_resolves_to_its_own_embedder(monkeypatch):
    """平台选对了，还要真的接到对应的实现上。"""
    from recall.embedding import Embedder
    from recall.mlx_embedding import MLXEmbedder

    monkeypatch.setattr(backend, "active_backend", lambda: "openvino")
    assert backend.embedder_class() is Embedder
    monkeypatch.setattr(backend, "active_backend", lambda: "mlx")
    assert backend.embedder_class() is MLXEmbedder
    # 两个实现必须暴露同一套接口，否则 engine/daemon 就得区分后端了。
    for name in ("encode", "encode_query", "_get_compiled"):
        assert callable(getattr(Embedder, name, None)), name
        assert callable(getattr(MLXEmbedder, name, None)), name


def test_daemon_import_survives_having_no_backend_at_all():
    """两个后端都没装时，导入和 stats/--help 仍要能用，只有真正嵌入才报错。"""
    code = "\n".join(
        [
            "from recall import backend",
            "backend.runtime_available = lambda name: False",
            "import recall.daemon as d",
            "assert d.MODEL_DIR",
            "try:",
            "    d.Embedder()",
            "except RuntimeError as e:",
            "    assert '没有可用的嵌入后端' in str(e), e",
            "else:",
            "    raise AssertionError('expected RuntimeError')",
        ]
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


# ── 轻量客户端：只有 numpy + tokenizers，把推理交给常驻服务 ──────────────


def test_metadata_resolution_never_needs_a_local_runtime(monkeypatch, fake_install):
    """配置/元数据解析不能要求本地装了推理框架。"""
    monkeypatch.setattr(backend, "BACKEND", "auto")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)
    assert fake_install == set()  # 一个后端都没装

    assert backend.configured_backend() == "mlx"
    assert backend.candidate_model_dirs() == (backend.MLX_MODEL_DIR, backend.MODEL_DIR)
    assert backend.model_dir_for("openvino") == backend.MODEL_DIR

    with pytest.raises(RuntimeError, match="没有可用的嵌入后端"):
        backend.active_backend()  # 只有真要跑本地推理时才报错


def test_candidate_dirs_are_scoped_when_a_backend_is_forced(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "openvino")
    assert backend.candidate_model_dirs() == (backend.MODEL_DIR,)
    monkeypatch.setattr(backend, "BACKEND", "mlx")
    assert backend.candidate_model_dirs() == (backend.MLX_MODEL_DIR,)


def test_lightweight_client_reuses_a_resident_daemon(monkeypatch, tmp_path, fake_install):
    """另一套环境跑常驻服务，只装基础依赖的客户端要能分块并复用它。"""
    from recall import daemon, support

    monkeypatch.setattr(backend, "BACKEND", "auto")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)
    model = tmp_path / "model"
    model.mkdir()
    (model / "tokenizer.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(backend, "MLX_MODEL_DIR", model)

    assert fake_install == set()  # 客户端本地没有任何推理框架
    assert support._tokenizer_path() == model / "tokenizer.json"

    client = daemon.EmbClient(str(tmp_path / "embed.sock"))
    client.info = {
        "ok": True,
        "backend": "mlx",
        "devices": ["CPU"],
        "model_dir": str(model.resolve()),
    }
    assert client.supports("CPU")
    assert not client.supports("GPU")  # 设备仍然要比
    client.info["model_dir"] = str((tmp_path / "other").resolve())
    assert not client.supports("CPU")  # 换模型目录就不再复用


def test_whoami_reports_metadata_without_a_local_runtime(monkeypatch, tmp_path, fake_install):
    """CI 只装 .[dev]：whoami 不能因为本地没有推理框架就抛错。"""
    from recall import daemon

    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)

    class Fake:
        model_dir = tmp_path / "model"

    payload = daemon.whoami_payload("CPU", Fake())
    assert payload["ok"] and payload["backend"] == daemon.BACKEND_NAME
    assert payload["dim"] == 512 and payload["devices"] == ["CPU"]


# ── GGUF 后端 ────────────────────────────────────────────────────────


def test_gguf_is_opt_in_and_never_a_platform_default(monkeypatch, fake_install):
    """GGUF 通常是量化权重，向量空间和 fp32 不同，不能进 auto。"""
    fake_install.update({"mlx", "openvino", "gguf"})  # 连 llama.cpp 都装了
    monkeypatch.setattr(backend, "BACKEND", "auto")
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: True)
    assert backend.active_backend() == "mlx"
    monkeypatch.setattr(backend, "is_apple_silicon", lambda: False)
    assert backend.active_backend() == "openvino"
    assert "gguf" not in [p.name for p in backend.candidate_model_dirs()]


def test_gguf_is_selected_only_when_asked(monkeypatch, fake_install):
    monkeypatch.setattr(backend, "BACKEND", "gguf")
    assert backend.configured_backend() == "gguf"
    assert backend.candidate_model_dirs() == (backend.GGUF_MODEL_DIR,)
    assert backend.model_dir_for("gguf") == backend.GGUF_MODEL_DIR
    with pytest.raises(RuntimeError, match="没装 GGUF"):
        backend.active_backend()
    fake_install.add("gguf")
    assert backend.active_backend() == "gguf"


def test_gguf_resolves_to_its_own_embedder(monkeypatch):
    from recall.gguf_embedding import GGUFEmbedder

    monkeypatch.setattr(backend, "active_backend", lambda: "gguf")
    assert backend.embedder_class() is GGUFEmbedder
    for name in ("encode", "encode_query", "_get_compiled"):
        assert callable(getattr(GGUFEmbedder, name, None)), name


def test_gguf_weights_must_be_unambiguous(tmp_path):
    """多个量化档位放在一起会猜错，宁可要求用户明确。"""
    from recall.gguf_embedding import find_weights

    with pytest.raises(RuntimeError, match="没有 .gguf"):
        find_weights(tmp_path)
    (tmp_path / "a.gguf").write_bytes(b"a")
    assert find_weights(tmp_path).name == "a.gguf"
    (tmp_path / "b.gguf").write_bytes(b"b")
    with pytest.raises(RuntimeError, match="多个 .gguf"):
        find_weights(tmp_path)


def test_gguf_identity_uses_the_gguf_artifact(tmp_path):
    """GGUF 目录没有 manifest 时，身份退回 .gguf 自身摘要。"""
    from recall.identity import model_identity

    (tmp_path / "m.gguf").write_bytes(b"weights")
    first = model_identity(tmp_path)
    assert first and first.startswith("bge-cls-l2-v1:")
    (tmp_path / "m.gguf").write_bytes(b"other-weights")
    assert model_identity(tmp_path) != first


def _stub_reader(architecture=b"bert", pooling=2, dimension=512):
    class Part:
        def __init__(self, value):
            self.value = value

        def tolist(self):
            return self.value

        def __bytes__(self):
            return bytes(self.value)

    class Field:
        def __init__(self, value, kind="UINT32"):
            self.parts = [Part(value)]
            self.data = [0]
            self.types = [types.SimpleNamespace(name=kind)]

    fields = {
        "general.architecture": Field(list(architecture), "STRING"),
        "bert.pooling_type": Field([pooling]),
        "bert.embedding_length": Field([dimension]),
    }

    class Reader:
        def __init__(self, path):
            self.fields = fields

    return Reader


def _write_gguf_dir(tmp_path):
    from tokenizers import Tokenizer, models

    (tmp_path / "m.gguf").write_bytes(b"x")
    Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(
        str(tmp_path / "tokenizer.json")
    )


def test_scalar_gguf_metadata_fields_are_unwrapped():
    """gguf 里标量也是单元素数组；曾经因此把 512 错判成维度不符。"""
    from recall import setup

    reader = types.SimpleNamespace(
        fields={"bert.embedding_length": _stub_reader()(None).fields["bert.embedding_length"]}
    )
    assert setup._gguf_field(reader, "bert.embedding_length") == 512
    assert setup._gguf_field(reader, "missing") is None
    assert setup._gguf_field(_stub_reader()(None), "general.architecture") == "bert"


def test_validate_gguf_checks_architecture_pooling_and_dimension(monkeypatch, tmp_path):
    from recall import setup

    _write_gguf_dir(tmp_path)
    monkeypatch.setitem(
        sys.modules, "gguf", types.SimpleNamespace(GGUFReader=_stub_reader(pooling=2))
    )
    setup.validate_gguf(tmp_path, "m.gguf")  # CLS + 512：通过

    monkeypatch.setitem(
        sys.modules, "gguf", types.SimpleNamespace(GGUFReader=_stub_reader(pooling=1))
    )
    with pytest.raises(ValueError, match="不是 CLS"):
        setup.validate_gguf(tmp_path, "m.gguf")

    monkeypatch.setitem(
        sys.modules,
        "gguf",
        types.SimpleNamespace(GGUFReader=_stub_reader(architecture=b"llama")),
    )
    with pytest.raises(ValueError, match="只支持 BERT"):
        setup.validate_gguf(tmp_path, "m.gguf")

    monkeypatch.setitem(
        sys.modules,
        "gguf",
        types.SimpleNamespace(GGUFReader=_stub_reader(dimension=768)),
    )
    with pytest.raises(ValueError, match="需要 512"):
        setup.validate_gguf(tmp_path, "m.gguf")


def test_validate_gguf_reports_missing_files_clearly(tmp_path):
    from recall import setup

    with pytest.raises(ValueError, match="m.gguf"):
        setup.validate_gguf(tmp_path, "m.gguf")
    (tmp_path / "m.gguf").write_bytes(b"x")
    # 缺分词器时报清楚的错，而不是底层 from_file 的裸异常
    with pytest.raises(ValueError, match="tokenizer.json"):
        setup.validate_gguf(tmp_path, "m.gguf")
