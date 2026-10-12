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
