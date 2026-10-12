"""嵌入后端选择：Apple Silicon 用 MLX，Intel / x86_64 平台用 OpenVINO，GGUF 可选。

一个 recall 项目、三套推理后端。前两套按平台自动挑：

| 平台 | 默认后端 | 依赖 |
| --- | --- | --- |
| Apple Silicon（darwin + arm64） | MLX | `pip install '.[mlx]'` |
| Linux / Intel macOS / 其它 | OpenVINO | `pip install '.[openvino]'` |
| 任意平台（需显式指定） | GGUF | `pip install '.[gguf]'` |

``RECALL_BACKEND`` 取 ``auto``（默认）/ ``openvino`` / ``mlx`` / ``gguf``：

- ``auto`` 先按平台默认挑；该后端的依赖没装时回退到另一个，两个都没有才报错。
  所以 Linux 上的既有行为不变，Apple Silicon 上不装 OpenVINO 也能直接跑。
- ``gguf`` **只按显式指定生效**，不参与 auto：它通常是量化权重，向量空间和 fp32
  不是同一个，不能悄悄替换掉用户已有的索引。
- 显式指定则不做回退，缺依赖时报错而不是静默换后端。

这里刻意把两件事分开：

- **配置/元数据**（``configured_backend`` / ``candidate_model_dirs``）只读环境与平台，
  **从不要求本地装了推理框架**。轻量客户端（只装 numpy + tokenizers）靠它决定
  tokenizer 路径和该接受哪个常驻服务的模型目录。
- **运行时可用性**（``active_backend``）只有真的要**在本地跑推理**时才检查。

``daemon`` 会把结果一次性解析到模块级的 ``Embedder`` / ``MODEL_DIR`` 两个名字上，
保持和上游一样的可 monkeypatch 形态。
"""

from __future__ import annotations

import importlib.util
import pathlib
import platform
import sys

from .config import BACKEND, GGUF_MODEL_DIR, MLX_MODEL_DIR, MODEL_DIR

#: 后端名 → 需要能 import 的模块名（发行包名和 import 名并不一致）。
MODULES = {"openvino": "openvino", "mlx": "mlx", "gguf": "llama_cpp"}

#: 后端名 → 给人看的名字。
DISPLAY = {"openvino": "OpenVINO", "mlx": "MLX", "gguf": "GGUF (llama.cpp)"}

INSTALL_HINT = {
    "openvino": "pip install '.[openvino]'",
    "mlx": "pip install '.[mlx]'",
    "gguf": "pip install '.[gguf]'（llama-cpp-python 没有预编译轮子，需要 C++ 工具链与 cmake）",
}

#: ``gguf`` 只能显式选择：它通常是**量化**权重，向量空间和 fp32 权重不是同一个，
#: 不该被 auto 悄悄选中。
PLATFORM_BACKENDS = ("openvino", "mlx")
ALIASES = {"ov": "openvino", "openvino": "openvino", "mlx": "mlx", "gguf": "gguf"}

NO_BACKEND_MESSAGE = (
    "没有可用的嵌入后端：Apple Silicon 上装 MLX（pip install '.[mlx]'），"
    "其它平台装 OpenVINO（pip install '.[openvino]'）。"
)


def runtime_available(backend: str) -> bool:
    try:
        return importlib.util.find_spec(MODULES[backend]) is not None
    except (ImportError, ValueError):
        return False


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine().lower() in ("arm64", "aarch64")


def platform_default() -> str:
    """平台默认后端：Apple Silicon 走 MLX（Metal），其余走 OpenVINO。"""
    return "mlx" if is_apple_silicon() else "openvino"


def configured_backend() -> str:
    """当前配置指向的后端，**不检查运行时是否安装**。

    只用于「该用哪个模型目录 / 该接受哪个常驻服务」这类元数据判断，所以在只有
    numpy + tokenizers 的客户端环境里也能安全调用。未知取值的 ``RECALL_BACKEND``
    是配置错误，仍然报错。
    """
    want = (BACKEND or "auto").strip().lower()
    if want in ("", "auto"):
        return platform_default()
    if want in ALIASES:
        return ALIASES[want]
    raise ValueError(f"未知的 RECALL_BACKEND={BACKEND!r}；可选 auto / openvino / mlx")


def model_dir_for(backend: str) -> pathlib.Path:
    if backend == "mlx":
        return MLX_MODEL_DIR
    if backend == "gguf":
        return GGUF_MODEL_DIR
    return MODEL_DIR


def candidate_model_dirs() -> tuple[pathlib.Path, ...]:
    """按当前配置排序的候选模型目录，**不检查运行时**。

    显式指定后端时只有一个候选，避免悄悄用上另一个后端的模型；``auto`` 时两个
    都算数（按平台偏好排序），这样一份安装了 MLX 的服务和一台只装基础依赖的
    客户端可以共用同一个索引。
    """
    want = (BACKEND or "auto").strip().lower()
    if want not in ("", "auto"):
        return (model_dir_for(configured_backend()),)
    preferred = model_dir_for(platform_default())
    other = MODEL_DIR if preferred == MLX_MODEL_DIR else MLX_MODEL_DIR
    return (preferred, other)


def active_backend() -> str:
    """真正要在本地跑推理时的后端。缺依赖时报错，不做静默回退（显式指定时）。"""
    want = (BACKEND or "auto").strip().lower()
    if want in ("", "auto"):
        preferred = platform_default()
        fallback = "openvino" if preferred == "mlx" else "mlx"
        if runtime_available(preferred):
            return preferred
        if runtime_available(fallback):
            return fallback
        raise RuntimeError(NO_BACKEND_MESSAGE)
    backend = configured_backend()
    if not runtime_available(backend):
        raise RuntimeError(
            f"RECALL_BACKEND={backend} 但没装 {DISPLAY[backend]}：{INSTALL_HINT[backend]}"
        )
    return backend


def active_model_dir() -> pathlib.Path:
    """当前后端实际使用的模型目录（要求本地能跑推理）。"""
    return model_dir_for(active_backend())


def embedder_class(backend: str | None = None):
    """指定后端（默认取 active_backend）的嵌入器实现。

    两者接口一致：``encode`` / ``encode_query`` / ``_get_compiled`` / ``model_dir``。
    两个模块都在类内部才 import 各自的运行时，所以这里导入模块不会拉起推理引擎。
    """
    if (backend or active_backend()) == "mlx":
        from .mlx_embedding import MLXEmbedder

        return MLXEmbedder
    if (backend or active_backend()) == "gguf":
        from .gguf_embedding import GGUFEmbedder

        return GGUFEmbedder
    from .embedding import Embedder

    return Embedder
