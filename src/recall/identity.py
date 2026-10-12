"""嵌入模型身份：让索引记住自己是哪份权重算出来的。

同一份 `BAAI/bge-small-zh-v1.5` 经不同转换器导出（MLX safetensors / OpenVINO IR）
算出的向量语义等价，所以这两个后端应当共用同一个身份。但**换一份权重**——哪怕
同样是 512 维 BERT——产出的是完全不同的向量空间，混进同一个索引会让检索静默
变错（实测：换权重后 top-1 从正确文档漂移到无关文档，且没有任何报错）。

因此身份取「**源权重**的摘要」而不是「产物」的摘要：

- ``setup`` 会把下载到的 ``model.safetensors`` 摘要记成 ``source_sha256``，
  两个后端记的是同一份文件的同一个摘要 → 跨后端可共用索引。
- 没有 ``source_sha256`` 时（导入别人导出的 IR、或更早装的模型）退回哈希本地
  权重产物，此时跨后端会得到不同身份 → 要求重建，宁可多一次重建也不静默出错。
"""

from __future__ import annotations

import hashlib
import json
import pathlib

#: 池化/归一/前缀等语义变了也要算作不同身份，所以并进指纹里。
SCHEMA = "bge-cls-l2-v1"

#: 没有 source_sha256 时按顺序回退的权重产物。支持 glob（GGUF 文件名带量化档位）。
WEIGHT_FILES = ("model.safetensors", "ov/openvino_model.bin", "*.gguf")


def file_sha256(path: pathlib.Path) -> str:
    """流式算文件摘要，避免把 95 MB 权重整个读进内存。"""
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_digest(model_dir: pathlib.Path) -> str | None:
    """权重产物的摘要。单个文件就是它自己的 sha256，多个则合并算。"""
    for pattern in WEIGHT_FILES:
        matches = (
            sorted(model_dir.glob(pattern))
            if "*" in pattern
            else ([model_dir / pattern] if (model_dir / pattern).is_file() else [])
        )
        if not matches:
            continue
        if len(matches) == 1:
            return file_sha256(matches[0])
        combined = hashlib.sha256()
        for path in matches:
            combined.update(path.name.encode())
            combined.update(file_sha256(path).encode())
        return combined.hexdigest()
    return None


def recorded_source_digest(model_dir: pathlib.Path) -> str | None:
    """从 manifest.json 里取源权重摘要，没有就返回 None。"""
    manifest = pathlib.Path(model_dir) / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text())
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("source_sha256"):
        return str(data["source_sha256"])
    # 兼容本次改动之前装的 MLX 模型：sha256 映射里已经有 model.safetensors。
    recorded = data.get("sha256")
    if isinstance(recorded, dict) and recorded.get("model.safetensors"):
        return str(recorded["model.safetensors"])
    return None


def model_identity(model_dir) -> str | None:
    """模型目录的稳定指纹；无法判定时返回 None。

    真实后端一定算得出来：``MLXEmbedder`` 要求 ``model.safetensors``，
    ``Embedder`` 要求 ``ov/openvino_model.bin``，都在 :data:`WEIGHT_FILES` 里。
    只有测试替身或自定义目录才会返回 None，调用方据此跳过校验而不是拒绝服务。
    """
    model_dir = pathlib.Path(model_dir)
    digest = recorded_source_digest(model_dir)
    if digest is None:
        digest = _weight_digest(model_dir)
    if digest is None:
        return None
    return f"{SCHEMA}:{digest}"
