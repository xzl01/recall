"""Explicit model download/export or import; normal retrieval stays offline."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import socket
import tempfile
import urllib.request

from .backend import active_backend
from .config import MODEL_DIR, MLX_MODEL_DIR
from .identity import file_sha256

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
FILES = ("tokenizer.json", "ov/openvino_model.xml", "ov/openvino_model.bin")

# MLX 后端直接吃 safetensors，不需要导出 IR，所以只有三个文件。
MLX_FILES = ("config.json", "model.safetensors", "tokenizer.json")
# 默认走 ModelScope：国内可直连，且 BAAI 官方在那里同步了同一份权重。
DEFAULT_SOURCE = "modelscope"
ENDPOINTS = {
    "modelscope": "https://www.modelscope.cn/api/v1/models",
    "huggingface": "https://huggingface.co",
}
# 两个站点的默认分支名不同：ModelScope 用 master，HuggingFace 用 main。
DEFAULT_REVISION = {"modelscope": "master", "huggingface": "main"}


def validate(path):
    from tokenizers import Tokenizer

    for name in FILES:
        if not (path / name).is_file():
            raise ValueError(f"Missing model file: {path / name}")
    Tokenizer.from_file(str(path / "tokenizer.json"))


def validate_mlx(path):
    """校验 MLX 模型目录：BERT 结构 + 可加载的 tokenizer + 完整的权重键。"""
    from tokenizers import Tokenizer

    for name in MLX_FILES:
        if not (path / name).is_file():
            raise ValueError(f"Missing model file: {path / name}")
    config = json.loads((path / "config.json").read_text())
    if config.get("model_type") != "bert":
        raise ValueError(
            f"MLX backend only supports BERT checkpoints, got {config.get('model_type')!r}"
        )
    Tokenizer.from_file(str(path / "tokenizer.json"))
    from safetensors import safe_open

    with safe_open(str(path / "model.safetensors"), framework="numpy") as handle:
        keys = set(handle.keys())
    expected = {
        "embeddings.word_embeddings.weight",
        "embeddings.token_type_embeddings.weight",
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.0.output.dense.weight",
    }
    if not expected <= keys:
        raise ValueError("model.safetensors is not a BERT checkpoint (missing encoder weights)")


def _url(source, repo, revision, name):
    if source == "modelscope":
        return f"{ENDPOINTS[source]}/{repo}/repo?Revision={revision}&FilePath={name}"
    return f"{ENDPOINTS[source]}/{repo}/resolve/{revision}/{name}"


DOH_ENDPOINT = "https://cloudflare-dns.com/dns-query"


def _doh_resolve(host, timeout=10):
    """用 DNS-over-HTTPS 解析主机名，绕开被投毒的本地 DNS。

    某些网络把 huggingface.co 解析到无关 IP（实测拿到的是 Facebook 段），
    于是 TLS 握手直接失败，看起来像「被墙」。真实 IP 仍然可达，所以换一个
    可信解析器拿到正确地址后照常连接。
    """
    request = urllib.request.Request(
        f"{DOH_ENDPOINT}?name={host}&type=A",
        headers={"accept": "application/dns-json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    return [a["data"] for a in data.get("Answer", []) if a.get("type") == 1]


@contextlib.contextmanager
def _pinned_dns(host, address):
    """在作用域内把这个主机名固定解析到指定 IP；TLS SNI/证书仍按原主机名校验。"""
    original = socket.getaddrinfo

    def patched(name, *args, **kwargs):
        return original(address if name == host else name, *args, **kwargs)

    socket.getaddrinfo = patched
    try:
        yield
    finally:
        socket.getaddrinfo = original


def _fetch(url, destination, pinned=None):
    with contextlib.ExitStack() as stack:
        if pinned:
            stack.enter_context(pinned)
        with urllib.request.urlopen(url, timeout=300) as response, open(destination, "wb") as out:
            shutil.copyfileobj(response, out)


def _hint(hub, name, error):
    other = [k for k in ENDPOINTS if k != hub][0]
    return (
        f"下载 {name} 失败（{error}）。可以用 --from DIR 指向本地已有的模型目录，"
        f"或换个源：--hub {other}"
    )


def _download(stage, hub, repo, revision, hf_ip=None):
    for name in MLX_FILES:
        url = _url(hub, repo, revision, name)
        print(f"  下载 {name} ...", flush=True)
        try:
            _fetch(url, stage / name, _pinned_dns("huggingface.co", hf_ip) if hf_ip else None)
        except urllib.error.HTTPError as error:
            # 拿到了真实 HTTP 响应，说明连接是通的，问题在路径/分支名，不该再折腾 DNS。
            raise RuntimeError(_hint(hub, name, error)) from None
        except OSError as error:
            # 连接层失败。本地 DNS 被投毒时会表现为证书不匹配（连到无关 IP），
            # 这时用 DoH 取真实地址重试一次。
            if hub != "huggingface" or hf_ip:
                raise RuntimeError(_hint(hub, name, error)) from None
            try:
                addresses = _doh_resolve("huggingface.co")
            except OSError:
                addresses = []
            if not addresses:
                raise RuntimeError(_hint(hub, name, error)) from None
            print(f"  连接失败（{error}）；改用 DoH 解析到 {addresses[0]} 重试", flush=True)
            try:
                _fetch(url, stage / name, _pinned_dns("huggingface.co", addresses[0]))
            except OSError as retry_error:
                raise RuntimeError(_hint(hub, name, retry_error)) from None


def install_mlx(args):
    """装 MLX 后端的模型：从本地目录导入，或从 ModelScope / HuggingFace 下载。"""
    destination = MLX_MODEL_DIR
    if destination.exists() and any(destination.iterdir()):
        validate_mlx(destination)
        if args.existing:
            validate_mlx(args.existing)
            if any(
                (args.existing / n).read_bytes() != (destination / n).read_bytes()
                for n in MLX_FILES
            ):
                raise ValueError(
                    "A different model is already installed. Use a new RECALL_HOME/index."
                )
        print(f"MLX 模型已配置：{destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".model-stage-", dir=destination.parent) as temp:
        stage = Path(temp) / "model"
        stage.mkdir(parents=True)
        if args.existing:
            validate_mlx(args.existing)
            for name in MLX_FILES:
                shutil.copy2(args.existing / name, stage / name)
            revision = None
        else:
            revision = args.revision or DEFAULT_REVISION[args.hub]
            _download(stage, args.hub, args.repo, revision, args.hf_ip)
        validate_mlx(stage)
        manifest = {
            "model": args.repo,
            "source": "local" if args.existing else args.hub,
            "revision": revision,
            "backend": "mlx",
            "dimension": 512,
            "pooling": "CLS",
            "normalized": True,
            # 索引靠这个摘要认定「同一份权重」。两个后端记的应该是同一个值。
            "source_sha256": file_sha256(stage / "model.safetensors"),
            "sha256": {n: file_sha256(stage / n) for n in MLX_FILES},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            destination.rmdir()  # Only remove an empty destination.
        os.rename(stage, destination)
    print(f"MLX 模型已配置：{destination}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--mlx",
        action="store_true",
        help="强制装 MLX 后端（Apple Silicon）的 safetensors 模型，不导出 OpenVINO IR",
    )
    backend.add_argument(
        "--openvino",
        action="store_true",
        help="强制导出 OpenVINO IR（Intel / Linux 平台的默认行为）",
    )
    parser.add_argument(
        "--hub",
        choices=sorted(ENDPOINTS),
        default=None,
        help=f"MLX 模型的下载源（默认 {DEFAULT_SOURCE}；huggingface 走系统代理）",
    )
    parser.add_argument("--repo", default=MODEL_ID, help="MLX 模型的仓库 ID")
    parser.add_argument(
        "--revision",
        default=None,
        help="MLX 模型的仓库修订（默认按源取：modelscope=master，huggingface=main）",
    )
    parser.add_argument(
        "--hf-ip",
        default=None,
        help="把 huggingface.co 固定到这个 IP（本地 DNS 被投毒时用；默认失败后自动走 DoH）",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="existing", type=Path, help="Import an already exported BGE model directory"
    )
    group.add_argument(
        "--source", type=Path, help="Export local BGE safetensors weights instead of downloading"
    )
    a = parser.parse_args(argv)

    # 不指定后端时按平台默认：Apple Silicon 装 MLX 模型，其余导出 OpenVINO IR。
    # 显式给了 MLX 专属选项（--hub / --hf-ip）也视为要 MLX，避免用户还要多打一个开关。
    if a.openvino:
        chosen = "openvino"
    elif a.mlx or a.hub or a.hf_ip:
        chosen = "mlx"
    else:
        chosen = active_backend()
    a.hub = a.hub or DEFAULT_SOURCE

    if chosen == "mlx":
        if a.source:
            raise ValueError("--source 是 OpenVINO 导出用的；MLX 请用 --from 或 --hub")
        return install_mlx(a)
    if a.hf_ip or a.repo != MODEL_ID or a.revision:
        raise ValueError("--repo / --revision / --hf-ip 只对 MLX 后端有意义")

    return install_openvino(a)


def install_openvino(a):
    destination = MODEL_DIR
    if destination.exists() and any(destination.iterdir()):
        validate(destination)
        if a.existing:
            validate(a.existing)
            if any((a.existing / n).read_bytes() != (destination / n).read_bytes() for n in FILES):
                raise ValueError(
                    "A different model is already installed. Use a new RECALL_HOME/index."
                )
        if a.source:
            raise ValueError("Model already configured. Export into a new RECALL_HOME/index.")
        print(f"Model already configured: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix=".model-stage-", dir=destination.parent) as temp:
        stage = Path(temp) / "model"
        (stage / "ov").mkdir(parents=True)
        source_weights = None
        if a.existing:
            validate(a.existing)
            for name in FILES:
                shutil.copy2(a.existing / name, stage / name)
        else:
            try:
                from huggingface_hub import snapshot_download
                from optimum.intel import OVModelForFeatureExtraction
            except ImportError:
                raise RuntimeError(
                    "Model export needs setup dependencies: pip install '.[setup]'"
                ) from None
            source = a.source or Path(
                snapshot_download(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    allow_patterns=["*.json", "*.txt", "*.safetensors"],
                )
            )
            model = OVModelForFeatureExtraction.from_pretrained(
                str(source),
                export=True,
                compile=False,
                load_in_8bit=False,
                trust_remote_code=False,
            )
            model.save_pretrained(stage / "ov")
            shutil.copy2(source / "tokenizer.json", stage / "tokenizer.json")
            source_weights = source / "model.safetensors"
        validate(stage)
        manifest = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION if not a.existing and not a.source else None,
            "dimension": 512,
            "pooling": "CLS",
            "normalized": True,
            # 记录**源权重**摘要而不是 IR 产物摘要：MLX 后端用的是同一份
            # model.safetensors，因此两个后端能共用同一个索引身份。
            # 导入别人导出的 IR（--from）时拿不到源权重，只能留空退回产物摘要。
            "source_sha256": (
                file_sha256(source_weights)
                if source_weights is not None and source_weights.is_file()
                else None
            ),
            "sha256": {n: file_sha256(stage / n) for n in FILES},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            destination.rmdir()  # Only remove an empty destination.
        os.rename(stage, destination)
    print(f"Model configured: {destination}")
