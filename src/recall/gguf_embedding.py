"""GGUF（llama.cpp）上的 BGE CLS 嵌入后端 —— 不依赖 OpenVINO / MLX。

对外接口与 ``embedding.Embedder`` 完全一致（``encode`` / ``encode_query`` /
``_get_compiled`` / ``model_dir``）。

两个关键点都是实测出来的，不是照抄默认行为：

1. **用 HF 的 tokenizer 产生 token id 再喂给 llama.cpp，而不是让它自己分词。**
   llama.cpp 的 BERT/WPM 分词器会强制小写，而 BGE 的 tokenizer.json 是
   ``do_lower_case: false``：同一句话里 ``The`` 被 HF 切成 ``[UNK]``、
   被 llama.cpp 切成 ``the``。实测含大写英文的文本余弦只有 0.9657；改成喂 HF 的
   id 之后升到 0.999998（f32 GGUF）。所以这里覆盖 ``Llama.tokenize``。
2. **池化方式必须是 CLS。** GGUF 元数据带 ``bert.pooling_type``；不是 CLS 就报错，
   免得用错误的方式池化却给出看着正常的结果。

代价要说清楚：GGUF 是**另一份产物**（通常还量化过），向量空间和 fp32 权重不是
同一个。它因此天然拿到独立的索引身份，切换后端需要重建索引。
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .config import GGUF_MODEL_DIR, DEFAULT_DEVICE
from .embedding import QUERY_PREFIX, BUCKETS, MAX_LEN

DIM = 512
TOKENIZER_FILE = "tokenizer.json"
#: llama.cpp 的 CLS 池化枚举值（``LLAMA_POOLING_TYPE_CLS``）。
CLS_POOLING = 2


def find_weights(model_dir: pathlib.Path) -> pathlib.Path:
    """模型目录里那个 .gguf。允许多个时要求显式指定，避免猜错量化档位。"""
    found = sorted(pathlib.Path(model_dir).glob("*.gguf"))
    if not found:
        raise RuntimeError(
            f"GGUF 模型未配置：{model_dir} 里没有 .gguf 文件。"
            "请运行 recall setup --gguf，或设置 RECALL_GGUF_MODEL_DIR。"
        )
    if len(found) > 1:
        names = ", ".join(p.name for p in found)
        raise RuntimeError(
            f"{model_dir} 里有多个 .gguf（{names}）。"
            "GGUF 各量化档位的向量空间不同，请只保留要用的那个。"
        )
    return found[0]


#: 表示「就在 GPU 上」的设备名。llama.cpp 自己决定后端，``RECALL_DEVICE`` / ``--device``
#: 对这个后端没有意义 —— 它只是**常驻服务的兼容性匹配键**（见 ``daemon.whoami_payload``）。
#:
#: 日志必须分开这两者。以前打印 ``device={self.device}``，而实际上构造 ``Llama`` 时
#: 没有传 ``n_gpu_layers``，默认是 **0 —— 全程 CPU**；``llama_supports_gpu_offload()``
#: 为 True 只说明这个库**能**用 GPU，不代表这个模型真的 offload 了。
GPU_LABELS = ("GPU", "METAL")


def _actual_device(llm) -> str:
    """这个 GGUF 模型实际在哪算 —— 看 offload 了几层，不看库的能力。"""
    layers = getattr(getattr(llm, "model_params", None), "n_gpu_layers", 0) or 0
    try:
        layers = int(layers)
    except (TypeError, ValueError):
        layers = 0
    if layers == 0:
        return "CPU（n_gpu_layers=0，未 offload）"
    return f"GPU offload（n_gpu_layers={layers}）"


def _device_note(device) -> str:
    claimed = str(device or "").strip()
    if not claimed or claimed.upper() in GPU_LABELS:
        return ""
    return f"；--device={claimed} 不参与计算（仅用于常驻服务匹配）"


class GGUFEmbedder:
    """与 ``embedding.Embedder`` 同接口的 GGUF 实现。"""

    def __init__(
        self,
        model_dir=GGUF_MODEL_DIR,
        device=DEFAULT_DEVICE,
        buckets=BUCKETS,
        batch=1,
        verbose=False,
    ):
        import llama_cpp

        self.model_dir = pathlib.Path(model_dir)
        tokenizer_path = self.model_dir / TOKENIZER_FILE
        if not tokenizer_path.is_file():
            raise RuntimeError(
                f"GGUF 模型未配置：{self.model_dir} 缺少 {TOKENIZER_FILE}。"
                "分块与查询前缀都依赖 HF 分词器，不能只用 .gguf。"
            )
        self.weights = find_weights(self.model_dir)
        self.device = device
        self.buckets = tuple(sorted(buckets))
        self.batch = max(1, int(batch))
        self.verbose = verbose

        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(tokenizer_path))
        self.tk.enable_truncation(max_length=MAX_LEN)

        if not verbose:
            # llama.cpp 的 Metal 初始化会往 stderr 刷几十行，压掉。
            try:
                self._log_callback = llama_cpp.LLAMA_LOG_CALLBACK(
                    lambda level, text, user_data: None
                )
                llama_cpp.llama_log_set(self._log_callback, None)
            except Exception:
                self._log_callback = None

        t0 = time.perf_counter()
        self.llm = _hftokenized_llama()(
            model_path=str(self.weights),
            embedding=True,
            n_ctx=MAX_LEN,
            n_batch=MAX_LEN,
            verbose=False,
            hf_tokenizer=self.tk,
        )
        pooling = self.llm.pooling_type()
        expected = getattr(llama_cpp, "LLAMA_POOLING_TYPE_CLS", CLS_POOLING)
        if pooling != expected:
            raise RuntimeError(
                f"{self.weights.name} 的池化方式是 {pooling}，不是 CLS（{expected}）。"
                "recall 需要 CLS 池化；换一份转换正确的 GGUF。"
            )
        dim = int(self.llm.n_embd())
        if dim != DIM:
            raise RuntimeError(f"GGUF 输出维度是 {dim}，recall 需要 {DIM}。")
        if verbose:
            print(
                f"[gguf] 后端 GGUF 实跑 {_actual_device(self.llm)}"
                f"{_device_note(self.device)} batch={self.batch}，"
                f"模型就绪 {time.perf_counter() - t0:.2f}s（{self.weights.name}）",
                flush=True,
            )

    def _get_compiled(self, seq: int):
        """接口兼容占位：llama.cpp 自己管理批与上下文，没有静态形状预编译。"""
        return None

    def encode(self, texts, is_query=False, normalize=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.zeros((0, DIM), dtype=np.float32)
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]
        vectors = self.llm.embed(texts, normalize=bool(normalize))
        out = np.asarray(vectors, dtype=np.float32)
        if out.shape != (len(texts), DIM):
            raise RuntimeError(f"GGUF 返回了 {out.shape}，期望 {(len(texts), DIM)}。")
        return out

    def encode_query(self, text, **kw) -> np.ndarray:
        return self.encode(text, is_query=True, **kw)


def _hftokenized_llama():
    """构造一个把分词交给 HF 的 ``Llama`` 子类（延迟到真要用 llama_cpp 时）。

    ``Llama.embed()`` 内部就是 ``self.tokenize(text.encode())``，所以覆盖这一个方法
    就能把分词完全交给 HF —— 和另外两个后端用同一套 token id。
    """
    global _LLAMA_CLASS
    if _LLAMA_CLASS is None:
        from llama_cpp import Llama

        class HFTokenizedLlama(Llama):
            def __init__(self, *args, hf_tokenizer=None, **kwargs):
                self._hf_tokenizer = hf_tokenizer
                super().__init__(*args, **kwargs)

            def tokenize(self, text, add_bos=True, special=False):
                if self._hf_tokenizer is not None and isinstance(text, (bytes, bytearray)):
                    # HF 的 tokenizer.json 已经带 [CLS]/[SEP] 后处理，不需要再加。
                    return list(self._hf_tokenizer.encode(text.decode("utf-8")).ids)[:MAX_LEN]
                return super().tokenize(text, add_bos, special)

        _LLAMA_CLASS = HFTokenizedLlama
    return _LLAMA_CLASS


_LLAMA_CLASS = None
