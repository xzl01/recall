"""MLX（Apple Silicon）上的 BGE CLS 嵌入后端 —— 不依赖 OpenVINO。

对外接口与 ``embedding.Embedder`` 完全一致（``encode`` / ``encode_query`` /
``_get_compiled`` / ``model_dir``），所以 ``engine`` 和 ``daemon`` 不需要知道
当前用的是哪一种后端。

权重是同一份 BAAI/bge-small-zh-v1.5，并且保持同样的语义：CLS 池化、L2 归一、
中文查询前缀。因此产出的向量与 OpenVINO 后端处在同一向量空间，既有的
``vectors.npy`` 索引依然兼容。

与 OpenVINO 后端唯一的实现差别：MLX 没有静态形状编译，不必按 16/32/…/512
分桶 reshape，按批内实际最长序列 padding 即可。``_get_compiled`` 保留为空实现，
只为满足 daemon 预热协议的接口要求。
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np

from .config import MLX_MODEL_DIR, DEFAULT_DEVICE
from .embedding import QUERY_PREFIX, BUCKETS, MAX_LEN

DIM = 512
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json")

#: MLX 的 Metal 缓冲缓存上限（MB）。
#:
#: 这个必须设，不是可选优化：默认缓存**无上限**，连续编码时实测进程物理占用涨到
#: **4.0 GB**，而 RSS 只有 210 MB —— 从 RSS 完全看不出来。16 GB 的机器上索引历史
#: 到一半就会被内核杀掉：
#:
#:     memorystatus: python3.12 [...] has greater than 50% of compressed memory
#:     memorystatus: killing python3.12 [...] due to low-swap
#:
#: 设成 256 MB 之后同样负载的物理占用约 600 MB；配合批量结束时的 ``clear_cache()``
#: 能降到 200 MB 量级。
CACHE_LIMIT_MB = int(os.environ.get("RECALL_MLX_CACHE_LIMIT_MB", "256"))

#: 一次 ``encode`` 处理这么多条以上，结束后就把 Metal 缓冲还回去。
#: 单条查询走不到这个分支，不受影响。
CLEAR_CACHE_MIN_TEXTS = 64

# 加性注意力掩码里代表「屏蔽」的大负数。HF 用 float32 最小值，效果等价：
# softmax 之后两者都精确下溢到 0。
NEG = -1e9

_SQRT1_2 = 0.7071067811865476  # 1/sqrt(2)，用于 erf 形式的精确 gelu


class _Bert:
    """在 MLX 中前向一个 BERT encoder，返回 last_hidden_state。"""

    def __init__(self, model_dir: pathlib.Path):
        import mlx.core as mx

        self.mx = mx
        cfg = json.loads((model_dir / "config.json").read_text())
        if cfg.get("model_type") != "bert":
            raise ValueError(f"只支持 BERT 结构，config.json 里是 {cfg.get('model_type')!r}")
        self.hidden = int(cfg["hidden_size"])
        self.heads = int(cfg["num_attention_heads"])
        self.layers = int(cfg["num_hidden_layers"])
        self.eps = float(cfg.get("layer_norm_eps", 1e-12))
        if self.hidden % self.heads:
            raise ValueError("hidden_size 必须能被 num_attention_heads 整除")
        self.head_dim = self.hidden // self.heads
        self.scale = self.head_dim**-0.5

        self.w = mx.load(str(model_dir / "model.safetensors"))
        self._check()

    def _check(self) -> None:
        need = [
            "embeddings.word_embeddings.weight",
            "embeddings.token_type_embeddings.weight",
            "embeddings.position_embeddings.weight",
            "embeddings.LayerNorm.weight",
        ]
        for i in range(self.layers):
            p = f"encoder.layer.{i}."
            need += [
                p + "attention.self.query.weight",
                p + "attention.self.key.weight",
                p + "attention.self.value.weight",
                p + "attention.output.dense.weight",
                p + "attention.output.LayerNorm.weight",
                p + "intermediate.dense.weight",
                p + "output.dense.weight",
                p + "output.LayerNorm.weight",
            ]
        missing = [k for k in need if k not in self.w]
        if missing:
            raise ValueError(f"权重缺失（不是 BERT 检查点？）：{missing[:4]}")

    # ── 基础算子 ──────────────────────────────────────────────
    def _linear(self, x, prefix: str):
        return x @ self.w[prefix + ".weight"].T + self.w[prefix + ".bias"]

    def _layernorm(self, x, prefix: str):
        # HF 的 LayerNorm 用有偏方差（除以 N），mx.var 默认 ddof=0，一致。
        mx = self.mx
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) * mx.rsqrt(var + self.eps) * self.w[f"{prefix}.weight"] + self.w[
            f"{prefix}.bias"
        ]

    # ── 前向 ──────────────────────────────────────────────────
    def _embed(self, ids, token_type_ids):
        mx = self.mx
        w = self.w
        pos = mx.arange(ids.shape[1])
        x = (
            w["embeddings.word_embeddings.weight"][ids]
            + w["embeddings.token_type_embeddings.weight"][token_type_ids]
            + w["embeddings.position_embeddings.weight"][pos]
        )
        return self._layernorm(x, "embeddings.LayerNorm")

    def _layer(self, x, index: int, mask):
        mx = self.mx
        p = f"encoder.layer.{index}."
        b, length, hidden = x.shape

        def heads(t):
            return t.reshape(b, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)

        q = heads(self._linear(x, p + "attention.self.query"))
        k = heads(self._linear(x, p + "attention.self.key"))
        v = heads(self._linear(x, p + "attention.self.value"))
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale + mask
        ctx = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(b, length, hidden)

        x = self._layernorm(
            x + self._linear(ctx, p + "attention.output.dense"), p + "attention.output.LayerNorm"
        )
        h = self._linear(x, p + "intermediate.dense")
        h = 0.5 * h * (1.0 + mx.erf(h * _SQRT1_2))  # 精确 gelu，对应 hidden_act="gelu"
        return self._layernorm(x + self._linear(h, p + "output.dense"), p + "output.LayerNorm")

    def run(self, ids: np.ndarray, token_type_ids: np.ndarray, attention_mask: np.ndarray):
        """(B, L) int 输入 → (B, L, 512) float32 numpy 的 last_hidden_state。"""
        mx = self.mx
        x = self._embed(mx.array(ids.astype(np.int32)), mx.array(token_type_ids.astype(np.int32)))
        mask = mx.array(((1.0 - attention_mask) * NEG).astype(np.float32))[:, None, None, :]
        for i in range(self.layers):
            x = self._layer(x, i, mask)
        mx.eval(x)
        return np.array(x, dtype=np.float32)


class MLXEmbedder:
    """与 ``embedding.Embedder`` 同接口的 MLX 实现。"""

    def __init__(
        self,
        model_dir=MLX_MODEL_DIR,
        device=DEFAULT_DEVICE,
        buckets=BUCKETS,
        batch=1,
        verbose=False,
    ):
        self.model_dir = pathlib.Path(model_dir)
        missing = [n for n in MODEL_FILES if not (self.model_dir / n).is_file()]
        if missing:
            raise RuntimeError(
                f"MLX 模型未配置：{self.model_dir} 缺少 {', '.join(missing)}。"
                "请运行 recall setup --mlx，或设置 RECALL_MLX_MODEL_DIR。"
            )
        self.device = device
        self.buckets = tuple(sorted(buckets))
        self.batch = max(1, int(batch))
        self.verbose = verbose

        import mlx.core as mx

        self.mx = mx
        # 必须限制 Metal 缓冲缓存，否则长任务会把进程物理占用顶到 4 GB（见
        # CACHE_LIMIT_MB 的说明）。老版本没有 set_cache_limit 就退回 mx.metal。
        limit = CACHE_LIMIT_MB << 20
        setter = getattr(mx, "set_cache_limit", None) or getattr(
            getattr(mx, "metal", None), "set_cache_limit", None
        )
        if setter is not None:
            try:
                setter(limit)
            except Exception:
                pass

        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tk.enable_truncation(max_length=MAX_LEN)

        t0 = time.perf_counter()
        self.bert = _Bert(self.model_dir)
        # 预热：把 Metal kernel 编译和权重上传的开销挪到构造阶段，
        # 否则第一次真实查询会把它算进延迟里。
        self.bert.run(
            np.zeros((1, 8), np.int64), np.zeros((1, 8), np.int64), np.ones((1, 8), np.int64)
        )
        if verbose:
            print(
                f"[mlx] 后端 MLX device={self.device} batch={self.batch}，"
                f"模型就绪 {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

    def _get_compiled(self, seq: int):
        """接口兼容占位：MLX 无需静态形状预编译。"""
        return None

    def encode(self, texts, is_query=False, normalize=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        if not texts:
            return out
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        enc = self.tk.encode_batch(texts)
        for start in range(0, len(enc), self.batch):
            chunk = enc[start : start + self.batch]
            n = len(chunk)
            length = max(1, min(max(len(e.ids) for e in chunk), MAX_LEN))
            ids = np.zeros((n, length), dtype=np.int64)
            types = np.zeros((n, length), dtype=np.int64)
            mask = np.zeros((n, length), dtype=np.int64)
            for j, e in enumerate(chunk):
                take = min(len(e.ids), length)
                ids[j, :take] = e.ids[:take]
                types[j, :take] = e.type_ids[:take]
                mask[j, :take] = e.attention_mask[:take]
            last = self.bert.run(ids, types, mask)  # (n, length, 512)
            out[start : start + n] = last[:, 0, :]  # CLS 池化

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)
        if len(texts) >= CLEAR_CACHE_MIN_TEXTS:
            # 批量（索引）路径：把 Metal 缓冲还给系统。单条查询不走这里，
            # 免得为了省一点内存反复丢掉刚编译好的 kernel。
            self._clear_cache()
        return out.astype(np.float32, copy=False)

    def _clear_cache(self) -> None:
        clear = getattr(self.mx, "clear_cache", None) or getattr(
            getattr(self.mx, "metal", None), "clear_cache", None
        )
        if clear is not None:
            try:
                clear()
            except Exception:
                pass

    def encode_query(self, text, **kw) -> np.ndarray:
        return self.encode(text, is_query=True, **kw)
