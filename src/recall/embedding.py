"""BGE CLS embedding with OpenVINO static length buckets."""

from __future__ import annotations

import pathlib
import time

import numpy as np

from .config import MODEL_DIR, DEFAULT_DEVICE

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
BUCKETS = (16, 32, 64, 128, 256, 512)
MAX_LEN = 512


class Embedder:
    def __init__(
        self, model_dir=MODEL_DIR, device=DEFAULT_DEVICE, buckets=BUCKETS, batch=1, verbose=False
    ):
        # Resident-service clients need constants/classes, not the OpenVINO runtime.
        import openvino as ov

        self.model_dir = pathlib.Path(model_dir)
        self.device = device
        self.buckets = tuple(sorted(buckets))
        self.batch = batch
        self.verbose = verbose

        if (
            not (self.model_dir / "tokenizer.json").is_file()
            or not (self.model_dir / "ov/openvino_model.xml").is_file()
        ):
            raise RuntimeError(
                "BGE model not configured. Run recall setup or set RECALL_MODEL_DIR."
            )

        from tokenizers import Tokenizer

        self.tk = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.tk.enable_truncation(max_length=MAX_LEN)
        # 注：不用 transformers.AutoTokenizer —— 它会拖入 torch(0.77s) +
        # transformers(1.46s)，而 tokenizers 库结果完全一致、导入 0ms。

        xml = self.model_dir / "ov" / "openvino_model.xml"
        self.core = ov.Core()
        self.compiled: dict[int, object] = {}
        t0 = time.perf_counter()
        self._get_compiled(self.buckets[-1])  # 先编译最长的，暴露问题
        if verbose:
            print(
                f"[embedder] 设备 {self.device}，"
                f"已编译 {self.buckets[-1]} 桶，用时 {time.perf_counter() - t0:.2f}s"
            )

    def _get_compiled(self, seq: int):
        if seq not in self.compiled:
            m = self.core.read_model(str(self.model_dir / "ov" / "openvino_model.xml"))
            m.reshape({i.get_any_name(): [self.batch, seq] for i in m.inputs})
            self.compiled[seq] = self.core.compile_model(m, self.device)
        return self.compiled[seq]

    def _bucket_for(self, n: int) -> int:
        for b in self.buckets:
            if n <= b:
                return b
        return self.buckets[-1]

    def encode(self, texts, is_query=False, normalize=True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            texts = [QUERY_PREFIX + t for t in texts]

        enc = self.tk.encode_batch(list(texts))
        ids_list = [e.ids for e in enc]
        mask_list = [e.attention_mask for e in enc]
        type_list = [e.type_ids for e in enc]

        # 按桶分组，减少模型调用次数
        groups: dict[int, list[int]] = {}
        for i, ids in enumerate(ids_list):
            groups.setdefault(self._bucket_for(len(ids)), []).append(i)

        out = np.zeros((len(texts), 512), dtype=np.float32)
        for seq, idxs in groups.items():
            cm = self._get_compiled(seq)
            for start in range(0, len(idxs), self.batch):
                chunk = idxs[start : start + self.batch]
                n = len(chunk)
                ids = np.zeros((self.batch, seq), dtype=np.int64)
                am = np.zeros((self.batch, seq), dtype=np.int64)
                tt = np.zeros((self.batch, seq), dtype=np.int64)
                for j, i in enumerate(chunk):
                    L = len(ids_list[i])
                    ids[j, :L] = ids_list[i]
                    am[j, :L] = mask_list[i]
                    tt[j, :L] = type_list[i]
                inputs = {"input_ids": ids, "attention_mask": am, "token_type_ids": tt}
                res = cm({port.get_any_name(): inputs[port.get_any_name()] for port in cm.inputs})
                last = res[cm.output(0)]  # (batch, seq, 512)
                cls = last[np.arange(n), 0, :]  # CLS 池化
                out[chunk] = cls[:n]

        if normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.maximum(norms, 1e-12)
        return out

    def encode_query(self, text, **kw) -> np.ndarray:
        return self.encode(text, is_query=True, **kw)
