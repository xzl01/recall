"""对照验证 MLX 后端与 transformers(torch) 参考实现是否数值一致。

需要一个同时装了 torch / transformers 和 mlx / safetensors 的环境：

    pip install torch transformers mlx safetensors
    python tests/verify_mlx_vs_torch.py /path/to/bge-small-zh-v1.5-mlx

两边读的是同一份 model.safetensors，tokenizer 也用同一个 tokenizer.json，
所以差异只可能来自前向实现本身。脚本默认离线（参考实现只从本地目录加载权重）。

退出码非 0 表示超出容差。
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np  # noqa: E402

TOLERANCE = 1e-4

TEXTS = [
    "今天天气很好，我们一起去公园散步吧。",
    "外面阳光明媚，适合出门走走。",
    "召回索引使用 BGE 模型把文本编码成向量。",
    "数据库连接超时，需要重试机制。",
    "connection timeout retry backoff",
    "The quick brown fox jumps over the lazy dog.",
    "混合 mixed 中英文 text with 数字 12345 和符号 !@#$%",
    "短",
    "",
    "这是一段刻意拉长的文本，用来验证接近 512 token 上限时的行为。" * 12,
]


def pad(encodings):
    """和 MLX 后端一样：按批内实际最长序列 padding（不按 16/32/…/512 分桶）。"""
    length = max(1, min(max(len(e.ids) for e in encodings), 512))
    n = len(encodings)
    ids = np.zeros((n, length), np.int64)
    types = np.zeros((n, length), np.int64)
    mask = np.zeros((n, length), np.int64)
    for j, e in enumerate(encodings):
        take = min(len(e.ids), length)
        ids[j, :take] = e.ids[:take]
        types[j, :take] = e.type_ids[:take]
        mask[j, :take] = e.attention_mask[:take]
    return ids, types, mask


def reference(model_dir, texts):
    import torch
    from tokenizers import Tokenizer
    from transformers import BertModel

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.enable_truncation(max_length=512)
    model = BertModel.from_pretrained(str(model_dir), local_files_only=True).eval()
    ids, types, mask = pad(tok.encode_batch(texts))
    with torch.no_grad():
        out = (
            model(
                input_ids=torch.tensor(ids),
                attention_mask=torch.tensor(mask),
                token_type_ids=torch.tensor(types),
            )
            .last_hidden_state[:, 0, :]
            .numpy()
        )  # CLS 池化
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    a = parser.parse_args()
    model_dir = a.model_dir.expanduser().resolve()

    from recall.mlx_embedding import MLXEmbedder

    reference_vectors = reference(model_dir, TEXTS).astype(np.float32)
    mine_b1 = MLXEmbedder(model_dir=model_dir, batch=1).encode(TEXTS)
    mine_b8 = MLXEmbedder(model_dir=model_dir, batch=8).encode(TEXTS)

    rows = [
        ("MLX(batch=1) vs torch", mine_b1, reference_vectors),
        ("MLX(batch=8) vs torch", mine_b8, reference_vectors),
        ("MLX(batch=1) vs batch=8", mine_b1, mine_b8),
    ]
    worst = 0.0
    print(f"{'comparison':<26} {'max_abs':>11} {'mean_abs':>11} {'min_cos':>14}")
    for name, left, right in rows:
        diff = np.abs(left - right)
        cos = (left * right).sum(1)
        worst = max(worst, float(diff.max()))
        print(f"{name:<26} {diff.max():>11.3e} {diff.mean():>11.3e} {cos.min():>14.9f}")

    ok = worst < a.tolerance
    print()
    print(
        json.dumps(
            dict(
                texts=len(TEXTS),
                max_abs=worst,
                tolerance=a.tolerance,
                verdict="PASS" if ok else "FAIL",
            )
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
