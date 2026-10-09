"""Synthetic tokenizer and deterministic vectors; no models or network calls."""

import hashlib
import pathlib
import re
import numpy as np
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors


class FakeEmbedding:
    model_dir = pathlib.Path("/synthetic-model")

    def __init__(self, *args, **kwargs):
        pass

    def _get_compiled(self, size):
        return None

    def encode(self, texts, is_query=False, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        out = np.zeros((len(texts), 512), dtype=np.float32)
        for i, text in enumerate(texts):
            for term in re.findall(r"\w+", text.lower()):
                j = (
                    int.from_bytes(hashlib.blake2b(term.encode(), digest_size=2).digest(), "little")
                    % 512
                )
                out[i, j] += 1
            if not out[i].any():
                out[i, 0] = 1
        if normalize:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        return out

    def encode_query(self, text, **kwargs):
        return self.encode([text], is_query=True)


def install_fake():
    from recall import engine, support

    old_resolver = engine.resolve_embedder
    old_tokenizer = support._tokenizer
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)]
    )
    support._tokenizer = tokenizer
    engine.resolve_embedder = lambda *args, **kwargs: FakeEmbedding()

    def restore():
        engine.resolve_embedder = old_resolver
        support._tokenizer = old_tokenizer

    return restore
