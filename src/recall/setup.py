"""Explicit model download/export or import; normal retrieval stays offline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .config import MODEL_DIR

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
FILES = ("tokenizer.json", "ov/openvino_model.xml", "ov/openvino_model.bin")


def validate(path):
    from tokenizers import Tokenizer

    for name in FILES:
        if not (path / name).is_file():
            raise ValueError(f"Missing model file: {path / name}")
    Tokenizer.from_file(str(path / "tokenizer.json"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="existing", type=Path, help="Import an already exported BGE model directory"
    )
    group.add_argument(
        "--source", type=Path, help="Export local BGE safetensors weights instead of downloading"
    )
    a = parser.parse_args(argv)
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
        validate(stage)
        manifest = {
            "model": MODEL_ID,
            "revision": MODEL_REVISION if not a.existing and not a.source else None,
            "dimension": 512,
            "pooling": "CLS",
            "normalized": True,
            "sha256": {n: hashlib.sha256((stage / n).read_bytes()).hexdigest() for n in FILES},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            destination.rmdir()  # Only remove an empty destination.
        os.rename(stage, destination)
    print(f"Model configured: {destination}")
