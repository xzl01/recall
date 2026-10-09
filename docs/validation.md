# Validation

Default `pytest` uses synthetic tokenizer/embeddings and no external APIs. It covers window coverage, late keywords, strong lexical hits with low semantic rank, context boundaries, corpus/source isolation, incremental indexing, force rebuilds, numeric stable keys, index failures/recovery, concurrent readers/writers, CLI errors and daemon protocol behavior.

Run the real-model check separately on Linux with OpenVINO installed:

```bash
python tests/integration_openvino.py /path/to/existing/bge-small-zh-v1.5 --device CPU
python tests/integration_openvino.py /path/to/existing/bge-small-zh-v1.5 --device GPU
# With the optional setup dependencies and local safetensors weights:
python tests/integration_openvino.py /path/to/bge-weights --device CPU --export
```

Each check creates and deletes its own temporary state, imports/exports public model files, indexes synthetic sessions and a text directory, exercises JSON search/read/corpus filtering, and starts an isolated daemon. It asserts that resident clients avoid importing OpenVINO. It does not read a user's existing history or modify a production index.

The initial publication is validated on macOS with the synthetic test suite and on Linux x86_64 with the synthetic suite plus real CPU/GPU model checks. NPU and real inference on macOS/Windows are not acceptance claims. CI runs the synthetic suite and wheel build on Linux Python 3.10/3.12/3.13 and macOS Python 3.12; CI completion is checked separately from local success.
