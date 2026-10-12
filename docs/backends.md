# 后端：一个项目，三套嵌入引擎

recall 的检索流水线（分块、FTS5 词法检索、混合排序、邻居读取、常驻服务）与推理引擎无关。
嵌入层有三个可互换的实现：前两个按平台自动挑，GGUF 需要显式指定。

| 平台 | 默认后端 | 依赖 | 模型目录 |
| --- | --- | --- | --- |
| Apple Silicon（darwin / arm64） | **MLX**（Metal） | `pip install '.[mlx]'` | `$RECALL_HOME/models/bge-small-zh-v1.5-mlx` |
| Linux、Intel macOS、其它 | **OpenVINO** | `pip install '.[openvino]'` | `$RECALL_HOME/models/bge-small-zh-v1.5` |
| 任意平台（仅显式） | **GGUF**（llama.cpp） | `pip install '.[gguf]'` | `$RECALL_HOME/models/bge-small-zh-v1.5-gguf` |

三个后端加载的是**同一份** `BAAI/bge-small-zh-v1.5` 权重，语义完全一致（CLS 池化、
L2 归一、512 维、同一个中文查询前缀）。MLX 与 OpenVINO 用的是同一份 fp32 权重，
所以它们的索引可以互相读；GGUF 是**另一份产物**（通常还量化过），身份不同，
换过去要重建索引。

## 选择规则

`RECALL_BACKEND` 取 `auto`（默认）/ `openvino` / `mlx` / `gguf`：

- **`auto`**：先按平台默认挑（Apple Silicon → MLX，其余 → OpenVINO）；该后端的依赖
  没装时回退到另一个。于是 Linux 上的既有行为不变，Apple Silicon 上不装 OpenVINO
  也能直接跑。
- **`gguf` 只按显式指定生效，不参与 auto**：它通常是量化权重，向量空间和 fp32
  不是同一个，不该悄悄替换掉用户已有的索引。
- **显式指定**：`RECALL_BACKEND=openvino|mlx|gguf` 不做回退，缺依赖时直接报错，
  而不是静默换一个后端给你。

`recall setup` 同样跟随平台默认，`--mlx` / `--openvino` / `--gguf` 可强制。
显式给了 MLX 专属选项（`--hub` / `--hf-ip`）也视为要 MLX。

选中的后端在导入 `recall.daemon` 时解析一次，落到模块级 `Embedder` / `MODEL_DIR`
两个名字上，保持和上游一样的可 monkeypatch 形态（上游测试依赖这一点）。
解析失败时不在这里抛异常——那样连 `recall stats` / `recall --help` 都会挂，
而上游只有真正做嵌入时才要求后端；这里退化成占位类，构造时才报错。

## 安装

```bash
cd recall
python3 -m venv .venv
.venv/bin/pip install -e '.[mlx]'        # Apple Silicon
# .venv/bin/pip install -e '.[openvino]' # Linux / Intel
# .venv/bin/pip install -e '.[gguf]'     # 任意平台，需 C++ 工具链 + cmake

.venv/bin/recall setup                    # 按平台默认装模型
```

用 uv 的话：`uv pip install -e '.[mlx,dev]'`。

### GGUF 模型

```bash
recall setup --gguf                                  # 默认 f16（约 48 MB）
recall setup --gguf --gguf-file bge-small-zh-v1.5-q8_0.gguf
recall setup --gguf --from /path/to/dir              # 目录里放一个 .gguf + tokenizer.json
```

默认从 `CompendiumLabs/bge-small-zh-v1.5-gguf` 取权重。**必须同时带 HF 的
`tokenizer.json`**（recall 会从 `BAAI/bge-small-zh-v1.5` 一起下）：分块和查询前缀都用它，
而且嵌入本身也刻意走 HF 的 token id（见下面「为什么不用 llama.cpp 自己的分词」）。
目录里有多个 `.gguf` 时会拒绝启动，避免猜错量化档位。

`llama-cpp-python` 没有预编译轮子，`pip install '.[gguf]'` 会就地编译
（本机 Apple M4 约 1 分钟，需要 cmake 与 C++ 工具链）。

### 模型来源

MLX 后端的模型是 HF 布局的 `config.json` + `model.safetensors` + `tokenizer.json`，
不需要导出 IR，所以安装很快：

```bash
recall setup --mlx                          # 默认从 ModelScope 下（不依赖代理）
recall setup --mlx --hub huggingface        # 走系统代理直连 HF
recall setup --mlx --from /path/to/model    # 导入本地已有的 HF 布局目录
```

`--hub` 两个源的默认分支名不同（ModelScope 用 `master`，HF 用 `main`），
`--revision` 不指定时按源自动取。实测两个源提供的三个文件 **sha256 逐字节一致**，
记在模型目录的 `manifest.json` 里。

OpenVINO 那条路仍然用上游的 `pip install '.[setup]'` + `recall setup --openvino`，
通过 `huggingface_hub` 拉权重再导出 IR；这一步需要 HF 可达（代理，或
`HF_ENDPOINT=https://hf-mirror.com`）。

#### 本地 DNS 被投毒时

有些网络把 `huggingface.co` 解析到无关 IP（实测拿到的是 Facebook 段的
`157.240.12.50`，IPv6 甚至是 `2a03:2880:...:face:b00c:...`），真实地址是 CloudFront
的 `3.164.110.x`。这种情况下不走代理的直连会因证书不匹配而失败。

注意 **`curl` 默认不读 macOS 系统代理**（只看 `https_proxy` 环境变量），而 Python 的
`urllib` 会读，所以用 curl 测会误判成「被墙」——Python 其实一直是通的。

`recall setup --mlx` 因此在连接层失败时会自动改用 Cloudflare DoH 解析出真实 IP
再重试一次（`--hf-ip` 可手动指定）。拿到真实 HTTP 响应（比如 404）时不会触发，
因为那说明网络是通的、问题在路径或分支名。

## 接口契约

两个后端暴露完全相同的方法，`engine` 和 `daemon` 不需要知道跑的是哪一个：

| 成员 | 说明 |
| --- | --- |
| `encode(texts, is_query=False, normalize=True)` | → `(n, 512)` float32，已 L2 归一 |
| `encode_query(text)` | 加中文前缀后编码 |
| `_get_compiled(seq)` | OpenVINO 按 16/32/…/512 分桶预编译；MLX 是空实现 |
| `model_dir` | 当前后端的模型目录，daemon 用它判断常驻实例能否复用 |

`encode_query` 的前缀是 `为这个句子生成表示以用于检索相关文章：`，与 OpenVINO 后端一致。

## 验证

### 1. 数值正确性（MLX 对照 torch + transformers）

`tests/verify_mlx_vs_torch.py` 用同一份 `model.safetensors` 跑 `transformers.BertModel`
作为黄金参考，再和 MLX 实现比对 10 条中英文混合文本（含空串和超过 512 token 被截断的
长文本）。两边共用同一个 `tokenizer.json`，所以差异只可能来自前向实现本身。

需要一个同时装了 torch 和 mlx 的环境（和运行时环境分开，不影响「不装 torch」）：

```bash
python3 -m venv /tmp/refvenv
/tmp/refvenv/bin/pip install torch transformers mlx safetensors
PYTHONPATH="$PWD/src" /tmp/refvenv/bin/python tests/verify_mlx_vs_torch.py \
    ~/.local/share/recall/models/bge-small-zh-v1.5-mlx
```

最后一次运行结果：

| 对比 | max abs | mean abs | min cosine |
| --- | --- | --- | --- |
| MLX(batch=1) vs torch | 8.047e-07 | 9.122e-08 | 0.999999881 |
| MLX(batch=8) vs torch | 8.047e-07 | 9.228e-08 | 0.999999940 |
| MLX(batch=1) vs batch=8 | 1.611e-07 | 2.739e-08 | 0.999999940 |

差异是 float32 舍入量级（容差 1e-4），说明 BERT 前向（权重映射、attention 缩放、
加性掩码、LayerNorm、gelu、CLS 池化）与参考实现一致；也说明按批 padding 不影响结果。
脚本用退出码表示是否超容差。

### 2. 测试套件

```bash
.venv/bin/python -m pytest -q
```

- `tests/test_backend.py` —— 平台默认、依赖回退、显式指定不回退、未知值报错，
  以及「两个后端都没装时 import 仍可用、构造才报错」。
- `tests/integration_mlx.py` —— 真实模型的 MLX 冒烟测试，对齐上游
  `integration_openvino.py` 的形态：`setup --mlx` → 索引 → 检索 → `read` → 常驻
  daemon，并断言客户端进程**既不导入 `openvino` 也不导入 `mlx`**。
  `tests/integration_openvino.py` 保持原样，服务 OpenVINO 平台。

### 3. 端到端

在 Apple Silicon 上索引本仓库源码（215 个片段），中文语义查询命中正确文件：

| 查询 | 命中 |
| --- | --- |
| 怎么选嵌入后端 | `src/recall/mlx_embedding.py` |
| 索引一致性怎么保证 | `src/recall/engine.py` |
| 模型权重放哪里 | `README.zh-CN.md` |

### 4. 跨后端一致性

两个后端互换了还能不能读同一个索引？实测（同一台 Apple Silicon，
OpenVINO CPU vs MLX Metal，同一批中英文文本）：

| 对比 | max abs | min cosine |
| --- | --- | --- |
| MLX vs torch 参考 | **8.047e-07** | 0.999999881 |
| OpenVINO vs torch 参考 | **1.353e-03** | 0.999961436 |
| OpenVINO vs MLX | 1.353e-03 | 0.999961495 |

**偏离参考实现的是 OpenVINO，不是 MLX** —— MLX 比 OpenVINO 近三个数量级。
这个 1.4e-03 来自 OpenVINO 自己的图优化/算子融合，是上游既有的行为，
不是这次改动引入的。

实际影响可以忽略：用 6 篇文档、5 个查询（中英文混合）做排序对比，
**两个后端的 top-k 排序 5/5 完全一致**，分数差异只在小数点后第 4 位
（例如同一篇文档 +0.5136 vs +0.5144）。所以索引跨后端可读，
但不应当把两个后端的分数当成可以逐位比较的东西。

### 5. 模型身份写进索引

「跨后端可读」能成立的前提是**两个后端用的是同一份权重**。换一份权重——哪怕
同样是 512 维 BERT——向量空间会整体改变，而检索仍然会返回看着正常的结果。
所以索引里现在记一份模型指纹（`meta.embedding_id`），检索与追加索引前都比对：

- 指纹取**源权重** `model.safetensors` 的 sha256（写进模型目录的
  `manifest.json` 的 `source_sha256`），而不是各自转换产物的摘要，因此
  MLX 与 OpenVINO 导出的是同一个指纹，跨后端共用索引仍然成立。
- 拿不到 `source_sha256` 时（例如 `--from` 导入别人导出的 IR）退回哈希本地权重
  产物；此时跨后端会得到不同指纹，需要重建索引。
- 不一致时**直接报错**并要求重建，不再静默给出错误结果。
- 更早建立的索引没有这张表：检索会提示一次「未记录指纹」，跑一次 `recall index`
  即补齐；补齐之前无法校验。

实测：把最后一层 LayerNorm 取反得到另一份 512 维 BERT，`setup --mlx --from`
会接受它（结构合法），但用它检索同一个索引时被拒绝：

```
索引是用另一份嵌入模型建立的，继续使用会让检索结果静默变错：
  索引记录: bge-cls-l2-v1:<...>
  当前模型: bge-cls-l2-v1:<...>
  模型目录: ...
```

改之前同一场景是**静默出错**：top-1 从正确的 `network.md`（+0.5222）漂移到
无关的 `weather.md`（−0.268）。

### 6. OpenVINO 路径未受影响

`embedding.py` 与 `tests/integration_openvino.py` **零改动**，并在本机
（Apple Silicon + OpenVINO CPU）实跑通过：

```bash
RECALL_BACKEND=openvino python tests/integration_openvino.py <IR目录> --device CPU
# {"device": "CPU", "model_export": false, ... "daemon_and_lazy_client": true}
```

`recall setup --openvino` 的 IR 导出链路也验证过（经系统代理从 HF 拉权重 →
`optimum-intel` 导出 → 校验 → 原子发布）。

### 7. GGUF（llama.cpp）

沿用同一批文本，对照 MLX 的 fp32 向量（已确认与 `transformers` 参考差 8.0e-07）：

| 变体 | 大小 | max_abs | min_cos | 查询 min_cos |
| --- | --- | --- | --- | --- |
| f32 | 95 MB | 3.06e-04 | 0.999998450 | 0.999999166 |
| **f16（默认）** | 48 MB | 3.06e-04 | 0.999998450 | 0.999997854 |
| q8_0 | 27 MB | 4.72e-03 | 0.999443531 | 0.999614298 |
| q4_k_m | 15 MB | 2.90e-02 | 0.978065073 | 0.982161939 |

f16 与 fp32 的最大绝对差 3.1e-04 —— 比 OpenVINO 自己相对参考实现的 1.4e-03 还小，
所以默认选 f16（体积只有一半）。q4_k_m 的漂移接近 3e-02，够用但明显更粗。

GGUF 元数据里 `bert.pooling_type = 2`（CLS）、`bert.embedding_length = 512`，
`GGUFEmbedder` 会校验这两项（以及 `general.architecture == bert`），不符就报错。

#### 为什么不用 llama.cpp 自己的分词

`Llama.embed()` 内部是 `self.tokenize(text.encode())`，直接用它会让 llama.cpp 的
BERT/WPM 分词器处理文本。**它会强制小写，而 BGE 的 `tokenizer.json` 是
`do_lower_case: false`**：同一句话里 `The` 被 HF 切成 `[UNK]`、被 llama.cpp 切成
`the`。实测含大写英文的文本余弦只有 **0.9657**（9 条文本里有 2 条明显偏离），
而 token 恰好一致的那些是 0.999999。

所以 `gguf_embedding` 覆盖了 `tokenize`，把 HF 的 token id 直接喂给 llama.cpp。
改完之后全部 9 条都落在 3.1e-04 以内，和另外两个后端用同一套 token id，
分块边界（448 HF token）也就和嵌入侧一致了。

#### 身份与索引

GGUF 目录的 `manifest.json` **刻意不写 `source_sha256`**，身份按 `.gguf` 自身摘要算。
于是 GGUF 索引和 fp32 索引天然不同：实测用 GGUF 建索引后再用 MLX 检索会被拦下并
列出两个指纹。这是有意的 —— 量化档位之间的向量空间差异（q4 到 3e-02）不该被当成
同一个索引。


## 改动清单

| 文件 | 改动 |
| --- | --- |
| `src/recall/backend.py` | **新增**。平台默认、后端选择、模型目录与嵌入器类解析 |
| `src/recall/mlx_embedding.py` | **新增**。MLX 上的 BERT 前向 + `MLXEmbedder` |
| `src/recall/gguf_embedding.py` | **新增**。GGUF（llama.cpp）+ `GGUFEmbedder`，覆盖 `tokenize` 用 HF 的 id |
| `src/recall/identity.py` | **新增**。索引记录的模型身份（源权重摘要 / `*.gguf` 产物摘要） |
| `src/recall/config.py` | 新增 `MLX_MODEL_DIR` / `GGUF_MODEL_DIR` / `BACKEND` |
| `src/recall/daemon.py` | `Embedder` / `MODEL_DIR` 按后端解析（带无后端时的降级）；`whoami` 上报后端；`supports()` 按设备与模型目录判断 |
| `src/recall/support.py` | 分块用的 tokenizer 跟随当前后端的模型目录；恢复日志改为原子发布 |
| `src/recall/setup.py` | 拆出 `install_mlx` / `install_openvino` / `install_gguf`；`recall setup` 按平台默认分发；新增 `--mlx` / `--openvino` / `--gguf` / `--gguf-repo` / `--gguf-file` / `--hub` / `--hf-ip` |
| `src/recall/engine.py` | 索引记录并校验模型身份；去重按 `(corpus, parent)`；`read` 的兄弟窗口限定同语料 |
| `src/recall/cli.py` | 帮助文本补充后端与平台矩阵 |
| `pyproject.toml` | 新增 `mlx` / `gguf` extra；补 macOS 分类器 |
| `tests/test_backend.py` | **新增**。后端选择逻辑单测 |
| `tests/integration_mlx.py` / `tests/integration_gguf.py` | **新增**。两套真实模型冒烟测试 |
| `tests/verify_mlx_vs_torch.py` | **新增**。与 torch 参考实现的数值对照脚本 |
| `README.md` / `README.zh-CN.md` / `.gitignore` / `docs/backends.md` | 文档与忽略规则 |

上游的 `embedding.py`（OpenVINO 实现）与 `tests/integration_openvino.py` 未改动。

## 已知限制

- **MLX 的 Metal 缓冲缓存必须设上限**。默认无上限：连续编码时进程**物理占用**
  会涨到 4 GB，而 RSS 只有 210 MB —— 从 RSS 完全看不出来，16 GB 的机器上索引到
  一半就会被内核以 low-swap 杀掉。`MLXEmbedder` 构造时设 256 MB 上限（可用
  `RECALL_MLX_CACHE_LIMIT_MB` 覆盖），并在每次批量编码（≥64 条）后
  `clear_cache()`。实测：默认 4.0 GB → 仅设上限 606 MB → 加上定期清理 204 MB。
- **MLX 后端只支持 BERT 结构**。`mlx_embedding._Bert` 按 `model_type == "bert"` 校验，
  换成别的架构会显式报错，而不是给出错误结果。
- **MLX 后端下 `--device` 不参与计算**。MLX 固定走 Metal，这个参数只是被原样记录，
  `RECALL_DEVICE=GPU` 这类 OpenVINO 设备名不会影响 MLX 的结果。
- **`_get_compiled()` 在 MLX 下是空实现**。OpenVINO 需要按静态形状预编译，MLX 没有
  这个概念，按批内实际最长序列 padding 即可；保留该方法只为满足 daemon 预热协议的
  接口要求。
- **OpenVINO 那条路没有 DoH 回退**。它走 `huggingface_hub`，需要 HF 本身可达
  （系统代理，或设置 `HF_ENDPOINT`）；DoH 回退只做在 MLX 和 GGUF 的下载器里。
- **MLX 只在 Apple Silicon 上有轮子**，其它平台装 `.[mlx]` 会被 pip 直接拒绝——
  这是有意的，比静默装不上更清楚。
- **GGUF 的 `--device` 同样只是记录**，实际由 llama.cpp 自己选后端（有 Metal 就用）。
- **GGUF 装的是 llama-cpp-python，没有预编译轮子**：`pip install '.[gguf]'` 会就地编译，
  需要 C++ 工具链与 cmake；没装的机器上会直接编译失败。
- **GGUF 索引不可与 fp32 索引互换**（身份不同）。这是刻意的：量化档位之间差到 3e-02，
  混用会让检索质量无声下降。要换档位就重建索引。
- **GGUF 走 HF 的 token id，绕开 llama.cpp 对小写的处理**。这依赖
  `llama-cpp-python` 的 `embed()` 内部调用 `self.tokenize(...)` 这一实现细节；
  升级大版本后应重跑 `tests/integration_gguf.py`（它会比对数值与身份校验）。
