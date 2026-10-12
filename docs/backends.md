# 后端：一个项目，两套嵌入引擎

recall 的检索流水线（分块、FTS5 词法检索、混合排序、邻居读取、常驻服务）与推理引擎无关。
只有嵌入层有两个可互换的实现，按平台自动挑：

| 平台 | 默认后端 | 依赖 | 模型目录 |
| --- | --- | --- | --- |
| Apple Silicon（darwin / arm64） | **MLX**（Metal） | `pip install '.[mlx]'` | `$RECALL_HOME/models/bge-small-zh-v1.5-mlx` |
| Linux、Intel macOS、其它 | **OpenVINO** | `pip install '.[openvino]'` | `$RECALL_HOME/models/bge-small-zh-v1.5` |

两个后端加载的是**同一份** `BAAI/bge-small-zh-v1.5` 权重，语义完全一致（CLS 池化、
L2 归一、512 维、同一个中文查询前缀），所以 `vectors.npy` 的形状和含义跨后端相同，
既有索引不用重建。

## 选择规则

`RECALL_BACKEND` 取 `auto`（默认）/ `openvino` / `mlx`：

- **`auto`**：先按平台默认挑（Apple Silicon → MLX，其余 → OpenVINO）；该后端的依赖
  没装时回退到另一个。于是 Linux 上的既有行为不变，Apple Silicon 上不装 OpenVINO
  也能直接跑。
- **显式指定**：`RECALL_BACKEND=openvino|mlx` 不做回退，缺依赖时直接报错，
  而不是静默换一个后端给你。

`recall setup` 同样跟随平台默认，`--mlx` / `--openvino` 可强制另一个。
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

.venv/bin/recall setup                    # 按平台默认装模型
```

用 uv 的话：`uv pip install -e '.[mlx,dev]'`。

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

### 5. OpenVINO 路径未受影响

`embedding.py` 与 `tests/integration_openvino.py` **零改动**，并在本机
（Apple Silicon + OpenVINO CPU）实跑通过：

```bash
RECALL_BACKEND=openvino python tests/integration_openvino.py <IR目录> --device CPU
# {"device": "CPU", "model_export": false, ... "daemon_and_lazy_client": true}
```

`recall setup --openvino` 的 IR 导出链路也验证过（经系统代理从 HF 拉权重 →
`optimum-intel` 导出 → 校验 → 原子发布）。


## 改动清单

| 文件 | 改动 |
| --- | --- |
| `src/recall/backend.py` | **新增**。平台默认、后端选择、模型目录与嵌入器类解析 |
| `src/recall/mlx_embedding.py` | **新增**。MLX 上的 BERT 前向 + `MLXEmbedder` |
| `src/recall/config.py` | 新增 `MLX_MODEL_DIR`（`RECALL_MLX_MODEL_DIR`）与 `BACKEND`（`RECALL_BACKEND`） |
| `src/recall/daemon.py` | `Embedder` / `MODEL_DIR` 按后端解析（带无后端时的降级）；`whoami` 增加 `backend`；`supports()` 同时校验后端 |
| `src/recall/support.py` | 分块用的 tokenizer 跟随当前后端的模型目录 |
| `src/recall/setup.py` | 拆出 `install_mlx` / `install_openvino`；`recall setup` 按平台默认分发；新增 `--mlx` / `--openvino` / `--hub` / `--repo` / `--revision` / `--hf-ip` |
| `src/recall/cli.py` | 帮助文本补充后端与平台矩阵 |
| `pyproject.toml` | 新增 `mlx` extra；补 macOS 分类器 |
| `tests/test_backend.py` | **新增**。后端选择逻辑单测 |
| `tests/integration_mlx.py` | **新增**。MLX 真实模型冒烟测试 |
| `tests/verify_mlx_vs_torch.py` | **新增**。与 torch 参考实现的数值对照脚本 |
| `README.md` / `.gitignore` / `docs/backends.md` | 文档与忽略规则 |

上游的 `embedding.py`（OpenVINO 实现）与 `tests/integration_openvino.py` 未改动。

## 已知限制

- **MLX 后端只支持 BERT 结构**。`mlx_embedding._Bert` 按 `model_type == "bert"` 校验，
  换成别的架构会显式报错，而不是给出错误结果。
- **MLX 后端下 `--device` 不参与计算**。MLX 固定走 Metal，这个参数只是被原样记录，
  `RECALL_DEVICE=GPU` 这类 OpenVINO 设备名不会影响 MLX 的结果。
- **`_get_compiled()` 在 MLX 下是空实现**。OpenVINO 需要按静态形状预编译，MLX 没有
  这个概念，按批内实际最长序列 padding 即可；保留该方法只为满足 daemon 预热协议的
  接口要求。
- **OpenVINO 那条路没有 DoH 回退**。它走 `huggingface_hub`，需要 HF 本身可达
  （系统代理，或设置 `HF_ENDPOINT`）；DoH 回退只做在 MLX 的下载器里。
- **MLX 只在 Apple Silicon 上有轮子**，其它平台装 `.[mlx]` 会被 pip 直接拒绝——
  这是有意的，比静默装不上更清楚。
