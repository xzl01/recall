# recall

给 agent 使用的本地历史与代码检索工具：索引七个 agent 的会话历史（Pi、DeepSeek Harness、Codex、opencode、CodeBuddy、WorkBuddy、Claude Code）和选定文本目录，用 BGE 语义嵌入与 SQLite FTS5 混合检索，返回可追溯原文、稳定 ID 和相邻上下文。

检索不需要回答模型或 API key。初次模型安装按指令下载公开权重，日常检索在本地执行。历史索引默认放在 `~/.local/share/recall`，不在代码仓库里。

## 安装

一套代码、三套嵌入引擎。前两套按平台自动挑：**Apple Silicon 用 MLX（Metal），Intel / Linux 用
OpenVINO**；第三套 **GGUF（llama.cpp）** 跨平台但要显式指定。三者加载同一份
`BAAI/bge-small-zh-v1.5`，池化、维度、中文查询前缀都一致。

GGUF 通常是**量化**权重，向量空间和 fp32 不是同一个，所以它不会被自动选中；索引会记录
建立它的模型身份，换引擎或换量化档位需要重建索引。详见[后端说明](docs/backends.md)。

真实模型推理已在 Linux x86_64/OpenVINO CPU、GPU 上验证（NPU 依赖相应驱动与编译器）、
在 Apple Silicon/MLX 上对照 `transformers` 参考实现验证（最大绝对误差 8.0e-07），
以及 Apple Silicon/GGUF 上对照同一参考验证（f16 最大绝对误差 3.8e-04）。
当前不支持 Windows。

```bash
git clone https://github.com/xzl01/recall.git
cd recall
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[mlx]'        # Apple Silicon；Intel / Linux 用 '.[openvino]'；GGUF 用 '.[gguf]'
recall setup                   # 按平台默认装模型；--mlx / --openvino / --gguf 可强制

recall index sessions
recall index dir ./my-project --ext .py,.md,.ts
recall --chat --json "上次触摸手势怎么配置的"
recall --code --json --full "timeout handler"
recall --read STABLE_ID --json
```

如果已经有 BGE OpenVINO 模型目录（`tokenizer.json` 和 `ov/openvino_model.xml/.bin`），安装 `.[openvino]` 后执行 `recall setup --from /path/to/model`，可复用本地模型。

`recall index sessions` 会索引它能找到的每个 agent 的历史，各自一个语料（`sessions:pi`、`sessions:dsh`、`sessions:codex`、`sessions:opencode`、`sessions:codebuddy`、`sessions:workbuddy`、`sessions:claude`）：

| agent | 位置 | 格式 |
| --- | --- | --- |
| Pi | `~/.pi/agent/sessions` | 纯文本 JSONL |
| DeepSeek Harness | `~/.dsh/sessions` | zstd 压缩 JSONL 事件流 |
| Codex CLI | `~/.codex/sessions` | 纯文本 JSONL |
| opencode | `~/.local/share/opencode/opencode.db` | SQLite |
| CodeBuddy | `~/.codebuddy/projects` | 纯文本 JSONL |
| WorkBuddy | `~/.workbuddy/projects` | 纯文本 JSONL |
| Claude Code | `~/.claude/projects` | 纯文本 JSONL |

目录不存在的来源会打印一行「未找到 …，跳过」，所以没跑过某个 agent 的机器不需要配置。
`--chat` 搜全部 agent 历史（`sessions` 前缀），`--code` 搜已索引文本目录。也支持
`recall index codex` 这样只索引一个。

注意 **CodeBuddy IDE 的聊天记录在服务端**，本地只有 UI 状态，没有可索引的正文；
`~/.codebuddy/projects` 是 CodeBuddy CLI 写的位置。细节见 [docs/agent-history.md](docs/agent-history.md)。

## 检索规则

- 长记录按 448 BGE token、64 token 重叠分窗。
- 混合检索保留强关键词候选，避免融合排序把精确标识符挤出结果。
- 关键词最多保留 128 个不同词，过长时取首尾，避免问题后面的有效词被丢失。
- 默认补同一来源前后各 1 个窗口；会话跨消息时标注时间与角色，代码不跨原始记录。`--neighbors 0` 关闭补充。
- 默认每条最多 4000 字符，围绕命中正文分配字符预算；`--full` 取消上限。`--read` 默认读取同一原始记录前后各 2 个窗口。

JSON 中的 `score` 是相似度，`rank_score` 是融合分，均不是置信度。`selection_reason` 说明如何入选，`context_records` 保留邻居各自的位置、时间和角色。事实应由原文支持；版本冲突要核对时间及当前代码。

## 配置与常驻服务

默认设备是 CPU，可设置 `RECALL_DEVICE=GPU` 或 `--device GPU`。用 `recall daemon --device GPU --idle-exit 0` 在前台启动常驻服务；其他终端使用相同配置即可复用。服务未启动、设备或模型目录不同则回退到进程内推理。

`RECALL_HOME`、`RECALL_MODEL_DIR`、`RECALL_SESSIONS_DIR`、`RECALL_SOCKET` 可独立设置；CLI 也支持 `--home`、`--model-dir`、`--sessions-dir`。索引更新使用跨进程锁、暂存发布和配对恢复日志；不要单独删除数据库行或向量。删除的源记录会保留到对应语料 `--force` 重建。

通用 skill 在 [skills/recall](skills/recall/SKILL.md)，可复制到 harness 的 skill 目录。安装 skill 不等于安装程序或建立索引。

进一步说明：[英文完整用法](README.md)、[架构](docs/architecture.md)、[迁移](docs/migration.md)、[评测](docs/evaluation.md)、[验证](docs/validation.md)。MIT 许可证；仓库不包含私人会话、索引、模型权重或回答模型凭据。
