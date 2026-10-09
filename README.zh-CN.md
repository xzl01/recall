# recall

给 agent 使用的本地历史与代码检索工具：索引 Pi JSONL 会话和选定文本目录，用 BGE 语义嵌入与 SQLite FTS5 混合检索，返回可追溯原文、稳定 ID 和相邻上下文。

检索不需要回答模型或 API key。初次模型安装按指令下载公开权重，日常检索在本地执行。历史索引默认放在 `~/.local/share/recall`，不在代码仓库里。

## 安装

真实模型推理已在 Linux x86_64/OpenVINO CPU、GPU 上验证；NPU 依赖相应驱动与编译器。macOS 验证合成数据单元测试，未宣称验证模型推理。当前不支持 Windows。

```bash
git clone https://github.com/xzl01/recall.git
cd recall
python3 -m venv .venv
. .venv/bin/activate
pip install '.[setup]'
recall setup

recall index sessions
recall index dir ./my-project --ext .py,.md,.ts
recall --chat --json "上次触摸手势怎么配置的"
recall --code --json --full "timeout handler"
recall --read STABLE_ID --json
```

如果已经有 BGE OpenVINO 模型目录（`tokenizer.json` 和 `ov/openvino_model.xml/.bin`），安装 `.[openvino]` 后执行 `recall setup --from /path/to/model`，可复用本地模型。

`--chat` 只搜 Pi 历史，`--code` 搜已索引文本目录，其他 harness 的历史不会自动进入索引。可用 `--sessions-dir PATH` 指定 Pi 格式的合成数据或另一处会话目录。

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
