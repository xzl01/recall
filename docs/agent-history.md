# Agent 历史：六个来源

`recall index sessions` 会读取它能找到的每个 agent 的历史，各自写进一个独立语料：

| agent | 默认位置 | 格式 | 语料名 | 路径覆盖 |
| --- | --- | --- | --- | --- |
| Pi | `~/.pi/agent/sessions` | 纯文本 JSONL | `sessions:pi` | `RECALL_SESSIONS_DIR` |
| DeepSeek Harness | `~/.dsh/sessions` | zstd 压缩 JSONL 事件流 | `sessions:dsh` | `RECALL_DSH_SESSIONS_DIR` |
| Codex CLI | `~/.codex/sessions` | 纯文本 JSONL | `sessions:codex` | `RECALL_CODEX_SESSIONS_DIR` |
| opencode | `~/.local/share/opencode/opencode.db` | SQLite | `sessions:opencode` | `RECALL_OPENCODE_DB` |
| CodeBuddy | `~/.codebuddy/projects` | 纯文本 JSONL | `sessions:codebuddy` | `RECALL_CODEBUDDY_SESSIONS_DIR` |
| WorkBuddy | `~/.workbuddy/projects` | 纯文本 JSONL | `sessions:workbuddy` | `RECALL_WORKBUDDY_SESSIONS_DIR` |

```bash
recall index sessions          # 全部
recall index codex             # 只要一个
recall index dsh --force       # 重建该语料
recall "上次那个超时怎么配的" --chat          # 所有语料一起检索
recall search "keepalive" --corpus sessions:codex
```

`--chat` 用的是 `sessions%` 前缀，所以新增 agent 的语料会自动被它覆盖，不用改检索侧。
目录不存在的来源会打印一行「未找到 …，跳过」，所以从没跑过某个 agent 的机器不需要任何配置。

## CodeBuddy 与 WorkBuddy

腾讯这两个产品**共用同一套本地写入器**（会话文件里带一个 `__codebuddyLocal` 字段），
所以是同一个格式、同一个读取器，只是根目录不同。

但要注意它们**本地留存的东西不一样**：

| | 本地有对话正文吗 | 说明 |
| --- | --- | --- |
| WorkBuddy | **有** | `~/.workbuddy/projects/<项目>/<会话>.jsonl`，纯 JSONL |
| CodeBuddy CLI | 会有（同一格式） | 写 `~/.codebuddy/projects` |
| CodeBuddy IDE | **没有** | 聊天记录在服务端；本地只剩 UI 状态、输入草稿、todos、plans |

也就是说：只装过 **CodeBuddy IDE** 的机器上 `sessions:codebuddy` 会是空的，这不是读取器
的问题 —— 盘上确实没有正文。实测 `~/Library/Application Support/CodeBuddy CN/` 里最大的
一条 `state.vscdb` 记录是产品配置缓存（77 KB），`genie-history/*/conversations/` 全是空目录，
`chat.ChatSessionStore.index` 是 `{"version":1,"entries":{}}`。

## 只索引对话正文

四个来源都**只取 user / assistant 的文本**，不索引工具调用与工具输出。理由：
工具结果又大又是噪声，一次 `ls -R` 或构建日志就能产出上千个窗口，把索引撑爆而
检索价值很低；而助手自己的回复通常已经把结论写清楚了。此外单条记录超过
**200,000 字符**会被跳过（数量会打印出来）。

这是刻意的取舍，不是漏掉。需要工具输出的话，用 `recall index dir` 把相关日志目录
当普通文本索引。

## 三条上限，以及为什么需要它们

| 上限 | 值 | 位置 |
| --- | --- | --- |
| 单行 JSON | 1 MiB（在 `json.loads` **之前**就跳过） | `sources.MAX_LINE_BYTES` |
| 单条记录正文 | 200,000 字符 | `sources.MAX_RECORD_CHARS` |
| 读取缓冲区 | 1 MiB 分块，内存与文件大小无关 | `sources._bounded_lines` |

这些不是过早优化，是实测踩出来的。一台机器上的真实历史：

| agent | 记录数 | 正文 | 说明 |
| --- | --- | --- | --- |
| codex | 15,665 | 31.9 M字符 | 原始 **3.77 GB / 31 万行**，其中 487 行 >1 MiB |
| dsh | 3,959 | 8.8 M字符 | 97 个 zstd 会话 |
| opencode | 1,812 | 0.8 M字符 | SQLite 里 15,522 个 part，只有 1,932 个是 text |
| workbuddy | 1,454 | 0.8 M字符 | 18 个会话文件 / 56.5 MB |
| codebuddy | 0 | — | IDE 不落盘，见上文 |
| pi | 0 | — | 这台机器没装 Pi |

两个具体的坑：

1. **单行上限必须在 `json.loads` 之前生效。** 那些超长行是工具输出、图片 base64
   和加密推理块 —— 把它们解析成对象图会把内存顶到 GB 级，而它们要么不在我们读取
   的字段里，要么正文超过记录上限本来就会被丢弃。
2. **逐行读取必须用二进制分块，不能用文本模式迭代。** 实测一个 **986 MB** 的
   Codex 会话文件：

   | 读法 | 峰值 RSS |
   | --- | --- |
   | `for line in open(errors="replace")` | **990 MB** |
   | `open("rb")` + 1 MiB 分块自己切行 | **21 MB** |

   Python 文本模式逐行迭代在这个文件上几乎按文件大小吃内存（接近 1:1）。

改之前，只**读取**（还没嵌入）四个 agent 的历史峰值 **2061 MB**；改成有界读取后是
**108 MB**。超长行与超长记录都会在读完后打印数量，不会静默丢弃。

索引体积也要有心理准备：正文被切成 448 token、重叠 64 的窗口，每个窗口一个 512 维
float32 向量（2 KB）。上面那份历史约 4.7 万个窗口，对应约 95 MB 向量，加上 SQLite
里的正文，整个索引在百 MB 量级。只想索引一部分时用 `recall index <agent>`。

## 各来源的事件映射

### Pi

```json
{"type": "message", "timestamp": "...", "message": {"role": "user",
 "content": [{"type": "text", "text": "..."}]}}
```

- 取 `type == "message"`，拼 `message.content[]` 里带 `text` 的块
- role 用 `message.role`；记录前缀 `[<session-id 前 8 位>]`

### DeepSeek Harness

会话是 `session.v4.jsonl.zstd`：**zstd 压缩**的 JSONL 事件流，事件种类很多
（`step/start`、`tool/call`、`tool/result`、`assistant/message` …）。只取两类：

| 事件 | 正文位置 |
| --- | --- |
| `user/message` | `data.content[]` |
| `assistant/message` | `data.message.content[]` |

- 时间戳是 `time`（epoch 毫秒），统一转成 ISO 字符串
- 解压需要 zstd：Python 3.14+ 自带 `compression.zstd`，更早版本 `pip install '.[dsh]'`
- DSH 是**流式**写出的，帧头常常没有 content size。所以解压必须走流式接口 ——
  要求已知大小的 `decompress()` 会直接报
  `could not determine content size in frame header`（实测踩过）

### Codex CLI

`~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`，每行 `{type, timestamp, ordinal, payload}`：

- 取 `type == "response_item"` 且 `payload.role` 为 `user` / `assistant`
- 正文来自 `payload.content[]` 的 `text` 字段
- 明确跳过 `developer`（系统提示）、`reasoning`（思维链）、`function_call`，
  以及 `event_msg` / `turn_context` / `token_usage_record` 等非对话事件

### opencode

opencode 已经迁到 SQLite：`~/.local/share/opencode/opencode.db`（旧的
`storage/{session,message,part}` 目录现在是空的）。相关表：

| 表 | 作用 |
| --- | --- |
| `session` | `id`、`slug`、`title`、`directory` |
| `message` | `id`、`session_id`、`data`（JSON，含 `role`） |
| `part` | `id`、`message_id`、`data`（JSON，`type` 为 `text`/`tool`/`reasoning`/`step-*`） |

取 `part.data.type == "text"` 的条目，`role` 从 `message.data.role` 关联，
时间用 `part.time_created`（epoch 毫秒）。来源标识用 `库名:会话 slug`（例如
`opencode:lucky-comet`），因为这里没有文件路径。

数据库以**只读**方式打开（`mode=ro`），不会干扰正在运行的 opencode。

### CodeBuddy / WorkBuddy

两个产品落盘布局一致：`<根>/<编码后的项目路径>/<会话 uuid>.jsonl`，一个会话一个文件，
纯 JSONL。每行是一个事件，**只有 `type == "message"` 是对话正文**，而且正文在**顶层**
（不是像 pi 那样嵌在 `message` 里）：

```json
{"type": "message", "role": "user", "timestamp": 1789543534454,
 "sessionId": "...", "cwd": "...",
 "content": [{"type": "input_text", "text": "..."}]}
```

| 字段 | 说明 |
| --- | --- |
| `role` | `user`（块类型 `input_text`）/ `assistant`（块类型 `output_text`） |
| `content` | 块数组，取每个块的 `text` |
| `timestamp` | **epoch 毫秒**，统一转成 ISO 字符串 |

同一个文件里还有 `function_call`、`function_call_result`、`reasoning`、
`file-history-snapshot`、`ai-title` —— 全部跳过（工具调用与思维链，理由同其它 agent）。

实测一台机器上 WorkBuddy 是 18 个会话文件 / 56.5 MB / 1454 条正文。

## 兼容旧索引

老版本把 Pi 会话存在一个裸 `sessions` 语料里。现在索引 Pi 时会**就地改名**为
`sessions:pi`，不需要重跑索引：

```
已把 214 条旧的 sessions 记录迁移为 sessions:pi
```

改名只在本次确实要索引 Pi（`recall index sessions` 或 `recall index pi`）时发生。

## 加一个新 agent

1. 在 `src/recall/sources.py` 写一个 `iter_<name>_file(path)`，yield `(role, ts, text)`
2. 加进 `AGENTS` 和 `LABELS`
3. 在 `iter_records` 里接上（文件型走 `_jsonl_files`，非文件型参照 opencode 的分支）
4. 在 `config.py` 加路径常量，在 `engine.SESSION_ROOTS` 注册
5. **三处测试隔离都要加上新根目录**：`tests/test_engine.py` 的 `setUp`、
   `tests/test_cli.py` 的 `setUp`、以及 `tests/integration_mlx.py` /
   `tests/integration_gguf.py` 的 `env`。漏掉任何一处，`index sessions` 就会去索引
   开发机上**真实的**历史 —— 又慢又读了私有数据。这个坑踩过两次。

共用格式的来源（像 CodeBuddy / WorkBuddy）不需要新读取器，在 `AGENTS` 里加一个名字、
在 `iter_records` 的 `readers` 字典里指向同一个函数即可。

`tests/test_sources.py` 每个来源都有一个用合成数据的最小用例，照着加即可。
