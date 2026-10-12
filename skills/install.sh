#!/usr/bin/env bash
#
# 安装 recall skill，并可选把 recall 可执行文件挂到 PATH。
#
# 用法:
#   ./skills/install.sh                  只装 skill 到还不存在的位置，已存在的一律不动
#   ./skills/install.sh --force          覆盖已有 skill 副本（skill 内容更新后用这个）
#   ./skills/install.sh --shim           额外生成 <bin-dir>/recall 转发脚本，让 PATH 能找到
#   ./skills/install.sh --shim --force   覆盖已存在的转发脚本
#   ./skills/install.sh --bin-dir DIR    转发脚本写到 DIR（默认 ~/.local/bin 或 $RECALL_BIN_DIR）
#   ./skills/install.sh --list           只列目标与状态，不写任何东西
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 仓库根 = skills/ 的上一级。转发脚本里的路径由它推导，所以没有写死的占位符要人改。
repo="$(cd "${here}/.." && pwd)"
src="${here}/recall"

# 顺序即输出顺序。~/.agents/skills 是全局目录，DSH 从这里读。
targets=(
  "$HOME/.agents/skills"     # 全局（DSH）
  "$HOME/.codex/skills"      # Codex CLI
  "$HOME/.claude/skills"     # Claude Code
  "$HOME/.workbuddy/skills"  # WorkBuddy
  "$HOME/.codebuddy/skills"  # CodeBuddy
)

force=0
list=0
shim=0
bin_dir="${RECALL_BIN_DIR:-$HOME/.local/bin}"

while [ $# -gt 0 ]; do
  case "$1" in
    --force | -f) force=1 ;;
    --list | -l) list=1 ;;
    --shim) shim=1 ;;
    --bin-dir)
      [ $# -ge 2 ] || {
        echo "--bin-dir 需要一个参数" >&2
        exit 2
      }
      bin_dir="$2"
      shift
      ;;
    -h | --help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      exit 2
      ;;
  esac
  shift
done

[ -f "${src}/SKILL.md" ] || {
  echo "找不到 ${src}/SKILL.md" >&2
  exit 1
}

# 把路径写成 shell 单引号字面量：路径里有空格或引号也不会把生成的脚本弄坏。
quote() { printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"; }

# 转发脚本：检出可能在可移除卷上，所以先判断再 exec。直接转发的话，
# "卷没挂载" 会表现成 command not found —— 看着像没装，是另一个问题。
emit_shim() {
  cat <<'SHIM'
#!/bin/sh
# 由 skills/install.sh 生成 —— 不要手改，重跑 install.sh --shim --force 即可。
#
# 转发到检出里的 venv，让 PATH 上的 recall 指向这份代码。
set -e
SHIM
  # shellcheck disable=SC2016  # 要的就是字面量，不要在这里展开
  printf 'REPO=${RECALL_REPO:-%s}\n' "$(quote "${repo}")"
  cat <<'SHIM'
BIN="$REPO/.venv/bin/recall"

if [ ! -x "$BIN" ]; then
  echo "recall: 找不到可执行的 $BIN" >&2
  if [ ! -d "$REPO" ]; then
    echo "recall: 检出目录不存在 —— 外置卷或网络盘没挂载？" >&2
    echo "recall: 或者用 RECALL_REPO 指向实际位置。" >&2
  else
    echo "recall: 检出在，但 venv 不完整。重建：" >&2
    echo "recall:   cd $REPO && uv venv && uv pip install -e '.[mlx,gguf,dsh]'" >&2
  fi
  exit 127
fi

exec "$BIN" "$@"
SHIM
}

installed=0
skipped=0
absent=0

for d in "${targets[@]}"; do
  shown="${d/#$HOME/~}/recall"

  # 删除前先确认路径确实是 <HOME>/*/skills/recall，别让变量出错时误删别的东西。
  case "${d}/recall" in
    "$HOME"/*/skills/recall) ;;
    *)
      echo "  跳过（路径不符合预期）: ${d}/recall" >&2
      continue
      ;;
  esac

  if [ ! -d "${d}" ]; then
    echo "  目标目录不存在，跳过  ${shown}"
    absent=$((absent + 1))
    continue
  fi

  if [ "${list}" = 1 ]; then
    if [ -e "${d}/recall" ]; then echo "  已存在  ${shown}"; else echo "  可安装  ${shown}"; fi
    continue
  fi

  if [ -e "${d}/recall" ] && [ "${force}" != 1 ]; then
    echo "  已存在，未改动  ${shown}（更新请加 --force）"
    skipped=$((skipped + 1))
    continue
  fi

  rm -rf "${d}/recall"
  cp -R "${src}" "${d}/recall"
  echo "  已安装  ${shown}"
  installed=$((installed + 1))
done

# ── PATH 转发脚本 ────────────────────────────────────────────────
if [ "${shim}" = 1 ]; then
  dest="${bin_dir}/recall"
  shown="${dest/#$HOME/~}"

  if [ "${list}" = 1 ]; then
    if [ -e "${dest}" ]; then echo "  已存在  ${shown}"; else echo "  可安装  ${shown}"; fi
  elif [ ! -d "${bin_dir}" ]; then
    echo "  跳过转发脚本（目录不存在）: ${shown}" >&2
  else
    tmp="$(mktemp)"
    emit_shim >"${tmp}"
    if [ -e "${dest}" ] && [ "${force}" != 1 ]; then
      if cmp -s "${tmp}" "${dest}"; then
        echo "  已是最新  ${shown}"
      else
        echo "  已存在且内容不同，未改动  ${shown}（覆盖请加 --force）"
        skipped=$((skipped + 1))
      fi
      rm -f "${tmp}"
    else
      mv "${tmp}" "${dest}"
      chmod +x "${dest}"
      echo "  已安装  ${shown}  →  ${repo}/.venv/bin/recall"
      installed=$((installed + 1))
    fi
  fi
fi

if [ "${list}" = 0 ]; then
  echo
  echo "安装 ${installed}，跳过 ${skipped}，目标目录不存在 ${absent}"
  if [ "${shim}" != 1 ]; then
    echo "（没有生成 PATH 转发脚本；需要的话加 --shim）"
  fi
fi
