#!/usr/bin/env bash
#
# 把仓库里的 recall skill 安装到各 agent 的 skills 目录。
#
# 用复制而不是符号链接：这个仓库常常放在外置卷上（/Volumes/...），卷一卸载符号链接
# 就断了，agent 读到的是坏链接。代价是仓库更新后要重跑一次，用 --force 覆盖旧副本。
#
# 用法:
#   ./skills/install.sh            只装到还不存在的位置，已存在的一律不动
#   ./skills/install.sh --force    覆盖已有副本（skill 内容更新后用这个）
#   ./skills/install.sh --list     只列目标与状态，不写任何东西
#
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/recall"

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
for arg in "$@"; do
  case "$arg" in
    --force | -f) force=1 ;;
    --list | -l) list=1 ;;
    -h | --help)
      sed -n '2,13p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "未知参数: $arg" >&2
      exit 2
      ;;
  esac
done

[ -f "${src}/SKILL.md" ] || {
  echo "找不到 ${src}/SKILL.md" >&2
  exit 1
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

  if [ "$list" = 1 ]; then
    if [ -e "${d}/recall" ]; then echo "  已存在  ${shown}"; else echo "  可安装  ${shown}"; fi
    continue
  fi

  if [ -e "${d}/recall" ] && [ "$force" != 1 ]; then
    echo "  已存在，未改动  ${shown}（更新请加 --force）"
    skipped=$((skipped + 1))
    continue
  fi

  rm -rf "${d}/recall"
  cp -R "$src" "${d}/recall"
  echo "  已安装  ${shown}"
  installed=$((installed + 1))
done

if [ "$list" = 0 ]; then
  echo
  echo "安装 ${installed}，跳过 ${skipped}，目标目录不存在 ${absent}"
fi
