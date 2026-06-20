#!/usr/bin/env bash
# === sano-floater-cleanup を GitHub の private リポジトリへ push（Linux/macOS）===
# gh CLI があれば作成+pushまで自動。無ければ手順を表示。
set -e
cd "$(dirname "$0")"

[ -d .git ] || git init
git add .
git commit -m "initial: sano-floater-cleanup (SOR pre-train cleanup + A/B verification + Gradio app)" || true
git branch -M main

if command -v gh >/dev/null 2>&1; then
    echo "[gh] private リポジトリを作成して push します..."
    gh repo create sano-floater-cleanup --private --source=. --remote=origin --push
else
    echo
    echo "gh CLI が見つかりません。以下を手動で実行してください:"
    echo "  1) GitHub で空の private リポジトリ 'sano-floater-cleanup' を作成"
    echo "  2) git remote add origin https://github.com/<your-account>/sano-floater-cleanup.git"
    echo "  3) git push -u origin main"
fi
