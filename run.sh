#!/usr/bin/env bash
# sano フローター低減クリーンアップ・アプリ起動（Linux/macOS）
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    echo "[setup] creating venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null 2>&1
echo "[setup] installing requirements..."
pip install -r requirements.txt
echo "[run] launching app at http://127.0.0.1:7860"
python app/app.py
