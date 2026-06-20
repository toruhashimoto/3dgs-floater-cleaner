@echo off
REM sano フローター低減クリーンアップ・アプリ起動（Windows）
cd /d "%~dp0"
if not exist ".venv" (
    echo [setup] creating venv...
    python -m venv .venv
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip >nul 2>&1
echo [setup] installing requirements...
pip install -r requirements.txt
echo [run] launching app at http://127.0.0.1:7860
python app\app.py
pause
