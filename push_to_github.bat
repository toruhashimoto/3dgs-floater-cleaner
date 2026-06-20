@echo off
REM === sano-floater-cleanup を GitHub の private リポジトリへ push（Windows）===
REM 事前に Git（同梱の Git Bash 等）が必要。gh CLI があれば作成+pushまで自動。
cd /d "%~dp0"

if not exist ".git" git init
git add .
git commit -m "initial: sano-floater-cleanup (SOR pre-train cleanup + A/B verification + Gradio app)"
git branch -M main

where gh >nul 2>&1
if %errorlevel%==0 (
    echo [gh] private リポジトリを作成して push します...
    gh repo create sano-floater-cleanup --private --source=. --remote=origin --push
) else (
    echo.
    echo gh CLI が見つかりません。以下を手動で実行してください:
    echo   1^) GitHub で空の private リポジトリ "sano-floater-cleanup" を作成
    echo   2^) git remote add origin https://github.com/^<your-account^>/sano-floater-cleanup.git
    echo   3^) git push -u origin main
)
pause
