@echo off
REM ============================================================
REM  3DGS Floater Cleaner launcher (double-click to start)
REM  ASCII-only on purpose: avoids cp932 mojibake on JP Windows.
REM  Uses a local venv (tkinter comes from the base Python).
REM ============================================================
chcp 65001 >nul
setlocal
cd /d "%~dp0"

set "VENV=%~dp0.venv_desktop"
if not exist "%VENV%\Scripts\python.exe" (
  echo [setup] creating venv ...
  py -3 -m venv "%VENV%" 2>nul || python -m venv "%VENV%"
)
if not exist "%VENV%\Scripts\python.exe" (
  echo [error] could not create a Python venv. Install python.org Python 3.x and retry.
  pause
  exit /b 1
)
call "%VENV%\Scripts\activate.bat"

python -c "import tkinter" 2>nul
if errorlevel 1 (
  echo [error] tkinter is not available in this Python. Install the standard python.org build ^(it bundles Tk^).
  pause
  exit /b 1
)

python -c "import numpy, scipy" 2>nul
if errorlevel 1 (
  echo [setup] installing numpy/scipy for floater measurement ...
  python -m pip install --quiet --disable-pip-version-check numpy scipy
)

echo [run] 3DGS Floater Cleaner ...
python "%~dp0desktop_app.py"
set "RC=%errorlevel%"
endlocal
if not "%RC%"=="0" pause
exit /b %RC%
