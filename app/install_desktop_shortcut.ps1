# Creates a Desktop shortcut to FloaterClean Trainer (run_desktop.bat).
# ASCII-only script body; the repo path (which may contain non-ASCII chars) is
# resolved at runtime via $PSScriptRoot, so it is never a literal in this file
# (avoids cp932 mangling under Windows PowerShell 5.1).
# Run:  powershell -ExecutionPolicy Bypass -File install_desktop_shortcut.ps1

$ErrorActionPreference = 'Stop'
$bat = Join-Path $PSScriptRoot 'run_desktop.bat'
if (-not (Test-Path $bat)) { throw "run_desktop.bat not found next to this script." }

$desktop = [Environment]::GetFolderPath('Desktop')
$lnkPath = Join-Path $desktop 'FloaterClean Trainer.lnk'

$ws = New-Object -ComObject WScript.Shell
$sc = $ws.CreateShortcut($lnkPath)
$sc.TargetPath = $bat
$sc.WorkingDirectory = $PSScriptRoot
$sc.WindowStyle = 1
$sc.Description = 'FloaterClean Trainer - low-floater 3DGS training (scale_reg)'
$sc.Save()

Write-Host "Shortcut created:" $lnkPath
