@echo off
setlocal
if not exist "%~dp0run.ps1" (
  echo Project launcher not found: "%~dp0run.ps1"
  echo Use the local Start-Demo shortcut, or run this file inside the complete project.
  pause
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
  echo Launch failed. See the error above for the exact missing path.
  pause
  exit /b 1
)
