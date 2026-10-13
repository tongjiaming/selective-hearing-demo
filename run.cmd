@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo Please install Python 3.12 and run scripts\setup.ps1 -Models first.
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m selective_hearing
