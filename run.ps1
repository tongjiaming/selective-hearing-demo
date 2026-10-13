$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    throw '请先运行 powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Models'
}
& .\.venv\Scripts\python.exe -m selective_hearing
