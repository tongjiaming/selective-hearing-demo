param(
    [switch]$Models,
    [string]$IndexUrl = 'https://pypi.org/simple',
    [string]$HfEndpoint = 'https://huggingface.co',
    [string]$RuntimeRoot
)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    # A short environment path avoids Windows' legacy 260-character limit in torch headers.
    if (Test-Path '.venv') { throw '已有不完整 .venv，请先保留或移走该目录，再重新安装。' }
    $repoBytes = [Text.Encoding]::UTF8.GetBytes((Get-Location).Path)
    $hashProvider = [Security.Cryptography.SHA256]::Create()
    try { $repoKey = ([BitConverter]::ToString($hashProvider.ComputeHash($repoBytes))).Replace('-', '').Substring(0, 8) }
    finally { $hashProvider.Dispose() }
    if (-not $RuntimeRoot) {
        $documents = [Environment]::GetFolderPath('MyDocuments')
        $RuntimeRoot = Join-Path $documents "Codex\sh-runtime-$repoKey"
    }
    $runtimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
    py -3.12 -m venv $runtimeRoot
    if ($LASTEXITCODE -ne 0) { throw '需要安装 Python 3.12' }
    New-Item -ItemType Junction -Path '.venv' -Target $runtimeRoot | Out-Null
}
$environmentItem = Get-Item -LiteralPath '.venv'
$runtimeRoot = if ($environmentItem.LinkType -eq 'Junction') { [string]@($environmentItem.Target)[0] } else { $environmentItem.FullName }
$runtimePython = Join-Path $runtimeRoot 'Scripts\python.exe'
& $runtimePython -m pip install -r requirements.txt --index-url $IndexUrl
if ($LASTEXITCODE -ne 0) { throw '基础依赖安装失败' }
if ($Models) {
    & $runtimePython -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'PyTorch 安装失败' }
    & $runtimePython -m pip install -r requirements-models.txt --index-url $IndexUrl
    if ($LASTEXITCODE -ne 0) { throw '模型依赖安装失败' }
    & $runtimePython scripts/download_models.py --hf-endpoint $HfEndpoint
    if ($LASTEXITCODE -ne 0) { throw '模型下载失败；可查阅 README 的手动下载方法' }
}
Write-Host '安装完成。运行 .\run.ps1 打开界面。'
Write-Host "运行环境：$runtimeRoot"
