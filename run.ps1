param([switch]$Check, [switch]$FileTest)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$entryPoint = Join-Path $projectRoot 'selective_hearing\__main__.py'
if (-not (Test-Path -LiteralPath $entryPoint)) {
    throw "项目文件缺失：$entryPoint。请从完整项目目录启动。"
}
$runtimeRoot = Join-Path $projectRoot '.venv'
if (Test-Path -LiteralPath $runtimeRoot) {
    $environmentItem = Get-Item -LiteralPath $runtimeRoot
    if ($environmentItem.LinkType -eq 'Junction') {
        $runtimeRoot = [string]@($environmentItem.Target)[0]
    }
}
$python = Join-Path $runtimeRoot 'Scripts\python.exe'
$pythonw = Join-Path $runtimeRoot 'Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $python) -or -not (Test-Path -LiteralPath $pythonw)) {
    throw "项目运行环境缺失：$runtimeRoot。此提示不代表电脑没有 Python；请在项目目录运行 scripts\setup.ps1 -Models。"
}
if ($Check) {
    Write-Output "Project: $projectRoot"
    Write-Output "Runtime: $runtimeRoot"
    & $python --version
    exit $LASTEXITCODE
}
# Resolve paths before starting, regardless of Explorer / shell working directory.
$module = if ($FileTest) { 'selective_hearing.file_ui' } else { 'selective_hearing' }
Start-Process -FilePath $pythonw -ArgumentList '-m', $module -WorkingDirectory $projectRoot
