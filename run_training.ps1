param(
    [ValidateSet("entropy-only", "logistic", "mlp")]
    [string]$Type = "logistic"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$runtimePath = Join-Path $projectRoot ".runtime"

if (-not (Test-Path $bundledPython)) {
    throw "Bundled Python was not found at $bundledPython"
}
if (-not (Test-Path $runtimePath)) {
    throw "Project runtime is missing. Install requirements before training."
}

Set-Location $projectRoot
$env:PYTHONPATH = "$runtimePath;$projectRoot"
& $bundledPython "scripts\train_probe.py" --type $Type
