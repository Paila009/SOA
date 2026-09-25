$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runtimePath = Join-Path $projectRoot ".runtime"
$pythonExe = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python was not found at $pythonExe"
}
if (-not (Test-Path $runtimePath)) {
    throw "The project-local .runtime directory is missing."
}

$env:PYTHONPATH = "$runtimePath;$projectRoot"
$env:HALLUCINATION_PYTHON = $pythonExe
Set-Location $projectRoot

function global:python { & $env:HALLUCINATION_PYTHON @args }
Write-Host "Hallucination detection environment ready." -ForegroundColor Green
Write-Host "Project: $projectRoot"
Write-Host "Dashboard: .\run_dashboard.ps1"
Write-Host "Training:  .\run_training.ps1 -Type logistic"
