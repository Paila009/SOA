$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bundledPython = Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$localPython = Get-Command python -ErrorAction SilentlyContinue

Set-Location $projectRoot
if (Test-Path $bundledPython) {
    & $bundledPython "dashboard\server.py" --host 127.0.0.1 --port 8766
} elseif ($localPython -and $localPython.Source -notlike "*WindowsApps*") {
    & $localPython.Source "dashboard\server.py" --host 127.0.0.1 --port 8766
} else {
    throw "Python was not found. Install Python 3.10 or newer, then rerun this script."
}
