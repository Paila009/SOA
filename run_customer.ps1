$ErrorActionPreference = "Stop"
$customerRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $customerRoot
$customerPython = Join-Path $customerRoot ".venv-customer\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $customerPython)) {
    throw "Set up the isolated environment first. See docs/CUSTOMER_SETUP.md."
}
& $customerPython -m customer
