@echo off
set "PROJECT_ROOT=%~dp0"
set "PYTHON_ROOT=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python"
set "PYTHONPATH=%PROJECT_ROOT%.runtime;%PROJECT_ROOT%"
set "PATH=%PYTHON_ROOT%;%PATH%"
cd /d "%PROJECT_ROOT%"
echo Hallucination detection environment ready.
echo Dashboard: run_dashboard.ps1
echo Training:  run_training.ps1 -Type logistic
