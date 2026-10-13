@echo off
rem eNegmatic launcher - sets up a local Python environment on first run.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python environment ^(first run only^)...
    python -m venv .venv || goto :err
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :err
)
".venv\Scripts\python.exe" -c "import rawpy, cv2, PySide6" 2>nul || ".venv\Scripts\python.exe" -m pip install -r requirements.txt
start "" ".venv\Scripts\pythonw.exe" main.py %*
exit /b 0

:err
echo.
echo Setup failed. Make sure Python 3.10+ is installed and on PATH.
pause
exit /b 1
