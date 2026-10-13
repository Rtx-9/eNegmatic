@echo off
rem Builds dist\eNegmatic\eNegmatic.exe and dist\eNegmatic-windows.zip
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv || goto :err
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt pyinstaller || goto :err
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean packaging\enegmatic.spec || goto :err
powershell -NoProfile -Command "Compress-Archive -Path 'dist\eNegmatic' -DestinationPath 'dist\eNegmatic-windows.zip' -Force" || goto :err
echo.
echo Done: dist\eNegmatic\eNegmatic.exe  and  dist\eNegmatic-windows.zip
exit /b 0

:err
echo.
echo Build failed.
pause
exit /b 1
