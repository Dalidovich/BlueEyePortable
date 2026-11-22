@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=%~dp0runtime\python\python.exe"
if not exist "%PYTHON_EXE%" (
    echo Portable Python runtime not found in "%~dp0runtime\python".
    echo Run Setup.bat on a computer with internet access first, then copy the whole folder.
    pause
    exit /b 1
)

set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONUTF8=1"
set "HF_HUB_OFFLINE=1"

"%PYTHON_EXE%" "%~dp0app\main.py" %*
if errorlevel 1 (
    echo.
    echo The application exited with an error. See output\logs\errors.log for details.
    pause
    exit /b 1
)
exit /b 0
