@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo  BlueEye Portable Tester - setup
echo ============================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup\bootstrap.ps1"
if errorlevel 1 goto :failed

echo.
echo ==^> Downloading / verifying the Blue-Eye model
"%~dp0runtime\python\python.exe" "%~dp0app\model_manager.py" download
if errorlevel 1 goto :failed

echo.
echo Setup complete. Launch the application with Start.bat
pause
exit /b 0

:failed
echo.
echo Setup failed. Fix the error above and run Setup.bat again; completed steps are skipped.
pause
exit /b 1
