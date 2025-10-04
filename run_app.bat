@echo off
setlocal ENABLEEXTENSIONS

REM Change to the directory containing this script
cd /d "%~dp0"

REM Activate virtual environment if present
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
if exist "venv\Scripts\activate.bat" call "venv\Scripts\activate.bat"

REM Prefer 'py' launcher if available; otherwise use 'python'
set "PYEXE=python"
where py >nul 2>&1 && set "PYEXE=py"

REM Basic sanity check that Python is available
%PYEXE% --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo [Error] Python is not available on PATH.
  echo Install Python 3 or add it to PATH, then re-run this script.
  pause
  exit /b 1
)

REM Launch the GUI application
%PYEXE% "app_gui.py"

REM Keep the terminal window open after exit
echo.
echo [Info] Application exited. Press any key to close this window.
pause >nul