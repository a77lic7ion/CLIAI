@echo off
setlocal

REM Build a Windows EXE for NetIntelliX with bundled assets
REM Requirements: Python 3.11+, pip installed

echo Installing build dependencies (PyInstaller)...
python -m pip install --upgrade pip >nul 2>&1
python -m pip install pyinstaller >nul 2>&1

echo Building EXE...
pyinstaller --onefile --windowed --name NetIntelliX app_gui.py ^
  --add-data "icons;icons" ^
  --add-data "sites.json;." ^
  --add-data "netintellix_logo.svg;." ^
  --hidden-import serial.tools.list_ports ^
  --hidden-import PIL.ImageTk

if %ERRORLEVEL% NEQ 0 (
  echo Build failed.
  exit /b %ERRORLEVEL%
)

echo Build complete. Find EXE at dist\NetIntelliX.exe
endlocal