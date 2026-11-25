@echo off
cd /d "%~dp0"
if not exist .venv-build (
  echo Run build.bat once first (it creates the Python environment).
  pause
  exit /b 1
)
call .venv-build\Scripts\activate.bat
python app.py %*
