@echo off
setlocal
cd /d "%~dp0"
echo === DesktopFBT build ===

set "PY="
where py >nul 2>nul && set "PY=py -3.11"
if not defined PY ( where python >nul 2>nul && set "PY=python" )
if not defined PY (
  echo Python was not found.
  echo The easiest way to get the EXE is GitHub Actions - see README.
  echo To build on this PC, install Python 3.11 from https://www.python.org/ ^(check "Add to PATH"^).
  pause
  exit /b 1
)

if not exist .venv-build (
  %PY% -m venv .venv-build || python -m venv .venv-build || goto :fail
)
call .venv-build\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt || goto :fail
pip install --no-deps rtmlib onnxconverter-common || goto :fail
pip install "pyinstaller>=6.3" || goto :fail

if not exist models_bundled\agformer_base.onnx (
  mkdir models_bundled 2>nul
  git fetch -q --depth 1 origin models
  for %%f in (motionbert_lite.onnx agformer_base.onnx agformer_small.onnx) do git show FETCH_HEAD:%%f > models_bundled\%%f
)
pyinstaller --noconfirm DesktopFBT.spec || goto :fail

set "REL=dev"
if exist RELEASE set /p REL=<RELEASE
powershell -NoProfile -Command "Compress-Archive -Path 'dist\DesktopFBT' -DestinationPath 'dist\DesktopFBT_%REL%.zip' -Force"

echo.
echo Done:
echo   dist\DesktopFBT\DesktopFBT.exe
echo   dist\DesktopFBT_%REL%.zip
pause
exit /b 0

:fail
echo.
echo Build failed. See the messages above.
pause
exit /b 1
