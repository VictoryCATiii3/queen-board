@echo off
setlocal EnableExtensions
REM ============================================================
REM  Queen Board - Windows launcher (CPU only)
REM  Double-click, or run from a terminal:  run.bat
REM  Everything installs into .venv in this folder. Nothing global.
REM
REM  Optional settings (set before running):
REM    set PORT=8000            set HOST=127.0.0.1
REM    set STOCKFISH_THREADS=4  set STOCKFISH_PATH=C:\path\to\stockfish.exe
REM    set SKIP_STOCKFISH=1     set NO_BROWSER=1
REM ============================================================
cd /d "%~dp0"

REM Never let pip touch the global Python, even by accident.
set "PIP_REQUIRE_VIRTUALENV=true"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

REM CPU only: hide GPUs from anything launched from here, and keep Queen off
REM (its official runner is CUDA-only and has no CPU mode).
set "CUDA_VISIBLE_DEVICES=-1"
set "QUEEN_DISABLE=1"

if not defined PORT set "PORT=8000"
if not defined HOST set "HOST=127.0.0.1"

REM ---- 1. Find Python 3.10+ -----------------------------------
set "PY="
where py >nul 2>&1 && (py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=py -3")
if not defined PY (
  where python >nul 2>&1 && (python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1 && set "PY=python")
)
if not defined PY (
  echo.
  echo  Python 3.10 or newer was not found.
  echo  Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  goto :fail
)

REM ---- 2. Virtual environment ---------------------------------
set "VPY=.venv\Scripts\python.exe"
if not exist "%VPY%" (
  echo Creating virtual environment in .venv ...
  %PY% -m venv .venv
  if errorlevel 1 goto :fail
)

REM Reinstall only when requirements.txt changes.
fc /b requirements.txt .venv\requirements.installed >nul 2>&1
if errorlevel 1 (
  echo Installing packages into .venv ...
  "%VPY%" -m pip install --upgrade pip >nul
  "%VPY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  copy /y requirements.txt .venv\requirements.installed >nul
)

REM ---- 3. Stockfish (CPU engine, downloaded into bin\stockfish) --
"%VPY%" tools\get_stockfish.py

REM ---- 4. Start ------------------------------------------------
echo.
echo  Queen Board is running at http://%HOST%:%PORT%
echo  Close this window or press Ctrl+C to stop.
echo.
if not "%NO_BROWSER%"=="1" start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://%HOST%:%PORT%'"

REM No --reload: it switches uvicorn to an event loop that can't start
REM subprocesses on Windows, which would break Stockfish.
"%VPY%" -m uvicorn server.app:app --host %HOST% --port %PORT%
if errorlevel 1 goto :fail
exit /b 0

:fail
echo.
echo  Something went wrong (see above).
pause
exit /b 1
