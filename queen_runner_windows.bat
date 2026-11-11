@echo off
setlocal EnableExtensions DisableDelayedExpansion
title Queen Board
cd /d "%~dp0"
REM =============================================================================
REM  Queen Board (Windows) -- play chess against QUEEN and see it explain its moves.
REM
REM  Run it: double-click this file, or run  queen_runner_windows.bat  in a terminal.
REM
REM  First run sets everything up inside this folder (.venv), downloads Stockfish
REM  and the QUEEN model (~8.4 GB, into the standard Hugging Face cache), then
REM  opens the board in your browser. Later runs start in seconds.
REM  Nothing is installed system-wide. Close the window or press Ctrl+C to stop.
REM
REM  Settings live in setup\settings.yaml (cpu, model, quantize, port, ...).
REM  It picks how to run QUEEN from your hardware: NVIDIA GPU with 10 GB+ at full
REM  size, 5.5-10 GB shrunk to 4-bit (nf4), otherwise the CPU (slow).
REM
REM  UNTESTED on real Windows hardware so far. Please report problems.
REM =============================================================================

set "PIP_REQUIRE_VIRTUALENV=true"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

REM ---- 1. Find Python 3.10+ ---------------------------------------------------
set "PY="
if defined PYTHON set "PY=%PYTHON%"
if not defined PY call :try_py py -3.13
if not defined PY call :try_py py -3.12
if not defined PY call :try_py py -3.11
if not defined PY call :try_py py -3.10
if not defined PY call :try_py python
if not defined PY call :try_py python3
if defined PY goto :py_ok

echo.
echo ==^> No Python 3.10+ found. Fetching a private Python into .tools\ (first run only)
if exist ".tools\uv.exe" goto :have_uv
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:UV_INSTALL_DIR = Join-Path (Get-Location) '.tools'; $env:UV_NO_MODIFY_PATH = '1'; irm https://astral.sh/uv/install.ps1 | iex"
if not exist ".tools\uv.exe" goto :fail_python
:have_uv
set "UV_PYTHON_INSTALL_DIR=%CD%\.tools\python"
set "UV_CACHE_DIR=%CD%\.tools\cache"
".tools\uv.exe" python install 3.12
if errorlevel 1 goto :fail_python
for /f "usebackq delims=" %%p in (`".tools\uv.exe" python find 3.12`) do set "PY=%%p"
if not defined PY goto :fail_python
:py_ok

REM ---- Settings from setup\settings.yaml (values set in this window still win) ---
for /f "usebackq tokens=1* delims==" %%a in (`"%PY%" tools\settings.py --cmd`) do set "%%a=%%b"
if /i "%QUEEN_DEVICE%"=="cpu" echo Settings: running QUEEN on the CPU, because cpu: true is set.
if not defined PORT set "PORT=8000"
if not defined HOST set "HOST=127.0.0.1"
set "URL=http://%HOST%:%PORT%"

REM ---- 2. Private environment in .venv ----------------------------------------
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :venv_ok
echo.
echo ==^> Creating a private Python environment in .venv (first run only)
"%PY%" -m venv .venv
if errorlevel 1 goto :fail_venv
:venv_ok

REM Which PyTorch: none (QUEEN on another machine), NVIDIA (cuda), or CPU.
set "FLAVOR=none"
if defined QUEEN_URL goto :flavor_done
set "FLAVOR=cpu"
if /i "%QUEEN_DEVICE%"=="cpu" goto :flavor_done
where nvidia-smi >nul 2>&1
if not errorlevel 1 set "FLAVOR=cuda"
:flavor_done

set "STAMP="
for /f "usebackq delims=" %%h in (`%VPY% -c "import hashlib; print(hashlib.sha256(open('pyproject.toml','rb').read()).hexdigest()[:16])"`) do set "STAMP=%%h-%FLAVOR%"
set "OLDSTAMP="
if exist ".venv\.installed" set /p OLDSTAMP=<".venv\.installed"
if "%OLDSTAMP%"=="%STAMP%" goto :installed

echo.
echo ==^> Installing packages into .venv (first run downloads PyTorch, a few GB; be patient)
%VPY% -m pip install --upgrade pip >nul
if "%FLAVOR%"=="none" goto :install_board
if "%FLAVOR%"=="cuda" call :torch_cuda
set "EXTRAS=torch"
if defined QUEEN_QUANTIZE set "EXTRAS=torch,quant"
%VPY% -m pip install -e ".[%EXTRAS%]"
if errorlevel 1 goto :fail_install
goto :install_done
:install_board
%VPY% -m pip install -e .
if errorlevel 1 goto :fail_install
:install_done
> ".venv\.installed" echo %STAMP%
:installed

REM ---- 3. How QUEEN should run on this computer -------------------------------
if "%FLAVOR%"=="none" goto :hw_done
echo.
for /f "usebackq tokens=1* delims==" %%a in (`%VPY% tools\hardware_check.py --cmd`) do set "%%a=%%b"
if not "%QUEEN_NEEDS_BNB%"=="1" goto :hw_done
echo ==^> Installing bitsandbytes for the shrunk (nf4) model
%VPY% -m pip install "bitsandbytes>=0.45"
if not errorlevel 1 goto :hw_done
echo Couldn't install bitsandbytes; running QUEEN on the CPU instead.
set "QUEEN_QUANTIZE="
set "QUEEN_DEVICE=cpu"
:hw_done

REM ---- 4. Stockfish (optional opponent; never stops the launcher) --------------
%VPY% tools\get_stockfish.py

REM ---- 5. QUEEN model -----------------------------------------------------------
if defined QUEEN_URL goto :remote
echo.
echo ==^> Checking the QUEEN model (downloads ~8.4 GB the first time)
.venv\Scripts\queen.exe fetch
if errorlevel 1 goto :fail_download
echo QUEEN model is ready.
set "QUEEN_PRELOAD=1"
goto :start
:remote
echo.
echo ==^> Using QUEEN at %QUEEN_URL%

REM ---- 6. Start the board and open the browser ---------------------------------
:start
echo.
echo ==^> Starting Queen Board at %URL%   (close this window or press Ctrl+C to stop)
echo Open %URL% in your browser if it doesn't open by itself.
if not "%NO_BROWSER%"=="1" start "" /b %VPY% tools\open_when_ready.py %URL%
REM No --reload: it switches uvicorn to an event loop that can't start
REM subprocesses on Windows, which would break Stockfish.
%VPY% -m uvicorn server.app:app --host %HOST% --port %PORT% --log-level warning
if errorlevel 1 goto :fail_server
exit /b 0


REM ============================== helpers ========================================

:try_py
REM Usage: call :try_py <command...>  -- sets PY to its full path if it is Python 3.10+.
%* -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 exit /b 0
for /f "usebackq delims=" %%p in (`%* -c "import sys; print(sys.executable)"`) do set "PY=%%p"
exit /b 0

:torch_cuda
REM The PyTorch on PyPI for Windows is CPU-only; the NVIDIA build comes from
REM pytorch.org. Replace a CPU-only build if one is already installed.
%VPY% -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" >nul 2>&1
if not errorlevel 1 exit /b 0
%VPY% -m pip uninstall -y torch >nul 2>&1
for %%v in (cu128 cu126 cu130) do (
  %VPY% -m pip install torch --index-url https://download.pytorch.org/whl/%%v && exit /b 0
)
echo Couldn't install the NVIDIA build of PyTorch; using the CPU build instead.
exit /b 1

:fail_python
echo.
echo Error: couldn't find or install Python 3.10+.
echo Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH", then run this again.
goto :fail
:fail_venv
echo.
echo Error: couldn't create the .venv folder here. Try moving this folder somewhere simple, e.g. C:\QueenBoard
goto :fail
:fail_install
echo.
echo Error: installing packages failed (see above). Check your internet connection and run this again.
goto :fail
:fail_download
echo.
echo Error: couldn't download QUEEN. Check your internet connection and free disk space (~9 GB), then run this again.
goto :fail
:fail_server
echo.
echo The board stopped. If it never started, port %PORT% may be in use: change port in setup\settings.yaml.
goto :fail
:fail
echo.
pause
exit /b 1
