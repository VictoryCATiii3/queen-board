#!/usr/bin/env bash
# =============================================================================
#  Queen Board (macOS) -- play chess against QUEEN and see it explain its moves.
#
#  Run it:   ./queen_runner_mac.sh                     (in Terminal)
#      or:   double-click "Queen Board.command"         (opens Terminal for you)
#
#  First run sets everything up inside this folder (.venv), downloads Stockfish
#  and the QUEEN model (~8.4 GB, into the standard Hugging Face cache), then
#  opens the board in your browser. Later runs start in seconds.
#  Nothing is installed system-wide. Ctrl+C (or closing the window) stops it.
#
#  Needs an Apple Silicon Mac (M1 or newer) with 16 GB+ memory for QUEEN to run
#  on the GPU. Intel Macs fall back to the CPU (very slow).
#
#  Settings live in setup/settings.yaml (cpu, model, download_folder, port, ...).
#  Anything below can also be given for one run on the command line, which wins
#  over the file: e.g.  PORT=8080 ./queen_runner_mac.sh
#    PORT=8000                    web port
#    QUEEN_MODEL=hce-4            the other released checkpoint (default pawn-8)
#    QUEEN_DEVICE=cpu             force CPU (very slow; default uses the Apple GPU)
#    QUEEN_MODEL_DIR=/path        use a checkpoint folder you already have
#    QUEEN_URL=http://ip:8765     use QUEEN running on another machine instead
#    HF_HOME=/path                where Hugging Face stores downloads (e.g. an external drive)
#    NO_BROWSER=1                 don't open a browser
# =============================================================================
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd -P)"

say()  { printf '\n\033[1;33m==>\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;35m!!\033[0m  %s\n' "$*"; }
fail() { printf '\n\033[1;31mError:\033[0m %s\n' "$*" >&2; read -rp "Press Enter to close..." _ || true; exit 1; }
trap 'fail "something went wrong (see above)."' ERR

# --- 1. Python 3.10+ (or a private copy via uv if the Mac doesn't have one) -----
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$(command -v "$c")"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  # macOS's built-in python3 is 3.9. Fetch uv and a private Python, both kept in .tools/.
  export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" UV_CACHE_DIR="$PWD/.tools/cache"
  if [ ! -x .tools/uv ]; then
    say "This Mac's Python is too old; fetching a private Python into .tools/ (first run only)"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/.tools" UV_NO_MODIFY_PATH=1 sh >/dev/null \
      || fail "couldn't download uv. Check your internet connection, or install Python 3.12 from python.org."
  fi
  .tools/uv python install 3.12 >/dev/null || fail "couldn't install Python 3.12."
  PY="$(.tools/uv python find 3.12)"
fi

# --- Settings from setup/settings.yaml (the command line still wins) --------------
SETTINGS_ENV="$("$PY" tools/settings.py --shell)" || fail "couldn't read setup/settings.yaml"
eval "$SETTINGS_ENV"
[ "${QUEEN_DEVICE:-}" = "cpu" ] && echo "Settings: running QUEEN on the CPU (cpu: true)."
[ -n "${HF_HOME:-}" ] && echo "Settings: QUEEN downloads go to $HF_HOME"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"


# --- This Mac ---------------------------------------------------------------
# QUEEN runs in bfloat16, which the Mac GPU backend only supports on Apple Silicon
# with macOS 14+. Anywhere else, use the CPU rather than fail.
if [ "$(uname -m)" != "arm64" ] && [ -z "${QUEEN_URL:-}" ]; then
  warn "This is an Intel Mac: QUEEN will run on the CPU, which can take minutes per move."
  export QUEEN_DEVICE="${QUEEN_DEVICE:-cpu}"
fi
MEM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
if [ "$MEM_GB" -gt 0 ] && [ "$MEM_GB" -lt 16 ] && [ -z "${QUEEN_URL:-}" ]; then
  warn "This Mac has ${MEM_GB} GB of memory. QUEEN needs about 9 GB on its own, so 16 GB+ is recommended."
fi
MACOS_MAJOR=$(sw_vers -productVersion 2>/dev/null | cut -d. -f1 || echo 0)
if [ "${MACOS_MAJOR:-0}" -lt 14 ] && [ -z "${QUEEN_URL:-}" ]; then
  warn "QUEEN needs macOS 14 (Sonoma) or newer to use the Apple GPU; using the CPU instead (very slow)."
  export QUEEN_DEVICE="${QUEEN_DEVICE:-cpu}"
fi
# If PyTorch meets an operation the Apple GPU doesn't support yet, run just that
# piece on the CPU instead of stopping with an error.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# --- 2. Private environment in .venv --------------------------------------------
export PIP_REQUIRE_VIRTUALENV=true PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x .venv/bin/python ]; then
  say "Creating a private Python environment in .venv (first run only)"
  "$PY" -m venv .venv || { rm -rf .venv; fail "couldn't create it."; }
fi
VPY=.venv/bin/python

EXTRAS="torch"; [ -n "${QUEEN_URL:-}" ] && EXTRAS=""
STAMP="$(shasum -a 256 pyproject.toml | cut -c1-16)-${EXTRAS}"
if [ "$(cat .venv/.installed 2>/dev/null)" != "$STAMP" ]; then
  say "Installing packages into .venv (first run downloads PyTorch; be patient)"
  "$VPY" -m pip install --upgrade pip >/dev/null
  if [ -n "$EXTRAS" ]; then "$VPY" -m pip install -e ".[${EXTRAS}]"; else "$VPY" -m pip install -e .; fi
  echo "$STAMP" > .venv/.installed
fi

# --- 2b. How QUEEN should run on this computer (GPU size, memory) --------------
if [ -n "$EXTRAS" ]; then
  HW_ENV="$("$VPY" tools/hardware_check.py --shell)" || HW_ENV=""
  eval "$HW_ENV"
  if [ "${QUEEN_NEEDS_BNB:-}" = "1" ]; then
    say "Installing bitsandbytes for the shrunk (nf4) model"
    "$VPY" -m pip install "bitsandbytes>=0.45" || { echo "Couldn't install bitsandbytes; using the CPU."; export QUEEN_QUANTIZE="" QUEEN_DEVICE=cpu; }
  fi
fi

# --- 3. Stockfish (optional opponent; never stops the launcher) -------------------
"$VPY" tools/get_stockfish.py || true

# --- 4. QUEEN model ------------------------------------------------------------
if [ -z "${QUEEN_URL:-}" ]; then
  say "Checking the QUEEN model (downloads ~8.4 GB the first time)"
  .venv/bin/queen fetch || fail "couldn't download QUEEN. Check your internet connection and try again."
  echo "QUEEN model is ready on device: $(.venv/bin/queen info | "$VPY" -c 'import json,sys; print(json.load(sys.stdin)["device"])')"
  export QUEEN_PRELOAD=1   # load it while you look at the board
else
  say "Using QUEEN at ${QUEEN_URL}"
fi

# --- 5. Start the board and open the browser ----------------------------------------
trap - ERR
say "Starting Queen Board at ${URL}   (Ctrl+C to stop)"
"$VPY" -m uvicorn server.app:app --host "$HOST" --port "$PORT" --log-level warning &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null; wait "$SERVER" 2>/dev/null; echo; echo "Stopped."; exit 0' INT TERM
trap 'kill "$SERVER" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    fail "the board didn't start. Is port ${PORT} already in use? Try: PORT=8080 $0"
  fi
  "$VPY" -c "import urllib.request; urllib.request.urlopen('${URL}/api/state', timeout=1)" 2>/dev/null && break
  sleep 0.5
done

echo "Open ${URL} in your browser if it doesn't open by itself."
[ "${NO_BROWSER:-0}" = "1" ] || open "$URL" >/dev/null 2>&1 || true
wait "$SERVER"
