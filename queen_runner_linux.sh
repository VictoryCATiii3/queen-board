#!/usr/bin/env bash
# =============================================================================
#  Queen Board (Linux) -- play chess against QUEEN and see it explain its moves.
#
#  Run it:   ./queen_runner_linux.sh        (or right-click > Run as a Program)
#
#  First run sets everything up inside this folder (.venv), downloads Stockfish
#  and the QUEEN model (~8.4 GB, into the standard Hugging Face cache), then
#  opens the board in your browser. Later runs start in seconds.
#  Nothing is installed system-wide. Ctrl+C (or closing this window) stops it.
#
#  Optional settings, e.g.  PORT=8080 ./queen_runner_linux.sh
#    PORT=8000                    web port
#    QUEEN_MODEL=hce-4            the other released checkpoint (default pawn-8)
#    QUEEN_DEVICE=cpu             force CPU (very slow; auto-picks NVIDIA if present)
#    QUEEN_QUANTIZE=int8|nf4      smaller GPUs (NVIDIA only)
#    QUEEN_MODEL_DIR=/path        use a checkpoint folder you already have
#    QUEEN_URL=http://ip:8765     use QUEEN running on another machine instead
#    HF_HOME=/path                where Hugging Face stores downloads
#    NO_BROWSER=1                 don't open a browser
# =============================================================================
set -euo pipefail
SELF="$(readlink -f "$0")"
cd "$(dirname "$SELF")"

# Double-clicked from the file manager (no terminal)? Reopen in a terminal window
# so progress is visible and Ctrl+C / closing the window stops it.
if [ ! -t 1 ] && [ -z "${QUEEN_IN_TERMINAL:-}" ]; then
  export QUEEN_IN_TERMINAL=1
  if command -v gnome-terminal >/dev/null 2>&1; then exec gnome-terminal -- "$SELF"; fi
  for t in x-terminal-emulator konsole xfce4-terminal mate-terminal xterm; do
    if command -v "$t" >/dev/null 2>&1; then exec "$t" -e "$SELF"; fi
  done
fi

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
URL="http://${HOST}:${PORT}"

say()  { printf '\n\033[1;33m==>\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31mError:\033[0m %s\n' "$*" >&2; read -rp "Press Enter to close..." _ || true; exit 1; }

# Keep the window open on errors when launched by double-click.
trap 'fail "something went wrong (see above)."' ERR

# --- 1. Python 3.10+ -----------------------------------------------------------
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
[ -n "$PY" ] || fail "Python 3.10 or newer is needed. On Ubuntu: sudo apt install python3 python3-venv"

# --- 2. Private environment in .venv --------------------------------------------
export PIP_REQUIRE_VIRTUALENV=true PIP_DISABLE_PIP_VERSION_CHECK=1
if [ ! -x .venv/bin/python ]; then
  say "Creating a private Python environment in .venv (first run only)"
  "$PY" -m venv .venv || { rm -rf .venv; fail "couldn't create it. On Ubuntu: sudo apt install python3-venv"; }
fi
VPY=.venv/bin/python

# What to install: everything, or just the board when QUEEN runs on another machine.
EXTRAS="torch"; [ -n "${QUEEN_URL:-}" ] && EXTRAS=""
STAMP="$(sha256sum pyproject.toml | cut -c1-16)-${EXTRAS}"
if [ "$(cat .venv/.installed 2>/dev/null)" != "$STAMP" ]; then
  say "Installing packages into .venv (first run downloads a few GB for PyTorch; be patient)"
  "$VPY" -m pip install --upgrade pip >/dev/null
  if [ -n "$EXTRAS" ]; then "$VPY" -m pip install -e ".[${EXTRAS}]"; else "$VPY" -m pip install -e .; fi
  echo "$STAMP" > .venv/.installed
fi

# --- 3. Stockfish (optional opponent; never stops the launcher) -------------------
"$VPY" tools/get_stockfish.py || true

# --- 4. QUEEN model ------------------------------------------------------------
if [ -z "${QUEEN_URL:-}" ]; then
  if [ -z "${QUEEN_DEVICE:-}" ] && ! command -v nvidia-smi >/dev/null 2>&1; then
    say "No NVIDIA GPU found: QUEEN will run on the CPU. That works, but each move can take minutes."
  fi
  say "Checking the QUEEN model (downloads ~8.4 GB the first time)"
  .venv/bin/queen fetch || fail "couldn't download QUEEN. Check your internet connection and try again."
  echo "QUEEN model is ready."
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
  "$VPY" -c "import urllib.request,sys; urllib.request.urlopen('${URL}/api/state', timeout=1)" 2>/dev/null && break
  sleep 0.5
done

echo "Open ${URL} in your browser if it doesn't open by itself."
if [ "${NO_BROWSER:-0}" != "1" ] && command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$URL" >/dev/null 2>&1 || true
fi
wait "$SERVER"
