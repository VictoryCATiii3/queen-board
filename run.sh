#!/usr/bin/env bash
# Queen Board launcher for Linux/macOS → http://localhost:8000
# Everything installs into .venv in this folder. Nothing global.
#
#   ./run.sh                                   Queen at http://127.0.0.1:8765 (start it with `queen serve`)
#   QUEEN_URL=http://192.168.1.50:8765 ./run.sh   use Queen running on another machine
#   CPU_ONLY=1 ./run.sh                        hide GPUs from this launcher (Queen can still be remote)
set -euo pipefail
cd "$(dirname "$0")"

export PIP_REQUIRE_VIRTUALENV=true          # never let pip touch the global Python
export PIP_DISABLE_PIP_VERSION_CHECK=1
if [ "${CPU_ONLY:-0}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=-1
fi

PY=${PYTHON:-python3}
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "Python 3.10+ is required (set PYTHON=/path/to/python3.x)." >&2; exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment in .venv ..."
  "$PY" -m venv .venv || { echo "venv failed. On Ubuntu: sudo apt install python3-venv" >&2; exit 1; }
fi
if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing packages into .venv ..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed
fi

.venv/bin/python tools/get_stockfish.py     # uses system Stockfish if present

echo
echo "  Queen Board is running at http://${HOST:-127.0.0.1}:${PORT:-8000}  (Ctrl+C to stop)"
echo
exec .venv/bin/python -m uvicorn server.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
