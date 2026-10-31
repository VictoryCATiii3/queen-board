#!/usr/bin/env bash
# Start Queen Board on http://localhost:8000
# Usage: ./run.sh            (uses ~/Workspace/QUEEN for Queen)
#        QUEEN_DIR=/path/to/queen_hce-4 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/uvicorn server.app:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
