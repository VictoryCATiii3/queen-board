# Queen Board

A local chessboard for playing against — and asking questions of — **Queen**
(`princeton-nlp/queen_pawn-8` by default, from *Language Models That Play Chess and Explain
Their Moves*), with Stockfish and a random mover as stand-in opponents.

This is stage 1 of a bigger plan:

1. **Board + engines** (this repo, now) — play, analyse, see arrows.
2. **Persistent Queen worker** — done: [queen-portable](../queen-portable) (`queen serve`) loads the model once and runs on Linux, macOS and Windows.
3. **MCP wrapper** — the same API as tools any LLM client can call.
4. **SillyTavern extension** — your character asks Queen, then explains in its own voice.

## Quick start

Both launchers create a `.venv` in this folder and install everything there.
Nothing is installed globally (`PIP_REQUIRE_VIRTUALENV` is set while they run).
They also download Stockfish into `bin/stockfish/` if it isn't already
installed, and only reinstall packages when `requirements.txt` changes.

**Windows (CPU only):** double-click `run.bat`. It needs Python 3.10+ from
python.org. The browser opens on http://127.0.0.1:8000. It hides GPUs
(`CUDA_VISIBLE_DEVICES=-1`) for the board itself; Queen runs separately (below). Stockfish uses half your CPU
cores by default (`set STOCKFISH_THREADS=8` to change).

**Linux / macOS:**

```bash
./run.sh                          # Queen from `queen serve` on this machine
CPU_ONLY=1 ./run.sh               # same CPU-only setup as run.bat
```

Pick an opponent: **Nobody** (play both sides), **Random mover**, **Stockfish**
(levels 0–20), or **Queen 4B**. Click or drag to move. *Analyze position* asks
the chosen engine about the current position; its best move and line are drawn
as arrows (gold = your side's moves, blue = replies).

### Hooking up Queen

Queen runs as its own small server from **queen-portable**, which loads the model
once and works on Linux + NVIDIA, Apple Silicon Macs, and Windows (quantized for
smaller GPUs). The board just sends it positions over HTTP, so the board can be on
a different computer from the model.

```bash
# on the machine with the GPU
queen serve --host 0.0.0.0        # --host 0.0.0.0 only if other machines need it

# on the machine with the board (same machine: no setting needed)
QUEEN_URL=http://<gpu-machine-ip>:8765 ./run.sh       # Windows: set QUEEN_URL=... then run.bat
```

If Queen isn't reachable, it's greyed out in the opponent list with the reason.

| Variable | Default | What it does |
|---|---|---|
| `QUEEN_URL` | `http://127.0.0.1:8765` | Where `queen serve` is listening |
| `QUEEN_TIMEOUT` | `600` | Seconds to wait for one analysis |
| `QUEEN_DEBUG` | unset | `1` = include Queen's full reply in the panel's raw data |
| `STOCKFISH_PATH` | `bin/stockfish/`, then PATH | Stockfish binary |
| `STOCKFISH_TIME` | `0.5` | Seconds per Stockfish move |
| `STOCKFISH_THREADS` | half your CPU cores | CPU threads Stockfish uses |
| `STOCKFISH_HASH` | `128` | Stockfish hash table, MB |
| `SKIP_STOCKFISH` | unset | `1` = don't download Stockfish |

Notes:
- The full game history is sent with every position; Queen uses it.
- Queen's longer lines can drift into illegal moves. The adapter keeps only the
  legal prefix of any line it draws.
- Queen's own evaluation (the `EVALUATION:` line, including mates) is converted to
  White's point of view for the eval display.
- **Questions** ("why not Nf3?") are sent to Queen as its prompt. This is
  experimental: Queen was trained on one fixed analysis prompt, so free-form answers
  can be rough, and the panel says so. Leave the box empty for the standard analysis.
- Queen's explanations can contain mistakes even when its move is right.

## API

All moves are validated server-side by python-chess.

| Method | Path | Body |
|---|---|---|
| GET | `/api/state` | — FEN, legal moves, history, check, result, PGN |
| GET | `/api/engines` | — which engines are set up |
| POST | `/api/new` | `{mode: "hotseat"|"engine", engine?, human_color?, fen?}` |
| POST | `/api/move` | `{uci: "e2e4"}` |
| POST | `/api/engine_move` | — engine plays its turn |
| POST | `/api/undo` | — |
| POST | `/api/analyze` | `{engine, question?}` → best move, eval, line, text |
| POST | `/api/stockfish/skill` | `{level: 0-20}` |

## Adding an engine

Subclass `server/engines/base.py:Engine`, implement `analyze()` (and
optionally `choose_move()`), register it in `server/engines/__init__.py`.

## Tests

```bash
.venv/bin/python -m pytest -q        # Windows: .venv\Scripts\python -m pytest -q
```

Tests use a fake UCI engine and a fake Queen server, so they run without a GPU.

## Colab

`notebooks/queen_board_colab.ipynb` runs the whole thing on a Colab GPU. Queen
needs BF16 and a CUDA-13-compatible driver, so a T4 won't work; use an L4 or
A100 runtime. The first cell checks your GPU before downloading anything.

## License note

This repo contains only the board and glue code. It does not include Queen's
weights; the model card says no blanket license is asserted for the combined
release, and upstream terms (SmolLM3: Apache-2.0, Lc0) apply. Check those
before redistributing anything built on Queen.
