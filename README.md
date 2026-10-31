# Queen Board

A local chessboard for playing against — and asking questions of — **Queen**
(`princeton-nlp/queen_hce-4`, from *Language Models That Play Chess and Explain
Their Moves*), with Stockfish and a random mover as stand-in opponents.

This is stage 1 of a bigger plan:

1. **Board + engines** (this repo, now) — play, analyse, see arrows.
2. **Persistent Queen worker** — load the model once instead of per move.
3. **MCP wrapper** — the same API as tools any LLM client can call.
4. **SillyTavern extension** — your character asks Queen, then explains in its own voice.

## Quick start

```bash
sudo apt install stockfish        # optional, for the Stockfish opponent
./run.sh                          # → http://localhost:8000
```

Pick an opponent: **Nobody** (play both sides), **Random mover**, **Stockfish**
(levels 0–20), or **Queen 4B**. Click or drag to move. *Analyze position* asks
the chosen engine about the current position; its best move and line are drawn
as arrows (gold = your side's moves, blue = replies).

### Hooking up Queen

The server runs the official `infer.py` per position. Point it at the folder
you downloaded from Hugging Face (the one with `infer.py` and `.venv/`):

```bash
QUEEN_DIR=~/Workspace/QUEEN/queen_hce-4 ./run.sh
```

| Variable | Default | What it does |
|---|---|---|
| `QUEEN_DIR` | `~/Workspace/QUEEN` (or its `queen_hce-4/`) | Where `infer.py` lives |
| `QUEEN_CMD` | `.venv/bin/python infer.py --fen "{fen}" --history {history} --gpu-memory-utilization 0.35` | Command template |
| `QUEEN_TIMEOUT` | `600` | Seconds before giving up |
| `QUEEN_DEBUG` | unset | Set to `1` to see the stdout tail in the panel |
| `STOCKFISH_PATH` | `stockfish` on PATH | Stockfish binary |
| `STOCKFISH_TIME` | `0.5` | Seconds per Stockfish move |

Notes:
- `{history}` is a temp JSON file of earlier FENs. Queen uses game history, so
  repetitions and context are passed properly. **Check the file format matches
  what infer.py expects** (assumed: a JSON list of FEN strings).
- Each Queen call currently reloads the model, so expect long waits per move.
  That's stage 2's job.
- Queen's longer lines can drift into illegal moves. The adapter keeps only the
  legal prefix of any line it draws.
- Free-form questions (“why not Nf3?”) are not sent to Queen yet: `infer.py`
  uses a fixed analysis prompt. The question box is wired through the API so
  it's ready once we find out how to prompt it.

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
.venv/bin/pytest -q
```

Tests use a fake UCI engine and a fake `infer.py`, so they run without a GPU.

## Colab

`notebooks/queen_board_colab.ipynb` runs the whole thing on a Colab GPU. Queen
needs BF16 and a CUDA-13-compatible driver, so a T4 won't work; use an L4 or
A100 runtime. The first cell checks your GPU before downloading anything.

## License note

This repo contains only the board and glue code. It does not include Queen's
weights; the model card says no blanket license is asserted for the combined
release, and upstream terms (SmolLM3: Apache-2.0, Lc0) apply. Check those
before redistributing anything built on Queen.
