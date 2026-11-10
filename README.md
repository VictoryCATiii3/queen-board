# Queen Board

Play chess against **QUEEN**, Princeton's 4B-parameter chess model that explains its
moves in plain English, on your own computer. Pick QUEEN as your opponent and it tells
you why it played each move; ask it about any position and it draws its plan as arrows.
Stockfish and a random mover are included as other opponents.

**Unofficial.** Not affiliated with or endorsed by the QUEEN authors. All credit for the
model goes to them:
- Paper: *Language Models that Play Chess and Explain Their Moves* (Adithya Bhaskar, Jeffrey Cheng & Danqi Chen, 2026), https://arxiv.org/abs/2610.03695
- Models: https://huggingface.co/princeton-nlp/queen_pawn-8 · https://huggingface.co/princeton-nlp/queen_hce-4

## Quick start

| System | Launcher | Status |
|---|---|---|
| Linux (NVIDIA GPU recommended) | `queen_runner_linux.sh` | tested |
| macOS (Apple Silicon, 16 GB+) | `queen_runner_mac.sh` or double-click `Queen Board.command` | in testing |
| Windows | `run.bat` | untested; board + Stockfish only for now |

**Linux:** open a terminal in this folder and run

```bash
./queen_runner_linux.sh
```

or double-click it in your file manager and choose *Run* (some file managers need
right-click > *Run as a Program*). It opens its own terminal window.

**macOS:** double-click **Queen Board.command**, or in Terminal run `./queen_runner_mac.sh`.
The first time, macOS may refuse to open a file downloaded from the internet: right-click
it, choose *Open*, then *Open* again. If your Mac's Python is older than 3.10 (macOS ships
3.9), the launcher fetches a private Python into `.tools/` in this folder. QUEEN runs on
the Apple GPU and needs macOS 14+ and an M1 or newer; otherwise it falls back to the CPU.

The first run takes a while: it creates a private Python environment in `.venv`,
installs PyTorch (a few GB), downloads Stockfish, and downloads QUEEN (~8.4 GB, into the
standard Hugging Face cache). Then it opens the board in your browser. Later runs start
in seconds. Nothing is installed outside this folder except the model download.
Press Ctrl+C or close the terminal window to stop.

**Hardware:** QUEEN runs best on an NVIDIA GPU with 10 GB+ of memory (RTX 3080/3090/4080
and up) or an Apple Silicon Mac with 16 GB+. Without either it runs on the CPU, which works
but can take minutes per move.

## Using it

- **Opponent:** Nobody (play both sides), Random mover, Stockfish (levels 0–20), or Queen 4B.
- **When QUEEN plays**, the panel on the right explains why it chose that move.
- **Analyze position** asks the chosen engine about the current position and draws its
  best move and line as arrows (gold = your side, blue = replies).
- **Questions** typed in the Ask box are sent to QUEEN as its prompt. QUEEN was trained
  on a single fixed analysis prompt, so it usually gives its standard analysis anyway.
- **Undo** takes back your last move (and the engine's reply). **Start from a position**
  takes any FEN.

QUEEN's moves are strong (about 2,700 Elo in the paper), but its explanations can contain
mistakes even when the move is right.

## Settings

Set these before the launcher, e.g. `QUEEN_MODEL=hce-4 ./queen_runner_linux.sh`.

| Variable | Default | What it does |
|---|---|---|
| `PORT` | `8000` | Web port |
| `QUEEN_MODEL` | `pawn-8` | `pawn-8` (stronger) or `hce-4` |
| `QUEEN_DEVICE` | auto | `cuda`, `mps`, or `cpu` |
| `QUEEN_QUANTIZE` | off | `int8` or `nf4`: smaller memory for NVIDIA GPUs under ~10 GB (experimental) |
| `QUEEN_MODEL_DIR` | | Use a QUEEN checkpoint folder you've already downloaded |
| `QUEEN_URL` | | Use QUEEN running on another machine (see below) |
| `HF_HOME` | `~/.cache/huggingface` | Where Hugging Face stores the model (e.g. an external drive) |
| `STOCKFISH_PATH` | auto | Stockfish binary to use |
| `STOCKFISH_TIME` / `STOCKFISH_THREADS` / `STOCKFISH_HASH` | `0.5` s / half your cores / `128` MB | Stockfish tuning |
| `NO_BROWSER` | | `1` = don't open a browser |

### QUEEN on another machine

Run the board on a laptop and QUEEN on a GPU machine on the same network:

```bash
# on the GPU machine (inside this folder, after running the launcher once)
.venv/bin/queen serve --host 0.0.0.0

# on the laptop
QUEEN_URL=http://<gpu-machine-ip>:8765 ./queen_runner_linux.sh
```

With `QUEEN_URL` set, the laptop skips installing PyTorch and the model.

## The `queen` command

The launcher also installs a command-line tool into `.venv/bin/queen`:

```bash
.venv/bin/queen analyze --fen "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
.venv/bin/queen serve            # QUEEN as a small HTTP API on port 8765
.venv/bin/queen info             # which device and model would be used
```

`POST /analyze {"fen": ..., "history": [...], "prompt": null}` on `queen serve` returns the
move, explanation and critical line as JSON, for use from other programs.

## How it works

The official QUEEN runner needs vLLM, which only runs on Linux with NVIDIA GPUs. QUEEN is
a Leela Chess Zero board encoder connected to the SmolLM3 language model by 16 gated
cross-attention "bridge" layers. `queen_portable/xattn.py` re-implements those bridges in
plain PyTorch (verified identical to the official math), so the same weights run through
ordinary `transformers` on NVIDIA, Apple Silicon or CPU. QUEEN's own helper code (board
encoding, move parsing) is loaded from the downloaded model folder, not copied here.

```
queen_portable/   QUEEN without vLLM: model loading, the bridge, CLI, HTTP server
server/           the board's API (FastAPI) and engine adapters (queen, stockfish, random)
web/              the board itself (plain HTML/JS/SVG, no build step)
tools/            Stockfish downloader, bridge equivalence check
```

Each piece works on its own: `queen_portable` can be used without the board, and the board
talks to every engine through the same small interface (`server/engines/base.py`).

### Checking against the official runner (developers)

On Linux + NVIDIA with the official QUEEN environment installed:

```bash
.venv/bin/python -m queen_portable.parity --upstream-python <official-venv>/bin/python
```

This plays a few positions through both runners with greedy decoding and compares the
moves and text. Our last run: 5/5 identical moves.

## Board API

| Method | Path | Body |
|---|---|---|
| GET | `/api/state` | — FEN, legal moves, history, result, outcome, engine's last explanation, PGN |
| GET | `/api/engines` | — which engines are set up |
| POST | `/api/new` | `{mode: "hotseat"|"engine", engine?, human_color?, fen?}` |
| POST | `/api/move` | `{uci: "e2e4"}` |
| POST | `/api/engine_move` | — engine plays its turn |
| POST | `/api/undo` | — |
| POST | `/api/analyze` | `{engine, question?}` → best move, eval, line, text |
| POST | `/api/stockfish/skill` | `{level: 0-20}` |

To add an engine, subclass `server/engines/base.py:Engine` and register it in
`server/engines/__init__.py`.

## Tests

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
```

Tests use stand-ins for Stockfish and QUEEN, so they run without a GPU or model download.

## Licensing

The code in this repo is Apache-2.0. QUEEN's weights and code are **not** included and not
covered by that license: they're downloaded from Princeton's Hugging Face repos. Their
model card says upstream component terms apply (SmolLM3: Apache-2.0; Leela Chess Zero
encoder: see lczero.org) and asserts no blanket license for the combined release. If the
authors would like anything here changed or removed, please open an issue and I'll act on
it promptly.
