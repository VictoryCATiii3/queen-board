# Queen Board

Play chess on your own computer against **QUEEN**, Princeton's 4-billion-parameter chess
model that explains its moves in plain English. When QUEEN plays a move, it tells you why.
Ask it about any position and it draws its plan on the board as arrows. Stockfish and a
random mover are included as other opponents, and you can swap between them mid-game.

Fair warning: QUEEN plays at roughly 2,700 Elo (grandmaster level) in the paper.
Good luck.

> **Unofficial.** This project is not affiliated with or endorsed by the QUEEN authors.
> All credit for the model goes to them:
> - Paper: *Language Models that Play Chess and Explain Their Moves* — Adithya Bhaskar,
>   Jeffrey Cheng & Danqi Chen (2026), https://arxiv.org/abs/2610.03695
> - Models: [princeton-nlp/queen_pawn-8](https://huggingface.co/princeton-nlp/queen_pawn-8) ·
>   [princeton-nlp/queen_hce-4](https://huggingface.co/princeton-nlp/queen_hce-4)

## What you need

| | Works well | Works, slowly |
|---|---|---|
| **Windows 10/11** | NVIDIA GPU | any PC with 16 GB RAM (runs on the CPU) |
| **macOS** | Apple Silicon (M1 or newer), 16 GB+, macOS 14+ | Intel Macs (CPU) |
| **Linux** | NVIDIA GPU | any PC with 16 GB RAM (CPU) |

- **Disk space:** about 20 GB free (the QUEEN model is ~9 GB; PyTorch with GPU support is several GB).
- **Internet:** for the first run only.
- **Python:** on Windows and macOS, nothing to install: the launcher uses your Python 3.10+
  or fetches a private copy into this folder. On Linux it needs Python 3.10+ with venv
  (Ubuntu: `sudo apt install python3 python3-venv`).

Tested on: Ubuntu with an RTX 3090, a Mac mini M4 (16 GB), and a Windows gaming laptop.

## Install and run

1. **Download** this repository: the green **Code** button > **Download ZIP**, or
   `git clone`. Unzip it somewhere simple, e.g. `C:\QueenBoard` on Windows.
   Avoid folders synced by OneDrive/iCloud: the setup creates several GB of files.
2. **Start it** for your system (below). The first run takes a while (10–15 GB of
   downloads); later runs start in seconds. The board then opens in your browser.
3. **Stop it** by closing its terminal window or pressing Ctrl+C there.

### Windows

Double-click **`queen_runner_windows.bat`**.

If Windows shows *"Windows protected your PC"*, click **More info** > **Run anyway**.
With an NVIDIA GPU it automatically installs the GPU version of PyTorch. When you press
Ctrl+C to stop, Windows asks *"Terminate batch job (Y/N)?"*: press `Y`.

### macOS

Double-click **`Queen Board.command`** (or run `./queen_runner_mac.sh` in Terminal).

The first time, macOS may say it can't verify the file: right-click it, choose **Open**,
then **Open** again. QUEEN runs on the Apple GPU.

### Linux

In a terminal in this folder:

```bash
./queen_runner_linux.sh
```

You can also double-click it in your file manager (some need right-click >
*Run as a Program*). It opens its own terminal window.

### How QUEEN runs on your computer

Each time it starts, the launcher checks your hardware and tells you what it picked:

| Your hardware | How QUEEN runs |
|---|---|
| NVIDIA GPU with 10 GB+ (RTX 3080, 3090, 4080, ...) | on the GPU, full size |
| NVIDIA GPU with 5.5–10 GB (most laptop GPUs) | on the GPU, shrunk to 4-bit (`nf4`, experimental) |
| Apple Silicon Mac | on the Apple GPU |
| anything else | on the CPU: works, but each QUEEN move can take many minutes |

On the CPU, QUEEN needs about 10 GB of *free* memory; close other programs (browser tabs
especially) or it will crawl. To force the CPU, set `cpu: true` in the settings.

## Where things are saved

**The QUEEN model** (~9 GB) is *not* stored in this folder. It downloads once into
Hugging Face's standard cache in your user folder, so reinstalling Queen Board (or other
programs that use the same model) won't download it again:

| System | Location |
|---|---|
| Windows | `C:\Users\<you>\.cache\huggingface\hub\models--princeton-nlp--queen_pawn-8` |
| macOS / Linux | `~/.cache/huggingface/hub/models--princeton-nlp--queen_pawn-8` |

`.cache` is a hidden folder. On Windows, paste `%USERPROFILE%\.cache\huggingface` into
File Explorer's address bar. To store the model elsewhere (an external drive, a bigger
disk), set `download_folder` in the settings *before* the first run. To print the exact
location: `.venv\Scripts\queen.exe fetch` (Windows) or `.venv/bin/queen fetch`.

**Everything else** stays inside this folder:

| Folder | What's in it |
|---|---|
| `.venv/` | Queen Board's private Python packages, including PyTorch (hidden on macOS/Linux) |
| `.tools/` | a private Python, only if your computer didn't have a recent one |
| `bin/stockfish/` | the Stockfish engine |
| `games/` | your saved games (see [Game logs](#game-logs)) |

**To uninstall,** delete this folder and the model folder above.

## Using it

- **Opponent:** Nobody (play both sides), Random mover, Stockfish (levels 0–20), or Queen 4B.
- **When QUEEN plays,** the panel on the right explains why it chose that move.
- **Analyze position** asks the chosen engine about the current position and draws its
  best move and line as arrows (gold = your side, blue = replies).
- **Questions** typed in the Ask box are sent to QUEEN. It was trained on a single fixed
  analysis prompt, so it usually answers with its standard analysis anyway.
- **Undo** takes back your last move and the engine's reply.
- **Swap opponents mid-game** any time: the new one takes over the current position.
- **Start from a position** accepts any FEN.

QUEEN's moves are strong, but its explanations can contain mistakes even when the move
is right.

## Settings

Open **`setup/settings.yaml`** in any text editor, change a value, save, and start the
launcher again. Each setting is explained in the file.

| Setting | Default | What it does |
|---|---|---|
| `cpu` | `false` | `true` = run QUEEN on the CPU. `false` = use the GPU when there's a suitable one |
| `model` | `pawn-8` | `pawn-8` (stronger) or `hce-4` |
| `quantize` | (auto) | `int8` or `nf4` to shrink QUEEN for smaller NVIDIA GPUs; picked automatically when needed |
| `download_folder` | Hugging Face default | Where the model downloads (e.g. an external drive) |
| `model_folder` | | Use a QUEEN folder you've already downloaded |
| `queen_url` | | Use QUEEN running on another computer (below) |
| `port` | `8000` | The board's web address: `http://127.0.0.1:8000` |
| `open_browser` | `true` | Open the browser automatically |
| `save_games` | `true` | Keep a log of every game in `games/` |
| `stockfish_time` | `0.5` | Seconds Stockfish thinks per move |

Every setting can also be given for a single run on the command line, which wins over
the file, e.g. `PORT=8080 ./queen_runner_linux.sh`. The underlying environment variable
names are listed in `tools/settings.py`.

### QUEEN on another computer

Play on a laptop while QUEEN runs on a GPU computer on the same network:

1. On the GPU computer, run the launcher once, then start just QUEEN:
   `.venv/bin/queen serve --host 0.0.0.0` (Windows: `.venv\Scripts\queen.exe serve --host 0.0.0.0`)
2. On the laptop, set `queen_url: http://<gpu-computer-ip>:8765` in `setup/settings.yaml`
   and start the launcher. It skips downloading PyTorch and the model.

## Game logs

Every game is saved in **`games/`** as two files named by when it started:

- `2026-11-10_23-41-05.pgn`: the game itself, updated after every move. Opens in any
  chess program or site (e.g. Lichess's *Import game* page).
- `2026-11-10_23-41-05.jsonl`: everything that happened, one entry per line. It records
  each move and who played it (you, Stockfish at what level, or QUEEN), QUEEN's full
  explanation of its moves, Stockfish's evaluation and planned line, every analysis you
  asked for along with your question, opponent swaps, undos, and the result.

Set `save_games: false` to turn this off.

## Troubleshooting

| Problem | Fix |
|---|---|
| QUEEN takes minutes per move | It's running on the CPU. That's expected without a suitable GPU. Check the launcher window for what it picked |
| "only N GB of memory is free" | Close other programs, especially browser tabs, and restart |
| "No usable NVIDIA GPU found" but you have one | Update your NVIDIA driver from nvidia.com, delete `.venv`, and run the launcher again |
| Board doesn't start: port in use | Change `port` in the settings |
| Stockfish missing from the opponent list | The download failed. macOS: `brew install stockfish`; Ubuntu: `sudo apt install stockfish`; or rerun the launcher |
| Odd moves or garbled explanations on a laptop GPU | The shrunk `nf4` mode is experimental: set `cpu: true` |
| Something else | Open an issue and paste the text from the launcher window |

## The `queen` command

The launcher also installs a command-line tool (`.venv/bin/queen`, or
`.venv\Scripts\queen.exe` on Windows):

```bash
queen analyze --fen "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
queen serve            # QUEEN as a small HTTP API on port 8765
queen info             # which device and model would be used
queen fetch            # download the model now / print where it is
```

`POST /analyze` on `queen serve` with `{"fen": ..., "history": [...], "prompt": null}`
returns the move, explanation and critical line as JSON, for use from other programs.

## How it works

The official QUEEN code needs vLLM, which only runs on Linux with NVIDIA GPUs. QUEEN is a
Leela Chess Zero board encoder connected to the SmolLM3 language model by 16 gated
cross-attention "bridge" layers. `queen_portable/xattn.py` re-implements those bridges in
plain PyTorch (verified identical to the official math), so the same, unmodified weights
run through ordinary `transformers` on NVIDIA GPUs, Apple Silicon or the CPU. QUEEN's own
helper code (board encoding, move parsing) is loaded from the downloaded model, not copied
into this repository.

```
queen_portable/   QUEEN without vLLM: model loading, the bridge, CLI, HTTP server
server/           the board's API (FastAPI) and engines (queen, stockfish, random), game logs
web/              the board itself (plain HTML/JS/SVG, no build step)
setup/            settings.yaml: the settings you edit
tools/            Stockfish downloader, settings reader, hardware check, bridge check
tests/            automated tests (stand-ins for QUEEN and Stockfish; no GPU needed)
```

Each piece works on its own: `queen_portable` can be used without the board, and the board
talks to every engine through the same small interface (`server/engines/base.py`).

## For developers

**Manual install** (instead of the launchers), in a virtual environment:

```bash
pip install -r requirements.txt        # same as: pip install -e ".[torch]"
uvicorn server.app:app --port 8000
```

On Windows with an NVIDIA GPU, install PyTorch from https://pytorch.org first; the default
Windows build is CPU-only. Developed and tested with Python 3.10–3.12, PyTorch 2.14 and
transformers 5.20.

**Tests:**

```bash
pip install -e ".[dev]"
python -m pytest -q
```

The tests use stand-ins for Stockfish and QUEEN and write their game logs to a temporary
folder, so they run without a GPU or the model.

**Checking against the official runner** (Linux + NVIDIA, with the official QUEEN
environment installed):

```bash
.venv/bin/python -m queen_portable.parity --upstream-python <official-venv>/bin/python
```

This plays a few positions through both with greedy decoding and compares moves and text.
Last result: 5/5 identical moves.

**Board API:**

| Method | Path | Body / returns |
|---|---|---|
| GET | `/api/state` | FEN, legal moves, history, result, engine's last explanation, PGN |
| GET | `/api/engines` | which engines are available |
| POST | `/api/new` | `{mode: "hotseat"\|"engine", engine?, human_color?, fen?}` |
| POST | `/api/opponent` | `{mode, engine?}`: swap opponent mid-game |
| POST | `/api/move` | `{uci: "e2e4"}` |
| POST | `/api/engine_move` | the engine plays its turn |
| POST | `/api/undo` | |
| POST | `/api/analyze` | `{engine, question?}` → best move, eval, line, text |
| POST | `/api/stockfish/skill` | `{level: 0-20}` |

To add an engine, subclass `server/engines/base.py:Engine` and register it in
`server/engines/__init__.py`.

## License

The code in this repository is licensed under Apache-2.0 (see `LICENSE`). QUEEN's
weights and code are **not** included here and are not covered by that license: they're
downloaded from Princeton's Hugging Face repositories. Their model card says the
upstream components' terms apply (SmolLM3: Apache-2.0; Leela Chess Zero encoder: see
lczero.org) and asserts no blanket license for the combined release.

If the QUEEN authors would like anything here changed or removed, please open an issue
and I'll act on it promptly.
