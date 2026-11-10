# queen-portable

**Unofficial** tools for running Princeton's QUEEN chess-language model
(*Language Models that Play Chess and Explain Their Moves*, Bhaskar, Cheng & Chen, 2026)
on more hardware, and plugging it into chess apps.

Not affiliated with or endorsed by the authors. All credit for the model goes to them:
- Paper: https://arxiv.org/abs/2610.03695
- Models: https://huggingface.co/princeton-nlp/queen_pawn-8 · https://huggingface.co/princeton-nlp/queen_hce-4
- Code: https://github.com/queen-project/queen

This repo contains **only our own code**. Model weights and Princeton's helper code
are downloaded directly from their Hugging Face repos.

## What's different

The official runner needs vLLM, which only runs on Linux with NVIDIA GPUs. QUEEN's
board-reading "bridge" (16 gated cross-attention layers between the Leela encoder and
SmolLM3) is re-implemented here in plain PyTorch (`queen_portable/xattn.py`, verified
identical to upstream's math), so the model runs through ordinary `transformers` on Linux, macOS and Windows.

## Install

```bash
pip install -e ".[torch]"            # add ",quant" on Windows/Linux for --quantize
```

(One-step install scripts for Linux, macOS and Windows are planned.)

## Use

No setup needed: the model (~8.4 GiB) downloads into the standard Hugging Face
cache the first time you use it, and the best device (NVIDIA, Apple Silicon, or
CPU) is picked automatically.

```bash
queen analyze --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
queen serve                          # local server on http://127.0.0.1:8765
queen serve --host 0.0.0.0           # ...reachable from other machines on your network
queen info                           # which device and model would be used
```

Common flags: `--model {pawn-8,hce-4}`, `--device {auto,cuda,mps,cpu}`,
`--quantize {int8,nf4}` (NVIDIA only, for smaller GPUs), and `--model-dir PATH` to
use a checkpoint folder you've already downloaded.

Ask the server about a position:

```bash
curl -s localhost:8765/analyze -H 'Content-Type: application/json' \
  -d '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}'
```

From Python on any machine on your network:

```python
import chess
from queen_portable.providers import QueenHttpProvider

queen = QueenHttpProvider("http://<server-ip>:8765")
board = chess.Board()
move = queen.choose_move(board)
print(move, queen.last.text)   # the move and QUEEN's explanation
```

`StockfishProvider` has the same `choose_move(board)` method, so apps can swap engines.

## Platforms

| Platform | How | Status |
|---|---|---|
| Linux + NVIDIA | default | verified: 5/5 moves match the official runner |
| macOS (Apple Silicon, 16 GB+) | default (uses MPS) | ready to test |
| Windows + NVIDIA, under ~10 GB VRAM | `--quantize int8` or `nf4` | ready to test |

It's slower than the official vLLM runner (no batching), which is fine for one game at a time.

## Verifying against the official runner (developers)

```bash
python -m queen_portable.parity --upstream-python <venv-with-vLLM>/bin/python
```

Runs a few positions through both with greedy decoding and compares moves and text.

## Environment variables (optional)

Every flag has an environment-variable form for scripts: `QUEEN_MODEL`,
`QUEEN_MODEL_DIR`, `QUEEN_DEVICE`, `QUEEN_QUANTIZE`. To store models somewhere
else (e.g. an external drive), set Hugging Face's own `HF_HOME`.

## Licensing

Our code in this repo is Apache-2.0. The QUEEN weights and Princeton's code are **not**
covered by that: their model card states that upstream component terms apply (SmolLM3:
Apache-2.0; Leela Chess Zero encoder: see lczero.org) and asserts no blanket license for
the combined release. If the authors would like anything here changed or removed,
please open an issue and I'll act on it promptly.

Explanations and variations from the model may be wrong.
