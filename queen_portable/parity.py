"""Check the torch backend against Princeton's vLLM runner, using greedy decoding
(temperature 0) so both should pick the same tokens.

    python -m queen_portable.parity --model-dir ~/Workspace/QUEEN/queen_pawn-8 \
        --upstream-python ~/Workspace/QUEEN/queen_hce-4/.venv/bin/python
    (add --fen "<FEN>" to test your own positions)

Small differences late in long outputs are normal (different GPU kernels round
bf16 slightly differently). What matters: same best move, and a long shared
prefix of text.
"""
from __future__ import annotations

import argparse

from .backends.torch_native import TorchNativeBackend
from .backends.vllm_subprocess import VllmSubprocessBackend
from .config import add_config_args, config_from_args
from .types import AnalyzeRequest

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",   # mate in 1
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 b - - 3 8",
]


def common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fen", action="append")
    p.add_argument("--upstream-python", help="python of the venv that has vLLM (the official runner)")
    add_config_args(p)
    args = p.parse_args()
    fens = args.fen or POSITIONS

    cfg = config_from_args(args)
    if args.upstream_python:
        cfg.upstream_python = args.upstream_python
    # vLLM grabs ~72% of the GPU, so finish every reference run before loading ours.
    ref = VllmSubprocessBackend(cfg)
    refs = []
    for i, fen in enumerate(fens, 1):
        print(f"vllm reference {i}/{len(fens)}...")
        refs.append(ref.analyze(AnalyzeRequest(fen=fen), temperature=0))
    print("loading torch backend...")
    ours = TorchNativeBackend(cfg)

    agree = 0
    for fen, a in zip(fens, refs):
        b = ours.analyze(AnalyzeRequest(fen=fen), temperature=0)
        ra, rb = a.raw.get("raw_text", ""), b.raw.get("raw_text", "")
        same = a.best_move_uci == b.best_move_uci
        agree += same
        print(f"\n{fen}\n  vllm : {a.best_move_uci} ({a.move_source})"
              f"\n  torch: {b.best_move_uci} ({b.move_source})  {'OK' if same else 'DIFFERENT'}"
              f"\n  text : {common_prefix(ra, rb)} of {len(ra)}/{len(rb)} chars identical")
    print(f"\nbest move agreement: {agree}/{len(fens)}")


if __name__ == "__main__":
    main()
