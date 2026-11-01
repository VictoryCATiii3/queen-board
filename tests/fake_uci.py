#!/usr/bin/env python3
"""Minimal fake UCI engine for tests: plays the first legal move."""
import sys

import chess

board = chess.Board()
for line in sys.stdin:
    cmd = line.strip()
    if cmd == "uci":
        print("id name FakeUCI\noption name Skill Level type spin default 20 min 0 max 20\noption name Threads type spin default 1 min 1 max 1024\noption name Hash type spin default 16 min 1 max 33554432\nuciok", flush=True)
    elif cmd == "isready":
        print("readyok", flush=True)
    elif cmd.startswith("position"):
        parts = cmd.split()
        if parts[1] == "startpos":
            board = chess.Board(); rest = parts[2:]
        else:
            i = parts.index("moves") if "moves" in parts else len(parts)
            board = chess.Board(" ".join(parts[2:i])); rest = parts[i:]
        if rest and rest[0] == "moves":
            for m in rest[1:]:
                board.push_uci(m)
    elif cmd.startswith("go"):
        mv = sorted(board.legal_moves, key=lambda m: m.uci())[0]
        print(f"info depth 1 score cp 25 pv {mv.uci()}", flush=True)
        print(f"bestmove {mv.uci()}", flush=True)
    elif cmd == "quit":
        break
