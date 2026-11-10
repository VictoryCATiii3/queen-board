"""Drop-in move sources for your chess system. It talks to MoveProvider and
doesn't care whether Stockfish or QUEEN is behind it.

    provider = QueenHttpProvider("http://<linux-box-ip>:8765")
    # or: provider = StockfishProvider(r"C:\\path\\to\\stockfish.exe")
    move = provider.choose_move(board)            # board is a python-chess Board
"""
from __future__ import annotations

import json
import urllib.request
from typing import Protocol

import chess
import chess.engine

from .types import Analysis


def board_history(board: chess.Board) -> list[str]:
    """Prior FENs, oldest first, excluding the current position (QUEEN's format)."""
    replay = board.root()
    fens = []
    for move in board.move_stack:
        fens.append(replay.fen())
        replay.push(move)
    return fens


class MoveProvider(Protocol):
    def choose_move(self, board: chess.Board) -> chess.Move: ...


class QueenHttpProvider:
    def __init__(self, url: str = "http://127.0.0.1:8765", game_id: int = 0, timeout: float = 600):
        self.url, self.game_id, self.timeout = url.rstrip("/"), game_id, timeout
        self.last: Analysis | None = None  # keep the explanation for your UI

    def analyze(self, board: chess.Board) -> Analysis:
        body = json.dumps({"fen": board.fen(), "history": board_history(board),
                           "game_id": self.game_id}).encode()
        req = urllib.request.Request(self.url + "/analyze", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        self.last = Analysis(**data)
        return self.last

    def choose_move(self, board: chess.Board) -> chess.Move:
        move = chess.Move.from_uci(self.analyze(board).best_move_uci)
        if move not in board.legal_moves:  # upstream already checks; belt and braces
            raise ValueError(f"QUEEN returned illegal move {move}")
        return move


class StockfishProvider:
    def __init__(self, path: str, think_time: float = 0.5, skill_level: int | None = None):
        self.engine = chess.engine.SimpleEngine.popen_uci(path)
        if skill_level is not None:
            self.engine.configure({"Skill Level": skill_level})
        self.think_time = think_time

    def choose_move(self, board: chess.Board) -> chess.Move:
        return self.engine.play(board, chess.engine.Limit(time=self.think_time)).move

    def close(self) -> None:
        self.engine.quit()
