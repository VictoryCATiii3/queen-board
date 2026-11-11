"""Engine interface.

Every opponent (Stockfish, Queen, a random mover, later a SillyTavern bot)
implements the same two calls, so the board and the API never care which
one is plugged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import chess


@dataclass
class Analysis:
    """What an engine says about a position.

    best_move:  UCI string, e.g. "e2e4" (None if the engine has nothing).
    eval_cp:    centipawns from White's point of view (None if unknown).
    mate_in:    moves to mate from White's point of view (+ = White mates).
    line:       principal variation as UCI moves, starting with best_move.
    text:       human-readable explanation (Queen fills this; others summarise).
    raw:        anything engine-specific, passed through for debugging.
    """

    best_move: str | None = None
    eval_cp: int | None = None
    mate_in: int | None = None
    line: list[str] = field(default_factory=list)
    text: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_move": self.best_move,
            "eval_cp": self.eval_cp,
            "mate_in": self.mate_in,
            "line": self.line,
            "text": self.text,
            "raw": self.raw,
        }


class Engine:
    name: str = "engine"
    can_explain: bool = False  # True if it produces real explanations (Queen)

    async def start(self) -> None:  # open processes, load models
        pass

    async def stop(self) -> None:
        pass

    async def available(self) -> tuple[bool, str]:
        """(ok, reason). Lets the UI grey out engines that aren't installed."""
        return True, ""

    async def choose_move(self, board: chess.Board) -> chess.Move:
        analysis = await self.analyze(board)
        if not analysis.best_move:
            raise RuntimeError(f"{self.name} returned no move")
        return chess.Move.from_uci(analysis.best_move)

    async def choose_move_detailed(self, board: chess.Board) -> tuple[chess.Move, dict | None]:
        """The move plus whatever the engine reported while choosing it (for the game log).
        Engines that report nothing just return their move."""
        return await self.choose_move(board), None

    async def analyze(self, board: chess.Board, question: str | None = None) -> Analysis:
        raise NotImplementedError
