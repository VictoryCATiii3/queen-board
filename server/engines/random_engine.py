"""A deliberately terrible opponent. Useful for testing the game loop
without any engine installed."""
from __future__ import annotations

import random

import chess

from .base import Analysis, Engine


class RandomEngine(Engine):
    name = "random"

    async def analyze(self, board: chess.Board, question: str | None = None) -> Analysis:
        moves = list(board.legal_moves)
        if not moves:
            return Analysis(text="No legal moves.")
        # Prefer captures and checks a little so games aren't completely aimless.
        weighted = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
        move = random.choice(weighted or moves)
        return Analysis(
            best_move=move.uci(),
            line=[move.uci()],
            text=f"I picked {board.san(move)} more or less at random.",
        )
