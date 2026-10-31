"""Stockfish (or any UCI engine) via python-chess.

Set STOCKFISH_PATH if the binary isn't on your PATH as `stockfish`.
On Ubuntu:  sudo apt install stockfish   (installs to /usr/games/stockfish)
"""
from __future__ import annotations

import os
import shutil

import chess
import chess.engine

from .base import Analysis, Engine


def _find_binary() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env if os.path.exists(env) else None
    return shutil.which("stockfish") or (
        "/usr/games/stockfish" if os.path.exists("/usr/games/stockfish") else None
    )


class StockfishEngine(Engine):
    name = "stockfish"

    def __init__(self, think_time: float | None = None, skill_level: int | None = None):
        self.path = _find_binary()
        self.think_time = think_time or float(os.environ.get("STOCKFISH_TIME", "0.5"))
        # 0..20. Unlike Queen, Stockfish CAN be weakened.
        self.skill_level = skill_level if skill_level is not None else int(
            os.environ.get("STOCKFISH_SKILL", "20")
        )
        self._engine: chess.engine.UciProtocol | None = None
        self._transport = None

    async def available(self) -> tuple[bool, str]:
        if not self.path:
            return False, "Stockfish not found. Install it or set STOCKFISH_PATH."
        return True, ""

    async def start(self) -> None:
        if self._engine or not self.path:
            return
        self._transport, self._engine = await chess.engine.popen_uci(self.path)
        if "Skill Level" in self._engine.options:
            await self._engine.configure({"Skill Level": self.skill_level})

    async def stop(self) -> None:
        if self._engine:
            await self._engine.quit()
        self._engine = None

    async def set_skill(self, level: int) -> None:
        self.skill_level = max(0, min(20, level))
        if self._engine and "Skill Level" in self._engine.options:
            await self._engine.configure({"Skill Level": self.skill_level})

    async def choose_move(self, board: chess.Board) -> chess.Move:
        await self.start()
        assert self._engine
        result = await self._engine.play(board, chess.engine.Limit(time=self.think_time))
        return result.move

    async def analyze(self, board: chess.Board, question: str | None = None) -> Analysis:
        await self.start()
        assert self._engine
        info = await self._engine.analyse(board, chess.engine.Limit(time=self.think_time))
        score = info.get("score")
        pv = [m.uci() for m in info.get("pv", [])]
        a = Analysis(best_move=pv[0] if pv else None, line=pv)
        if score is not None:
            white = score.white()
            a.mate_in = white.mate()
            a.eval_cp = white.score() if a.mate_in is None else None
        a.text = _summarise(board, a)
        a.raw = {"depth": info.get("depth"), "nodes": info.get("nodes")}
        return a


def _summarise(board: chess.Board, a: Analysis) -> str:
    """Stockfish doesn't explain itself; give a plain one-liner so the
    panel isn't empty. Queen replaces this with real explanations."""
    if not a.best_move:
        return "No move found."
    san_line = []
    b = board.copy()
    for uci in a.line[:6]:
        mv = chess.Move.from_uci(uci)
        if mv not in b.legal_moves:
            break
        san_line.append(b.san(mv))
        b.push(mv)
    if a.mate_in is not None:
        side = "White" if a.mate_in > 0 else "Black"
        ev = f"{side} mates in {abs(a.mate_in)}"
    elif a.eval_cp is not None:
        ev = f"eval {a.eval_cp / 100:+.2f} (White's view)"
    else:
        ev = "eval unknown"
    return f"Best: {san_line[0]} — {ev}. Line: {' '.join(san_line)}"
