"""Stockfish (or any UCI engine) via python-chess. Runs entirely on the CPU.

Where it looks, in order:
  1. STOCKFISH_PATH
  2. <repo>/bin/stockfish/**  (run.bat downloads it there on Windows)
  3. `stockfish` on PATH, /usr/games/stockfish  (Ubuntu: sudo apt install stockfish)

Tuning: STOCKFISH_THREADS (default: half your CPU cores), STOCKFISH_HASH (MB, default 128).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import chess
import chess.engine

from .base import Analysis, Engine

REPO_BIN = Path(__file__).resolve().parents[2] / "bin" / "stockfish"


def _find_binary() -> str | None:
    env = os.environ.get("STOCKFISH_PATH")
    if env:
        return env if os.path.exists(env) else None
    if REPO_BIN.exists():
        for p in sorted(REPO_BIN.rglob("stockfish*")):
            if p.is_file() and (p.suffix.lower() == ".exe" or (os.name != "nt" and os.access(p, os.X_OK))):
                return str(p)
    found = shutil.which("stockfish")
    if found:
        return found
    # Common install spots that may not be on PATH when launched by double-click
    # (Debian/Ubuntu package, Homebrew on Apple Silicon and Intel Macs).
    for p in ("/usr/games/stockfish", "/opt/homebrew/bin/stockfish", "/usr/local/bin/stockfish"):
        if os.path.exists(p):
            return p
    return None


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
        # A .py path (the test suite's fake engine) runs with this Python.
        cmd = [sys.executable, self.path] if self.path.endswith(".py") else self.path
        self._transport, self._engine = await chess.engine.popen_uci(cmd)
        opts: dict = {}
        if "Skill Level" in self._engine.options:
            opts["Skill Level"] = self.skill_level
        if "Threads" in self._engine.options:
            default = max(1, (os.cpu_count() or 2) // 2)
            opts["Threads"] = int(os.environ.get("STOCKFISH_THREADS", default))
        if "Hash" in self._engine.options:
            opts["Hash"] = int(os.environ.get("STOCKFISH_HASH", "128"))
        if opts:
            await self._engine.configure(opts)

    async def stop(self) -> None:
        if self._engine:
            await self._engine.quit()
        self._engine = None

    async def set_skill(self, level: int) -> None:
        self.skill_level = max(0, min(20, level))
        if self._engine and "Skill Level" in self._engine.options:
            await self._engine.configure({"Skill Level": self.skill_level})

    async def choose_move(self, board: chess.Board) -> chess.Move:
        return (await self.choose_move_detailed(board))[0]

    async def choose_move_detailed(self, board: chess.Board) -> tuple[chess.Move, dict | None]:
        await self.start()
        assert self._engine
        result = await self._engine.play(board, chess.engine.Limit(time=self.think_time),
                                         info=chess.engine.INFO_SCORE | chess.engine.INFO_PV)
        info = result.info or {}
        details: dict = {"skill_level": self.skill_level, "think_time": self.think_time,
                         "depth": info.get("depth"), "nodes": info.get("nodes")}
        score = info.get("score")
        if score is not None:
            white = score.white()
            details["mate_in"] = white.mate()
            details["eval_cp"] = white.score() if white.mate() is None else None
        if info.get("pv"):
            details["line"] = [m.uci() for m in info["pv"]]
        return result.move, details

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
