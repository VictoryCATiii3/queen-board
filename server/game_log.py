"""Per-game logs: every move, who played it, and everything the engines said.

Each game gets two files in games/ (next to this repo, or GAMES_DIR):

  2026-11-10_23-41-05.jsonl   one JSON object per line, in order: game start,
                              every move (with the engine's full response for
                              engine moves), analyses you asked for, opponent
                              swaps, undos, and the result
  2026-11-10_23-41-05.pgn     the game so far, rewritten after every move, so it
                              opens in any chess program even mid-game

JSON Lines is append-only, so a crash or Ctrl+C never loses earlier moves.
Turn logging off with SAVE_GAMES=0 (or `save_games: false` in setup/settings.yaml).

Knows nothing about the board or engines: callers hand it plain dicts.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _enabled() -> bool:
    return os.environ.get("SAVE_GAMES", "1").strip().lower() not in ("0", "false", "no", "off")


def _folder() -> Path:
    return Path(os.environ.get("GAMES_DIR") or ROOT / "games").expanduser()


class GameLog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stem: Path | None = None
        self._finished = False

    @property
    def path(self) -> Path | None:
        return None if self._stem is None else self._stem.with_suffix(".jsonl")

    def start(self, **info: Any) -> None:
        """Begin a new game's files. Nothing is written until the first event."""
        with self._lock:
            self._stem = None
            self._finished = False
            if not _enabled():
                return
            folder = _folder()
            try:
                folder.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"Game log: can't create {folder} ({e}); not saving games.")
                return
            base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            stem, n = folder / base, 1
            while stem.with_suffix(".jsonl").exists():
                n += 1
                stem = folder / f"{base}_{n}"
            self._stem = stem
        self.event("game_start", **info)

    def event(self, kind: str, **data: Any) -> None:
        with self._lock:
            if self._stem is None:
                return
            row = {"time": datetime.now().isoformat(timespec="seconds"), "event": kind, **data}
            try:
                with open(self._stem.with_suffix(".jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            except OSError as e:
                print(f"Game log: write failed ({e}); not saving this game any more.")
                self._stem = None

    def write_pgn(self, pgn: str) -> None:
        with self._lock:
            if self._stem is None:
                return
            try:
                self._stem.with_suffix(".pgn").write_text(pgn + "\n", encoding="utf-8")
            except OSError:
                pass

    def finish(self, **result: Any) -> None:
        """Record the result once, however many times the final position is seen."""
        if self._finished:
            return
        self._finished = True
        self.event("game_over", **result)

    def reopen(self) -> None:
        """An undo after the game ended: the game continues, so a later result is new."""
        self._finished = False
