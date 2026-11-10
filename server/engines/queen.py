"""Queen adapter: talks to a running `queen serve` (from queen-portable).

The board never loads the model itself. QUEEN runs as its own small server,
on this machine or another one on your network (e.g. the GPU box), and this
adapter just sends it positions. That keeps the board light enough for any
computer, and lets Mac/Windows/CPU-only machines use QUEEN running elsewhere.

Start QUEEN first:   queen serve                      (same machine)
                     queen serve --host 0.0.0.0       (reachable from other machines)

Configure with environment variables:

  QUEEN_URL      where `queen serve` is listening. Default http://127.0.0.1:8765
  QUEEN_TIMEOUT  seconds to wait for one analysis (default 600)
  QUEEN_DEBUG=1  include the full server reply in the analysis "raw" field
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import chess

from .base import Analysis, Engine

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
# QUEEN ends its answer with e.g. "EVALUATION: +1.25 ..." or "EVALUATION: +M1 ..."
# from the side to move's point of view.
_EVAL_RE = re.compile(r"EVALUATION:\s*([+-]?)\s*(M)?\s*(\d+(?:\.\d+)?)", re.I)


class QueenEngine(Engine):
    name = "queen"
    can_explain = True

    def __init__(self) -> None:
        self.url = os.environ.get("QUEEN_URL", "http://127.0.0.1:8765").rstrip("/")
        self.timeout = float(os.environ.get("QUEEN_TIMEOUT", "600"))
        self._lock = asyncio.Lock()  # the server handles one position at a time anyway

    # -- plumbing ---------------------------------------------------------------
    def _get(self, path: str, timeout: float) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as r:
            return json.loads(r.read())

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:  # the server reports errors as JSON
            try:
                msg = json.loads(e.read()).get("error", str(e))
            except Exception:
                msg = str(e)
            raise RuntimeError(f"QUEEN server error: {msg}") from None

    # -- Engine interface -------------------------------------------------------
    async def available(self) -> tuple[bool, str]:
        try:
            await asyncio.to_thread(self._get, "/health", 3)
            return True, ""
        except Exception:
            return False, (f"QUEEN isn't running at {self.url}. Start it with `queen serve` "
                           "(from queen-portable), or set QUEEN_URL to where it runs.")

    async def analyze(self, board: chess.Board, question: str | None = None) -> Analysis:
        replay = board.root()
        history = []
        for mv in board.move_stack:  # earlier positions, oldest first, excluding current
            history.append(replay.fen())
            replay.push(mv)
        payload = {"fen": board.fen(), "history": history, "game_id": 0}
        if question:
            payload["prompt"] = question
        async with self._lock:
            data = await asyncio.to_thread(self._post, "/analyze", payload)
        return parse_queen_reply(board, data, question)


# -- turning QUEEN's reply into the board's Analysis -------------------------------

def parse_queen_reply(board: chess.Board, data: dict[str, Any], question: str | None = None) -> Analysis:
    a = Analysis(raw={"move_source": data.get("move_source")})
    if os.environ.get("QUEEN_DEBUG"):
        a.raw["reply"] = data

    best = data.get("best_move_uci")
    if best:
        try:
            mv = chess.Move.from_uci(best)
            if mv in board.legal_moves:
                a.best_move = mv.uci()
        except ValueError:
            pass

    if data.get("critical_line"):
        a.line = _validated_line(board, data["critical_line"])
    if a.best_move and (not a.line or a.line[0] != a.best_move):
        a.line = [a.best_move]

    raw_text = (data.get("raw") or {}).get("raw_text") or data.get("text", "")
    a.eval_cp, a.mate_in = _parse_eval(raw_text, board.turn)

    a.text = data.get("text", "")
    if "random" in str(data.get("move_source", "")).lower():
        a.text = ("Warning: QUEEN's answer couldn't be parsed, so it fell back to a "
                  "random legal move.\n\n" + a.text)
    settings = (data.get("raw") or {}).get("inference_settings") or {}
    received = settings.get("prompt")
    a.raw["prompt_received"] = received
    if question:
        if received == question:
            note = (f"QUEEN received your question as its prompt: \"{received}\". It was trained "
                    "on a single fixed analysis prompt, so it often answers with its standard "
                    "analysis anyway.")
        elif received is not None:
            note = (f"Warning: you asked \"{question}\" but QUEEN received \"{received}\". "
                    "Your queen-portable may be out of date.")
        else:
            note = "Your question was sent, but the QUEEN server didn't report what prompt it used."
        a.text = f"({note})\n\n" + a.text
    return a


def _parse_eval(text: str, turn: chess.Color) -> tuple[int | None, int | None]:
    """QUEEN's EVALUATION line -> (centipawns, mate) from White's point of view."""
    m = None
    for m in _EVAL_RE.finditer(text or ""):
        pass  # keep the last one
    if not m:
        return None, None
    sign = -1 if m.group(1) == "-" else 1
    if turn == chess.BLACK:
        sign = -sign
    if m.group(2):
        return None, sign * int(float(m.group(3)))
    return int(round(sign * float(m.group(3)) * 100)), None


def _validated_line(board: chess.Board, line: Any) -> list[str]:
    """Keep only the prefix of the line that is actually legal.
    QUEEN's longer variations can drift into illegal moves."""
    if isinstance(line, dict) and isinstance(line.get("legal_steps"), list):
        tokens = [s.get("uci", "") for s in line["legal_steps"] if isinstance(s, dict)]
    elif isinstance(line, str):
        tokens = _UCI_RE.findall(line)
    else:
        tokens = _UCI_RE.findall(json.dumps(line))
    b = board.copy()
    out = []
    for t in tokens:
        try:
            mv = chess.Move.from_uci(t)
        except ValueError:
            break
        if mv not in b.legal_moves:
            break
        out.append(t)
        b.push(mv)
    return out
