"""Queen adapter. QUEEN can run in one of two places, behind the same interface:

  in this process (default)   The board loads QUEEN itself the first time it's
                              needed, via queen_portable. One program, one terminal.
  on another machine          Set QUEEN_URL to a running `queen serve`, e.g. the GPU
                              box on your network. The board then only sends positions
                              over HTTP, so it can run on any computer.

Settings (environment variables):

  QUEEN_URL      use a remote `queen serve` instead of loading QUEEN here
  QUEEN_TIMEOUT  seconds to wait for one remote analysis (default 600)
  QUEEN_DEBUG=1  include QUEEN's full reply in the analysis "raw" field
  QUEEN_MODEL, QUEEN_DEVICE, QUEEN_QUANTIZE, QUEEN_MODEL_DIR   passed to queen_portable
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any

import chess

from .base import Analysis, Engine

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
# QUEEN ends its answer with e.g. "EVALUATION: +1.25 ..." or "EVALUATION: +M1 ..."
# from the side to move's point of view.
_EVAL_RE = re.compile(r"EVALUATION:\s*([+-]?)\s*(M)?\s*(\d+(?:\.\d+)?)", re.I)


# -- where QUEEN runs ---------------------------------------------------------------
# Both transports take the same request dict and return queen_portable's Analysis dict.

class LocalQueen:
    """Loads QUEEN into this process on first use (and keeps it loaded)."""

    where = "this computer"

    def __init__(self) -> None:
        self._backend = None
        self._load_lock = threading.Lock()

    def available(self) -> tuple[bool, str]:
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:
            return False, ("QUEEN's model packages aren't installed here "
                           f"({', '.join(missing)} missing). Use the launcher script, or set "
                           "QUEEN_URL to a QUEEN running on another machine.")
        return True, ""

    def _load(self):
        with self._load_lock:
            if self._backend is None:
                from queen_portable.backends import load_backend
                from queen_portable.config import QueenConfig
                self._backend = load_backend(QueenConfig())
        return self._backend

    def analyze(self, payload: dict) -> dict:
        from queen_portable.types import AnalyzeRequest
        return self._load().analyze(AnalyzeRequest(**payload)).to_dict()


class RemoteQueen:
    """Talks to `queen serve` somewhere else over HTTP."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url, self.timeout = url.rstrip("/"), timeout
        self.where = self.url

    def available(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                json.loads(r.read())
            return True, ""
        except Exception:
            return False, (f"QUEEN isn't reachable at {self.url}. Start it there with "
                           "`queen serve --host 0.0.0.0`, or unset QUEEN_URL to run it here.")

    def analyze(self, payload: dict) -> dict:
        req = urllib.request.Request(self.url + "/analyze", data=json.dumps(payload).encode(),
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


def make_transport():
    url = os.environ.get("QUEEN_URL")
    if url:
        return RemoteQueen(url, float(os.environ.get("QUEEN_TIMEOUT", "600")))
    return LocalQueen()


class QueenEngine(Engine):
    name = "queen"
    can_explain = True

    def __init__(self) -> None:
        self.transport = make_transport()
        self._lock = asyncio.Lock()  # one position at a time on one model

    async def available(self) -> tuple[bool, str]:
        return await asyncio.to_thread(self.transport.available)

    async def warm_up(self) -> None:
        """Start loading a local QUEEN in the background. Never blocks or fails startup."""
        if not isinstance(self.transport, LocalQueen):
            return
        ok, why = self.transport.available()
        if not ok:
            print(f"QUEEN: {why}")
            return

        def load():
            print("QUEEN: loading the model in the background...", flush=True)
            try:
                backend = self.transport._load()
                device = getattr(backend, "device", "?")
                print(f"QUEEN: ready on {device}.", flush=True)
            except Exception as e:  # shown again in the UI if someone picks Queen
                print(f"QUEEN: failed to load: {e}", flush=True)

        threading.Thread(target=load, daemon=True).start()

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
            data = await asyncio.to_thread(self.transport.analyze, payload)
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
