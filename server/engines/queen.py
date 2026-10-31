"""Queen (princeton-nlp/queen_hce-4) adapter.

v0 approach: shell out to the official infer.py for each position and parse
the JSON it prints after its logs (keys per the model card: text, raw_text,
best_move_uci, move_source, plus a legality-checked critical line). That
reloads the model on every call (slow), but needs no changes to their code.
The next step is a persistent worker that loads the model once (README: Roadmap).

Configure with environment variables:

  QUEEN_DIR      the downloaded queen_hce-4 folder (has infer.py and .venv).
                 Default: ~/Workspace/QUEEN, or ~/Workspace/QUEEN/queen_hce-4
  QUEEN_CMD      command template. {fen} = position, {history} = path to a
                 JSON list of earlier FENs (Queen uses history; a FEN alone
                 can't express repetition). Default:
                 .venv/bin/python infer.py --fen "{fen}" --history {history}
                   --gpu-memory-utilization 0.35
  QUEEN_TIMEOUT  seconds before giving up (default 600)
  QUEEN_DEBUG=1  include the stdout tail in the analysis "raw" field
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import tempfile
from pathlib import Path
from typing import Any

import chess

from .base import Analysis, Engine

_KEYS = {
    "best_move": ["best_move_uci", "best_move", "move", "best_move_uci", "uci", "bestmove"],
    "eval": ["evaluation", "eval", "eval_cp", "score", "centipawns"],
    "text": ["text", "analysis", "explanation", "text", "summary", "reasoning"],
    "line": ["critical_line", "pv", "line", "principal_variation"],
}

_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")


class QueenEngine(Engine):
    name = "queen"
    can_explain = True

    def __init__(self) -> None:
        d = Path(os.path.expanduser(os.environ.get("QUEEN_DIR", "~/Workspace/QUEEN")))
        if not (d / "infer.py").exists() and (d / "queen_hce-4" / "infer.py").exists():
            d = d / "queen_hce-4"
        self.dir = d
        self.cmd = os.environ.get(
            "QUEEN_CMD",
            '.venv/bin/python infer.py --fen "{fen}" --history {history} '
            "--gpu-memory-utilization 0.35",
        )
        self.timeout = float(os.environ.get("QUEEN_TIMEOUT", "600"))
        self._lock = asyncio.Lock()  # one GPU job at a time

    async def available(self) -> tuple[bool, str]:
        if not (self.dir / "infer.py").exists():
            return False, f"Queen's infer.py not found in {self.dir} (set QUEEN_DIR)."
        return True, ""

    async def analyze(self, board: chess.Board, question: str | None = None) -> Analysis:
        ok, why = await self.available()
        if not ok:
            raise RuntimeError(why)
        # Earlier positions, oldest first, excluding the current one.
        replay = board.root()
        history = []
        for mv in board.move_stack:
            history.append(replay.fen())
            replay.push(mv)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(history, fh)
            hist_path = fh.name
        cmd = self.cmd.replace("{fen}", board.fen()).replace("{history}", shlex.quote(hist_path))
        try:
            return await self._run(board, cmd, question)
        finally:
            os.unlink(hist_path)

    async def _run(self, board: chess.Board, cmd: str, question: str | None) -> Analysis:
        async with self._lock:
            proc = await asyncio.create_subprocess_exec(
                *shlex.split(cmd),
                cwd=self.dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                out, err = await asyncio.wait_for(proc.communicate(), self.timeout)
            except asyncio.TimeoutError:
                proc.kill()
                raise RuntimeError(f"Queen timed out after {self.timeout:.0f}s")
        stdout = out.decode(errors="replace")
        if proc.returncode != 0:
            tail = err.decode(errors="replace")[-1500:]
            raise RuntimeError(f"Queen exited with code {proc.returncode}:\n{tail}")
        return parse_queen_output(board, stdout, question)


def _last_json(text: str) -> dict[str, Any] | None:
    """infer.py may print logs before its JSON; grab the last {...} block."""
    starts = [m.start() for m in re.finditer(r"^\{", text, re.M)] or [
        i for i, c in enumerate(text) if c == "{"
    ]
    for s in reversed(starts):
        try:
            obj = json.loads(text[s:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            # try trimming to the matching last brace
            e = text.rfind("}")
            try:
                obj = json.loads(text[s : e + 1])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def _find(obj: Any, names: list[str]) -> Any:
    """Depth-first search for the first key in `names` anywhere in obj."""
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] not in (None, ""):
                return obj[n]
        for v in obj.values():
            r = _find(v, names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find(v, names)
            if r is not None:
                return r
    return None


def parse_queen_output(board: chess.Board, stdout: str, question: str | None = None) -> Analysis:
    data = _last_json(stdout)
    a = Analysis(raw={"json": data} if data else {"stdout_tail": stdout[-3000:]})
    if os.environ.get("QUEEN_DEBUG"):
        a.raw["stdout_tail"] = stdout[-3000:]

    # Best move: trust a key if present, else first legal UCI-looking token.
    cand = _find(data, _KEYS["best_move"]) if data else None
    candidates = [str(cand)] if cand else []
    candidates += _UCI_RE.findall(stdout)
    for c in candidates:
        m = _UCI_RE.search(c)
        if not m:
            continue
        try:
            mv = chess.Move.from_uci(m.group(1))
        except ValueError:
            continue
        if mv in board.legal_moves:
            a.best_move = mv.uci()
            break

    ev = _find(data, _KEYS["eval"]) if data else None
    if isinstance(ev, (int, float)):
        # Heuristic: small floats are pawns, large numbers are centipawns.
        a.eval_cp = int(ev * 100) if abs(ev) < 50 and isinstance(ev, float) else int(ev)

    line = _find(data, _KEYS["line"]) if data else None
    if line:
        a.line = _validated_line(board, line)

    src = _find(data, ["move_source"]) if data else None
    if src:
        a.raw["move_source"] = src

    text = _find(data, _KEYS["text"]) if data else None
    a.text = text if isinstance(text, str) else (json.dumps(text, indent=1) if text else "")
    if src and "random" in str(src).lower():
        a.text = ("Warning: Queen's answer couldn't be parsed, so infer.py fell back "
                  "to a random legal move.\n\n" + a.text)
    if question:
        a.text = (
            f"(Queen uses a fixed analysis prompt — your question '{question}' "
            f"was not sent to it yet. Showing its standard analysis.)\n\n" + a.text
        )
    return a


def _validated_line(board: chess.Board, line: Any) -> list[str]:
    """Keep only the prefix of the line that is actually legal.
    Queen's longer variations can drift into illegal moves."""
    if isinstance(line, str):
        tokens = _UCI_RE.findall(line)
    elif isinstance(line, list):
        tokens = []
        for item in line:
            tokens += _UCI_RE.findall(json.dumps(item)) if not isinstance(item, str) else _UCI_RE.findall(item)
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
