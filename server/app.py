"""Queen Board — local chess server.

Run:  uvicorn server.app:app --reload --port 8000
Then open http://localhost:8000

The server owns the game (python-chess validates every move). The web page
is just a view: it asks for state, sends moves, and asks engines to play
or analyse. Anything else (SillyTavern, an MCP server, a 3D front end) can
drive the same API.
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .engines import ENGINES, Engine

WEB = Path(__file__).resolve().parent.parent / "web"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # The launcher sets QUEEN_PRELOAD=1 so the model loads while you look at the board,
    # instead of on Queen's first move.
    if os.environ.get("QUEEN_PRELOAD") == "1":
        await get_engine("queen").warm_up()
    yield
    for e in _engines.values():
        await e.stop()


app = FastAPI(title="Queen Board", lifespan=lifespan)


class Game:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.mode: Literal["hotseat", "engine"] = "hotseat"
        self.engine_name: str | None = None
        self.human_color: chess.Color = chess.WHITE
        self.start_fen = chess.STARTING_FEN
        # Why the engine made its last move (engines that can explain, i.e. Queen).
        self.engine_explanation: dict | None = None
        # Every opponent used this game, in order, for the PGN (opponents can swap mid-game).
        self.opponents: list[str] = []

    def engine_to_move(self) -> bool:
        return (
            self.mode == "engine"
            and not self.board.is_game_over()
            and self.board.turn != self.human_color
        )


game = Game()
_engines: dict[str, Engine] = {}


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise HTTPException(404, f"Unknown engine '{name}'")
    if name not in _engines:
        _engines[name] = ENGINES[name]()
    return _engines[name]


def result_text(board: chess.Board) -> str | None:
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return f"Draw — {outcome.termination.name.replace('_', ' ').lower()}"
    who = "White" if outcome.winner else "Black"
    return f"{who} wins — {outcome.termination.name.lower()}"


def outcome_info(board: chess.Board) -> dict | None:
    """Structured result so the UI can say "You win" / "You lose" / "Draw"."""
    if not board.is_game_over(claim_draw=True):
        return None
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    return {
        "winner": None if outcome.winner is None else ("white" if outcome.winner else "black"),
        "reason": outcome.termination.name.replace("_", " ").lower(),
    }


def analysis_payload(board: chess.Board, a, eng_name: str) -> dict:
    """An engine Analysis plus SAN versions for display (the browser doesn't know chess rules)."""
    d = a.to_dict()
    d["engine"] = eng_name
    b = board.copy()
    d["best_san"] = None
    if a.best_move:
        mv = chess.Move.from_uci(a.best_move)
        d["best_san"] = b.san(mv) if mv in b.legal_moves else a.best_move
    line_san = []
    for u in a.line:
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        line_san.append(b.san(mv))
        b.push(mv)
    d["line_san"] = line_san
    d["fen"] = board.fen()
    return d


def state() -> dict:
    b = game.board
    replay = chess.Board(game.start_fen)
    san = []
    for mv in b.move_stack:
        san.append(replay.san(mv))
        replay.push(mv)
    king_in_check = None
    if b.is_check():
        k = b.king(b.turn)
        king_in_check = chess.square_name(k) if k is not None else None
    return {
        "fen": b.fen(),
        "turn": "white" if b.turn else "black",
        "legal": [m.uci() for m in b.legal_moves],
        "history": san,
        "last_move": b.move_stack[-1].uci() if b.move_stack else None,
        "check": king_in_check,
        "result": result_text(b),
        "outcome": outcome_info(b),
        "engine_explanation": game.engine_explanation,
        "mode": game.mode,
        "engine": game.engine_name,
        "human_color": "white" if game.human_color else "black",
        "engine_to_move": game.engine_to_move(),
        "skill": _stockfish_skill(),
        "pgn": pgn(),
    }


def _stockfish_skill() -> int | None:
    eng = _engines.get("stockfish")
    if eng is not None:
        return eng.skill_level  # type: ignore[attr-defined]
    return int(os.environ.get("STOCKFISH_SKILL", "20"))


def _opponent_label() -> str | None:
    if game.mode != "engine":
        return None
    if game.engine_name == "stockfish":
        return f"Stockfish (level {_stockfish_skill()})"
    return {"queen": "Queen 4B", "random": "Random mover"}.get(game.engine_name or "", game.engine_name or "?")


def _note_opponent() -> None:
    label = _opponent_label()
    if label and (not game.opponents or game.opponents[-1] != label):
        game.opponents.append(label)


def _player(color: chess.Color) -> str:
    if game.mode == "hotseat" and not game.opponents:
        return "Human"
    if color == game.human_color:
        return "Human"
    return " / ".join(game.opponents) or "Human"


def pgn() -> str:
    g = chess.pgn.Game.from_board(game.board)
    if game.start_fen != chess.STARTING_FEN:
        g.headers["FEN"] = game.start_fen
    g.headers["Event"] = "Queen Board"
    g.headers["White"] = _player(chess.WHITE)
    g.headers["Black"] = _player(chess.BLACK)
    buf = io.StringIO()
    print(g, file=buf, end="")
    return buf.getvalue()


# ---------- API ----------

class NewGame(BaseModel):
    mode: Literal["hotseat", "engine"] = "hotseat"
    engine: str | None = None
    human_color: Literal["white", "black"] = "white"
    fen: str | None = None


class OpponentIn(BaseModel):
    mode: Literal["hotseat", "engine"]
    engine: str | None = None
    human_color: Literal["white", "black"] | None = None  # only used when leaving hotseat


class MoveIn(BaseModel):
    uci: str


class AnalyzeIn(BaseModel):
    engine: str = "stockfish"
    question: str | None = None


class SkillIn(BaseModel):
    level: int


@app.get("/api/state")
def get_state():
    return state()


@app.get("/api/engines")
async def list_engines():
    out = []
    for name in ENGINES:
        eng = get_engine(name)
        ok, why = await eng.available()
        out.append({"name": name, "available": ok, "reason": why, "explains": eng.can_explain})
    return out


@app.post("/api/new")
def new_game(req: NewGame):
    if req.mode == "engine" and not req.engine:
        raise HTTPException(400, "Pick an engine for engine mode")
    if req.engine:
        get_engine(req.engine)
    try:
        board = chess.Board(req.fen) if req.fen else chess.Board()
    except ValueError as e:
        raise HTTPException(400, f"Bad FEN: {e}")
    game.board = board
    game.start_fen = board.fen()
    game.mode = req.mode
    game.engine_name = req.engine if req.mode == "engine" else None
    game.human_color = chess.WHITE if req.human_color == "white" else chess.BLACK
    game.engine_explanation = None
    game.opponents = []
    _note_opponent()
    return state()


@app.post("/api/opponent")
def change_opponent(req: OpponentIn):
    """Swap the opponent mid-game without touching the position."""
    if req.mode == "engine":
        if not req.engine:
            raise HTTPException(400, "Pick an engine")
        get_engine(req.engine)
        if game.mode == "hotseat" and req.human_color:
            game.human_color = chess.WHITE if req.human_color == "white" else chess.BLACK
    game.mode = req.mode
    game.engine_name = req.engine if req.mode == "engine" else None
    _note_opponent()
    return state()


@app.post("/api/move")
def make_move(req: MoveIn):
    if game.engine_to_move():
        raise HTTPException(409, "It's the engine's turn")
    try:
        mv = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(400, f"Not a UCI move: {req.uci}")
    if mv not in game.board.legal_moves:
        raise HTTPException(400, f"Illegal move: {req.uci}")
    game.board.push(mv)
    return state()


@app.post("/api/engine_move")
async def engine_move():
    if not game.engine_to_move():
        raise HTTPException(409, "Not the engine's turn")
    eng = get_engine(game.engine_name)  # type: ignore[arg-type]
    ok, why = await eng.available()
    if not ok:
        raise HTTPException(503, why)
    before = game.board.copy()
    explanation = None
    try:
        if eng.can_explain:
            # One call gives both the move and the reasoning behind it; keep both.
            a = await eng.analyze(before.copy())
            if not a.best_move:
                raise RuntimeError("returned no move")
            mv = chess.Move.from_uci(a.best_move)
            explanation = analysis_payload(before, a, eng.name)
            explanation["ply"] = len(before.move_stack) + 1
        else:
            mv = await eng.choose_move(before.copy())
    except Exception as e:  # surface engine errors to the UI
        raise HTTPException(500, f"{eng.name} failed: {e}")
    if mv not in game.board.legal_moves:
        raise HTTPException(500, f"{eng.name} suggested an illegal move: {mv.uci()}")
    game.board.push(mv)
    game.engine_explanation = explanation
    return state()


@app.post("/api/undo")
def undo():
    """Hotseat: take back one ply. Engine mode: take back to the human's turn."""
    if not game.board.move_stack:
        return state()
    game.board.pop()
    if game.mode == "engine" and game.board.move_stack and game.board.turn != game.human_color:
        game.board.pop()
    ex = game.engine_explanation
    if ex and ex.get("ply", 0) > len(game.board.move_stack):
        game.engine_explanation = None  # that move was taken back
    return state()


@app.post("/api/analyze")
async def analyze(req: AnalyzeIn):
    eng = get_engine(req.engine)
    ok, why = await eng.available()
    if not ok:
        raise HTTPException(503, why)
    if game.board.is_game_over():
        raise HTTPException(400, "Game is over")
    try:
        a = await eng.analyze(game.board.copy(), req.question)
    except Exception as e:
        raise HTTPException(500, f"{eng.name} failed: {e}")
    return analysis_payload(game.board, a, eng.name)


@app.post("/api/stockfish/skill")
async def stockfish_skill(req: SkillIn):
    eng = get_engine("stockfish")
    await eng.set_skill(req.level)  # type: ignore[attr-defined]
    if game.engine_name == "stockfish" and game.board.move_stack:
        _note_opponent()  # level changed mid-game: keep the PGN honest
    return {"skill": eng.skill_level}  # type: ignore[attr-defined]


@app.get("/")
def index():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")
