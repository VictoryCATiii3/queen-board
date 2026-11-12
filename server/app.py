"""Queen Board — local chess server.

Run:  uvicorn server.app:app --port 8000      (the launchers do this for you)
Then open http://localhost:8000
Don't use --reload on Windows: it switches to an event loop that can't start
subprocesses, which breaks Stockfish.

The server owns the game (python-chess validates every move). The web page
is just a view: it asks for state, sends moves, and asks engines to play
or analyse. Anything else (SillyTavern, an MCP server, a 3D front end) can
drive the same API.
"""
from __future__ import annotations

import io
import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import chess
import chess.pgn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .game_log import GameLog
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
        # Random per game. Engines that sample (Queen) seed from it, so games differ
        # but any one game can be reproduced from its log.
        self.game_id: int = 0

    def over(self) -> bool:
        return game_over(self.board)

    def engine_to_move(self) -> bool:
        return self.mode == "engine" and not self.over() and self.board.turn != self.human_color


def final_outcome(board: chess.Board) -> tuple[str | None, str, str] | None:
    """(winner "white"/"black"/None, reason, PGN result) if the game has ended, else None.

    The one definition used everywhere (banner, engines, legal moves, logs, PGN).
    Besides checkmate/stalemate/etc., threefold repetition and the 50-move rule end
    the game automatically, like on most chess sites, but only once they have
    actually happened. (python-chess's claim_draw=True also counts a draw the
    player *could* claim with their next move, which would end games early.)"""
    o = board.outcome()
    if o is not None:
        winner = None if o.winner is None else ("white" if o.winner else "black")
        return winner, o.termination.name.replace("_", " ").lower(), o.result()
    if board.is_repetition(3):
        return None, "threefold repetition", "1/2-1/2"
    if board.halfmove_clock >= 100:
        return None, "fifty moves", "1/2-1/2"
    return None


def game_over(board: chess.Board) -> bool:
    return final_outcome(board) is not None


game = Game()
game_log = GameLog()
_engines: dict[str, Engine] = {}


def get_engine(name: str) -> Engine:
    if name not in ENGINES:
        raise HTTPException(404, f"Unknown engine '{name}'")
    if name not in _engines:
        _engines[name] = ENGINES[name]()
        _engines[name].new_game(game.game_id)
    return _engines[name]


def result_text(board: chess.Board) -> str | None:
    f = final_outcome(board)
    if f is None:
        return None
    winner, reason, _ = f
    return f"Draw — {reason}" if winner is None else f"{winner.title()} wins — {reason}"


def outcome_info(board: chess.Board) -> dict | None:
    """Structured result so the UI can say "You win" / "You lose" / "Draw"."""
    f = final_outcome(board)
    return None if f is None else {"winner": f[0], "reason": f[1]}


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


def _queen_settings() -> dict:
    keys = ("QUEEN_MODEL", "QUEEN_DEVICE", "QUEEN_QUANTIZE", "QUEEN_URL")
    return {k.lower()[6:]: os.environ[k] for k in keys if os.environ.get(k)}


def _who(color: chess.Color) -> str:
    """Who is playing this colour right now, for the log."""
    side = "White" if color == chess.WHITE else "Black"
    if game.mode == "hotseat":
        return f"Human ({side})"
    if color == game.human_color:
        return "Human"
    return _opponent_label() or "?"


def _log_move(before: chess.Board, mv: chess.Move, response: dict | None = None) -> None:
    game_log.event("move", ply=len(before.move_stack) + 1, move_number=before.fullmove_number,
                   color="white" if before.turn else "black", by=_who(before.turn),
                   san=before.san(mv), uci=mv.uci(), fen_before=before.fen(),
                   fen_after=game.board.fen(), response=response)
    game_log.write_pgn(pgn())
    o = outcome_info(game.board)
    if o:
        game_log.finish(result=final_outcome(game.board)[2], **o,
                        white=_player(chess.WHITE), black=_player(chess.BLACK))


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
        "legal": [] if game.over() else [m.uci() for m in b.legal_moves],
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
    f = final_outcome(game.board)
    g.headers["Result"] = f[2] if f else "*"
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
    game.game_id = random.randrange(1, 2**31)
    for e in _engines.values():
        e.new_game(game.game_id)
    _note_opponent()
    game_log.start(start_fen=game.start_fen, mode=game.mode, opponent=_opponent_label(),
                   human_color="white" if game.human_color else "black",
                   game_id=game.game_id, queen=_queen_settings())
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
    old = _opponent_label() or "nobody (both sides human)"
    game.mode = req.mode
    game.engine_name = req.engine if req.mode == "engine" else None
    _note_opponent()
    new = _opponent_label() or "nobody (both sides human)"
    if new != old:
        game_log.event("opponent_change", ply=len(game.board.move_stack), old=old, new=new)
    return state()


@app.post("/api/move")
def make_move(req: MoveIn):
    if game.over():
        raise HTTPException(409, "The game is over")
    if game.engine_to_move():
        raise HTTPException(409, "It's the engine's turn")
    try:
        mv = chess.Move.from_uci(req.uci)
    except ValueError:
        raise HTTPException(400, f"Not a UCI move: {req.uci}")
    if mv not in game.board.legal_moves:
        raise HTTPException(400, f"Illegal move: {req.uci}")
    before = game.board.copy()
    game.board.push(mv)
    _log_move(before, mv)
    return state()


@app.post("/api/engine_move")
async def engine_move():
    if not game.engine_to_move():
        raise HTTPException(409, "Not the engine's turn")
    eng = get_engine(game.engine_name)  # type: ignore[arg-type]
    ok, why = await eng.available()
    if not ok:
        raise HTTPException(503, why)
    board_now, before = game.board, game.board.copy()
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
            mv, details = await eng.choose_move_detailed(before.copy())
    except Exception as e:  # surface engine errors to the UI
        game_log.event("engine_error", ply=len(before.move_stack) + 1, engine=eng.name, error=str(e))
        raise HTTPException(500, f"{eng.name} failed: {e}")
    # The engine may have thought for minutes. If you started a new game, took a move
    # back, or swapped opponents meanwhile, its move belongs to a position that's gone.
    if (game.board is not board_now or game.board.fen() != before.fen()
            or len(game.board.move_stack) != len(before.move_stack)
            or game.engine_name != eng.name or not game.engine_to_move()):
        game_log.event("engine_move_discarded", engine=eng.name, uci=mv.uci(),
                       reason="the position changed while the engine was thinking")
        raise HTTPException(409, "The position changed while the engine was thinking; its move was discarded")
    if mv not in game.board.legal_moves:
        game_log.event("engine_error", ply=len(before.move_stack) + 1, engine=eng.name,
                       error=f"illegal move {mv.uci()}")
        raise HTTPException(500, f"{eng.name} suggested an illegal move: {mv.uci()}")
    game.board.push(mv)
    game.engine_explanation = explanation
    _log_move(before, mv, explanation if eng.can_explain else details)
    return state()


@app.post("/api/undo")
def undo():
    """Hotseat: take back one ply. Engine mode: take back to the human's turn."""
    if not game.board.move_stack:
        return state()
    was_over = game.over()
    taken = [game.board.pop().uci()]
    if game.mode == "engine" and game.board.move_stack and game.board.turn != game.human_color:
        taken.insert(0, game.board.pop().uci())
    game_log.event("undo", moves_taken_back=taken, fen_after=game.board.fen())
    game_log.write_pgn(pgn())
    if was_over:
        game_log.reopen()
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
    if game.over():
        raise HTTPException(400, "Game is over")
    board = game.board.copy()  # the position asked about, even if the game moves on meanwhile
    try:
        a = await eng.analyze(board.copy(), req.question)
    except Exception as e:
        raise HTTPException(500, f"{eng.name} failed: {e}")
    payload = analysis_payload(board, a, eng.name)
    game_log.event("analysis", ply=len(board.move_stack), fen=board.fen(),
                   engine=eng.name, question=req.question, response=payload)
    return payload


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
