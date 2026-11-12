"""Run:  pytest -q   (from the repo root)"""
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("STOCKFISH_PATH", str(HERE / "fake_uci.py"))
import tempfile  # noqa: E402
LOG_DIR = Path(tempfile.mkdtemp(prefix="queen-board-games-"))
os.environ["GAMES_DIR"] = str(LOG_DIR)   # never write test games into the repo
os.chmod(HERE / "fake_uci.py", 0o755)
sys.path.insert(0, str(HERE.parent))

import chess  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402
from server.engines.queen import parse_queen_reply  # noqa: E402

c = TestClient(app)
c.__enter__()  # one event loop for the whole module (engine subprocesses live on it)


def test_hotseat_game_and_illegal_move():
    s = c.post("/api/new", json={"mode": "hotseat"}).json()
    assert s["turn"] == "white" and "e2e4" in s["legal"]
    s = c.post("/api/move", json={"uci": "e2e4"}).json()
    assert s["turn"] == "black" and s["history"] == ["e4"]
    assert c.post("/api/move", json={"uci": "e2e4"}).status_code == 400
    s = c.post("/api/undo", json={}).json()
    assert s["history"] == []


def test_fools_mate_result():
    c.post("/api/new", json={"mode": "hotseat"})
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        s = c.post("/api/move", json={"uci": m}).json()
    assert s["result"].startswith("Black wins")
    assert s["check"] == "e1"


def test_promotion():
    c.post("/api/new", json={"mode": "hotseat", "fen": "8/P7/8/8/8/8/k7/4K3 w - - 0 1"})
    s = c.post("/api/move", json={"uci": "a7a8n"}).json()
    assert s["fen"].startswith("N7")


def test_vs_random_as_black():
    s = c.post("/api/new", json={"mode": "engine", "engine": "random", "human_color": "black"}).json()
    assert s["engine_to_move"]
    assert c.post("/api/move", json={"uci": "e2e4"}).status_code == 409
    s = c.post("/api/engine_move", json={}).json()
    assert s["turn"] == "black" and len(s["history"]) == 1


def test_uci_engine_play_and_analyze():
    s = c.post("/api/new", json={"mode": "engine", "engine": "stockfish"}).json()
    s = c.post("/api/move", json={"uci": "d2d4"}).json()
    s = c.post("/api/engine_move", json={}).json()
    assert len(s["history"]) == 2
    a = c.post("/api/analyze", json={"engine": "stockfish"}).json()
    assert a["best_move"] and a["best_san"] and a["eval_cp"] == 25


def test_queen_reply_parsing():
    board = chess.Board()
    reply = {
        "best_move_uci": "e2e4", "move_source": "best_move", "text": "Center control.",
        "critical_line": {"legal_steps": [{"uci": "e2e4"}, {"uci": "e7e5"}, {"uci": "g1f3"},
                                          {"uci": "d8e5"}]},  # last one illegal
        "raw": {"raw_text": "...\nEVALUATION: +0.30 from the player's perspective."},
    }
    a = parse_queen_reply(board, reply)
    assert a.best_move == "e2e4" and a.line == ["e2e4", "e7e5", "g1f3"]
    assert a.eval_cp == 30 and "Center" in a.text


def test_queen_eval_is_white_pov_and_mate():
    board = chess.Board("4k3/8/8/8/8/8/8/4K3 b - - 0 1")  # black to move
    a = parse_queen_reply(board, {"best_move_uci": "e8d8", "text": "",
                                  "raw": {"raw_text": "EVALUATION: +1.5 for the player"}})
    assert a.eval_cp == -150
    a = parse_queen_reply(chess.Board(), {"best_move_uci": "e2e4", "text": "",
                                          "raw": {"raw_text": "EVALUATION: +M1 from the player's perspective."}})
    assert a.mate_in == 1 and a.eval_cp is None


def test_engines_listing():
    names = {e["name"] for e in c.get("/api/engines").json()}
    assert names == {"random", "stockfish", "queen"}


@pytest.fixture
def fake_queen(monkeypatch):
    """A stand-in for `queen serve`: records requests, answers with the last legal move."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    seen = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            self._send({"ok": True})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(data)
            b = chess.Board(data["fen"])
            mv = sorted(b.legal_moves, key=lambda m: m.uci())[-1]
            self._send({"best_move_uci": mv.uci(), "move_source": "best_move",
                        "text": f"history={len(data['history'])}", "critical_line": None,
                        "raw": {"raw_text": "EVALUATION: +0.5"}})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("QUEEN_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    import server.app as app_mod
    app_mod._engines.pop("queen", None)
    yield seen
    srv.shutdown()
    app_mod._engines.pop("queen", None)


def test_queen_over_http(fake_queen):
    c.post("/api/new", json={"mode": "engine", "engine": "queen"})
    c.post("/api/move", json={"uci": "e2e4"})
    s = c.post("/api/engine_move", json={}).json()
    assert len(s["history"]) == 2
    a = c.post("/api/analyze", json={"engine": "queen", "question": "what is the plan?"}).json()
    assert "history=2" in a["text"] and a["best_san"]
    assert fake_queen[-1]["prompt"] == "what is the plan?" and len(fake_queen[-1]["history"]) == 2


def test_queen_unavailable_message(monkeypatch):
    from server.engines.queen import QueenEngine
    monkeypatch.setenv("QUEEN_URL", "http://127.0.0.1:9")  # nothing listens here
    ok, why = asyncio.run(QueenEngine().available())
    assert not ok and "queen serve" in why and "QUEEN_URL" in why


def test_queen_move_comes_with_reasoning(fake_queen):
    c.post("/api/new", json={"mode": "engine", "engine": "queen"})
    c.post("/api/move", json={"uci": "e2e4"})
    s = c.post("/api/engine_move", json={}).json()
    ex = s["engine_explanation"]
    assert ex["ply"] == 2 and ex["best_san"] and "history=1" in ex["text"]
    assert len(fake_queen) == 1  # one call gives both the move and the reasoning
    s = c.post("/api/undo", json={}).json()
    assert s["engine_explanation"] is None


def test_outcome_for_banner():
    s = c.post("/api/new", json={"mode": "hotseat"}).json()
    assert s["outcome"] is None
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:  # fool's mate
        s = c.post("/api/move", json={"uci": m}).json()
    assert s["outcome"] == {"winner": "black", "reason": "checkmate"}


def test_question_echo_is_checked():
    board = chess.Board()
    ok = parse_queen_reply(board, {"best_move_uci": "e2e4", "text": "x",
                                   "raw": {"inference_settings": {"prompt": "why?"}}}, "why?")
    assert "received your question" in ok.text
    bad = parse_queen_reply(board, {"best_move_uci": "e2e4", "text": "x",
                                    "raw": {"inference_settings": {"prompt": "Analyze..."}}}, "why?")
    assert "Warning" in bad.text


def test_queen_runs_in_process_without_queen_url(monkeypatch):
    """No QUEEN_URL: the board loads QUEEN itself (here a stand-in backend), one process."""
    from queen_portable.types import Analysis as QAnalysis
    from server.engines import queen as qmod

    class FakeBackend:
        calls = []

        def analyze(self, req):
            self.calls.append(req)
            mv = sorted(chess.Board(req.fen).legal_moves, key=lambda m: m.uci())[0]
            return QAnalysis.from_upstream({"best_move_uci": mv.uci(), "text": "in-process",
                                            "raw_text": "EVALUATION: -0.2",
                                            "inference_settings": {"prompt": req.prompt}})

    fake = FakeBackend()
    monkeypatch.delenv("QUEEN_URL", raising=False)
    monkeypatch.setattr(qmod.LocalQueen, "_load", lambda self: fake)
    import server.app as app_mod
    app_mod._engines.pop("queen", None)
    assert isinstance(app_mod.get_engine("queen").transport, qmod.LocalQueen)
    c.post("/api/new", json={"mode": "engine", "engine": "queen"})
    c.post("/api/move", json={"uci": "e2e4"})
    s = c.post("/api/engine_move", json={}).json()
    assert s["engine_explanation"]["text"].startswith("in-process")
    assert len(fake.calls) == 1 and len(fake.calls[0].history) == 1
    app_mod._engines.pop("queen", None)


def test_swap_opponent_mid_game(fake_queen):
    c.post("/api/new", json={"mode": "engine", "engine": "random"})
    c.post("/api/move", json={"uci": "e2e4"})
    s = c.post("/api/engine_move", json={}).json()
    fen = s["fen"]
    s = c.post("/api/opponent", json={"mode": "engine", "engine": "queen"}).json()
    assert s["engine"] == "queen" and s["fen"] == fen and len(s["history"]) == 2  # position kept
    c.post("/api/move", json={"uci": "d2d4"})
    s = c.post("/api/engine_move", json={}).json()
    assert s["engine_explanation"]["engine"] == "queen"
    assert "Random mover / Queen 4B" in s["pgn"]
    s = c.post("/api/opponent", json={"mode": "hotseat"}).json()
    assert s["mode"] == "hotseat" and not s["engine_to_move"] and len(s["history"]) == 4



def _log_rows():
    from server.app import game_log
    return [json.loads(line) for line in game_log.path.read_text().splitlines()]


def test_game_log_records_moves_engines_and_result(fake_queen):
    c.post("/api/new", json={"mode": "engine", "engine": "stockfish"})
    c.post("/api/move", json={"uci": "f2f3"})
    c.post("/api/engine_move", json={})
    c.post("/api/opponent", json={"mode": "engine", "engine": "queen"})
    c.post("/api/move", json={"uci": "g2g4"})
    c.post("/api/engine_move", json={})
    c.post("/api/analyze", json={"engine": "queen", "question": "what now?"})
    rows = _log_rows()
    kinds = [r["event"] for r in rows]
    assert kinds[0] == "game_start" and "opponent_change" in kinds and "analysis" in kinds
    moves = [r for r in rows if r["event"] == "move"]
    assert [m["by"] for m in moves] == ["Human", "Stockfish (level 20)", "Human", "Queen 4B"]
    assert moves[0]["response"] is None                      # humans don't respond
    assert "line" in moves[1]["response"] or "depth" in moves[1]["response"]  # Stockfish's search info
    assert moves[3]["response"]["text"]                       # Queen's explanation
    assert [r for r in rows if r["event"] == "analysis"][0]["question"] == "what now?"
    from server.app import game_log
    assert "1. f3" in game_log.path.with_suffix(".pgn").read_text()


def test_game_log_result_once_and_hotseat_names():
    c.post("/api/new", json={"mode": "hotseat"})
    for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        c.post("/api/move", json={"uci": m})
    c.get("/api/state")
    rows = _log_rows()
    assert [r["by"] for r in rows if r["event"] == "move"][:2] == ["Human (White)", "Human (Black)"]
    over = [r for r in rows if r["event"] == "game_over"]
    assert len(over) == 1 and over[0]["result"] == "0-1" and over[0]["reason"] == "checkmate"
    c.post("/api/undo", json={})
    assert _log_rows()[-1]["event"] == "undo"


def test_save_games_off(monkeypatch):
    monkeypatch.setenv("SAVE_GAMES", "0")
    c.post("/api/new", json={"mode": "hotseat"})
    from server.app import game_log
    assert game_log.path is None


def test_engine_move_discarded_if_position_changed_while_thinking(monkeypatch):
    """A slow engine must never play its move into a different game."""
    import server.app as app_mod
    c.post("/api/new", json={"mode": "engine", "engine": "random", "human_color": "black"})
    eng = app_mod.get_engine("random")

    async def slow_move(board):
        # Meanwhile, the user starts a new game (same start position, same side to move).
        app_mod.new_game(app_mod.NewGame(mode="engine", engine="random", human_color="black"))
        return chess.Move.from_uci("e2e4"), None   # legal there too: must still be rejected

    monkeypatch.setattr(eng, "choose_move_detailed", slow_move)
    r = c.post("/api/engine_move", json={})
    assert r.status_code == 409 and "discarded" in r.json()["detail"]
    assert c.get("/api/state").json()["history"] == []     # nothing was played


def test_claimable_draw_ends_the_game_everywhere():
    c.post("/api/new", json={"mode": "hotseat"})
    for m in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:   # start position now seen 3 times
        assert c.post("/api/move", json={"uci": m}).status_code == 200
    s = c.get("/api/state").json()
    assert s["outcome"]["winner"] is None and "repetition" in s["outcome"]["reason"]
    assert s["legal"] == [] and "1/2-1/2" in s["pgn"]
    assert c.post("/api/move", json={"uci": "g1f3"}).status_code == 409
    s = c.post("/api/opponent", json={"mode": "engine", "engine": "random"}).json()
    assert s["engine_to_move"] is False          # an engine won't play on past the draw


def test_each_game_gets_its_own_queen_seed(fake_queen):
    ids = []
    for _ in range(2):
        c.post("/api/new", json={"mode": "engine", "engine": "queen", "human_color": "black"})
        c.post("/api/engine_move", json={})
        c.post("/api/analyze", json={"engine": "queen"})
        ids.append((fake_queen[-2]["game_id"], fake_queen[-1]["game_id"]))
    (a1, a2), (b1, b2) = ids
    assert a1 == a2 and b1 == b2 and a1 != b1 and a1 != 0   # same within a game, new per game
    from server.app import game_log
    start = json.loads(game_log.path.read_text().splitlines()[0])
    assert start["game_id"] == b1



def test_page_and_scripts_are_never_served_stale():
    """Safari reused an old board.js after updates when no cache rule was sent."""
    for path in ("/", "/static/board.js", "/static/style.css"):
        r = c.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache", path
