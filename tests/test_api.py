"""Run:  pytest -q   (from the repo root)"""
import asyncio
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("STOCKFISH_PATH", str(HERE / "fake_uci.py"))
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
    assert not ok and "queen serve" in why
