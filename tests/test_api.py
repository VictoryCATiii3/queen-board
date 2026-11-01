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
from server.engines.queen import parse_queen_output  # noqa: E402

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


def test_queen_parser_trims_illegal_line():
    board = chess.Board()
    out = "loading model...\n" + json.dumps({
        "best_move": "e2e4", "evaluation": 0.3, "analysis": "Center control.",
        "critical_line": ["e2e4", "e7e5", "g1f3", "d8e5"],  # last one illegal
    })
    a = parse_queen_output(board, out)
    assert a.best_move == "e2e4" and a.line == ["e2e4", "e7e5", "g1f3"]
    assert a.eval_cp == 30 and "Center" in a.text


def test_engines_listing():
    names = {e["name"] for e in c.get("/api/engines").json()}
    assert names == {"random", "stockfish", "queen"}


@pytest.mark.skipif(sys.platform == "win32", reason="Queen is off on Windows")
def test_queen_subprocess_with_fake_infer(tmp_path, monkeypatch):
    """Fake queen_hce-4 folder: checks flags, history file and JSON parsing end to end."""
    (tmp_path / ".venv/bin").mkdir(parents=True)
    (tmp_path / ".venv/bin/python").symlink_to(sys.executable)
    (tmp_path / "infer.py").write_text(
        "import argparse, json, chess\n"
        "p = argparse.ArgumentParser(); p.add_argument('--fen'); p.add_argument('--history')\n"
        "p.add_argument('--gpu-memory-utilization')\n"
        "a = p.parse_args(); hist = json.load(open(a.history))\n"
        "b = chess.Board(a.fen); mv = sorted(b.legal_moves, key=lambda m: m.uci())[-1]\n"
        "print('INFO vllm noise {not json}')\n"
        "print(json.dumps({'text': f'history={len(hist)}', 'best_move_uci': mv.uci(), 'move_source': 'model'}))\n"
    )
    monkeypatch.setenv("QUEEN_DIR", str(tmp_path))
    import server.app as srv
    srv._engines.pop("queen", None)
    c.post("/api/new", json={"mode": "engine", "engine": "queen"})
    c.post("/api/move", json={"uci": "e2e4"})
    s = c.post("/api/engine_move", json={}).json()
    assert len(s["history"]) == 2
    a = c.post("/api/analyze", json={"engine": "queen"}).json()
    assert a["text"] == "history=2" and a["best_san"]


def test_queen_is_off_on_windows(monkeypatch):
    from server.engines.queen import QueenEngine
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("QUEEN_CMD", raising=False)
    ok, why = asyncio.run(QueenEngine().available())
    assert not ok and "CPU" in why


def test_queen_disable_flag(monkeypatch):
    from server.engines.queen import QueenEngine
    monkeypatch.setenv("QUEEN_DISABLE", "1")
    ok, _ = asyncio.run(QueenEngine().available())
    assert not ok


def test_stockfish_found_in_repo_bin(tmp_path, monkeypatch):
    import server.engines.stockfish as sf
    exe = tmp_path / "stockfish" / "stockfish-ubuntu-x86-64"
    exe.parent.mkdir()
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    monkeypatch.setattr(sf, "REPO_BIN", tmp_path)
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    found = sf._find_binary()
    assert found and found.endswith("stockfish-ubuntu-x86-64")


def test_stockfish_asset_picking():
    sys.path.insert(0, str(HERE.parent / "tools"))
    from get_stockfish import pick_assets
    old_style = ["stockfish-windows-x86-64-avx2.zip", "stockfish-windows-x86-64.zip",
                 "stockfish-windows-x86-64-avx512.zip", "stockfish-ubuntu-x86-64-avx2.tar",
                 "stockfish-macos-m1-apple-silicon.tar"]
    assert pick_assets(old_style, "nt", "AMD64", avx2=True)[:2] == [
        "stockfish-windows-x86-64.zip", "stockfish-windows-x86-64-avx2.zip"]
    assert pick_assets(old_style, "nt", "AMD64", avx2=False)[-1] == "stockfish-windows-x86-64-avx2.zip"
    assert pick_assets(old_style, "linux", "x86_64", avx2=True) == ["stockfish-ubuntu-x86-64-avx2.tar"]
    assert pick_assets(old_style, "darwin", "arm64", avx2=False) == ["stockfish-macos-m1-apple-silicon.tar"]
    universal = ["stockfish-windows.zip", "stockfish-linux.tar", "stockfish-windows-arm64.zip"]
    assert pick_assets(universal, "nt", "AMD64", avx2=True) == ["stockfish-windows.zip"]


@pytest.mark.skipif(sys.platform == "win32", reason="runs a .py via shebang")
def test_stockfish_verify_handshake():
    sys.path.insert(0, str(HERE.parent / "tools"))
    from get_stockfish import works
    assert works(HERE / "fake_uci.py")
    assert not works(HERE / "does-not-exist")
