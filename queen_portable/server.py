"""Tiny local HTTP server so other machines and apps can ask QUEEN about a position.

POST /analyze  {"fen": "...", "history": [...], "game_id": 0, "prompt": null}  ->  Analysis JSON
GET  /health

Standard library only. Meant for your home network, not the internet.
Start it with `queen serve`.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .backends import load_backend
from .config import QueenConfig
from .types import AnalyzeRequest


def make_handler(backend):
    lock = threading.Lock()  # one model, one request at a time

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/analyze":
                return self._send(404, {"error": "not found"})
            try:
                data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                req = AnalyzeRequest(fen=data["fen"], history=data.get("history", []),
                                     game_id=int(data.get("game_id", 0)),
                                     prompt=data.get("prompt") or None)
                with lock:
                    result = backend.analyze(req)
                self._send(200, result.to_dict())
            except Exception as e:  # report, don't crash the server
                self._send(500, {"error": str(e)})

    return Handler


def serve(cfg: QueenConfig, host: str = "127.0.0.1", port: int = 8765) -> None:
    backend = load_backend(cfg)
    print(f"QUEEN ready on http://{host}:{port}  (Ctrl+C to stop)")
    try:
        ThreadingHTTPServer((host, port), make_handler(backend)).serve_forever()
    except KeyboardInterrupt:
        pass
