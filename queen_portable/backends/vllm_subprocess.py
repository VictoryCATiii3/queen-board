"""Linux + NVIDIA only: call Princeton's own infer.py unchanged.

This is the reference every other backend is checked against. It reloads the
model on every call (slow), which is fine for a baseline.
"""
from __future__ import annotations

import json
import tempfile
import subprocess
from pathlib import Path

from ..config import QueenConfig
from ..types import Analysis, AnalyzeRequest


def parse_upstream_stdout(stdout: str) -> dict:
    """infer.py prints library logs, then one JSON object. Return that object."""
    decoder = json.JSONDecoder()
    found = None
    for i, ch in enumerate(stdout):
        if ch != "{" or (i > 0 and stdout[i - 1] != "\n"):
            continue  # only try objects that start a line
        try:
            obj, _ = decoder.raw_decode(stdout, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "best_move_uci" in obj:
            found = obj  # keep the last one
    if found is None:
        raise RuntimeError("no QUEEN JSON found in infer.py output:\n" + stdout[-2000:])
    return found


class VllmSubprocessBackend:
    def __init__(self, cfg: QueenConfig):
        self.cfg = cfg
        model_dir = cfg.resolve_model_dir()
        self.infer = model_dir / "infer.py"
        if cfg.upstream_python:
            py = Path(cfg.upstream_python).expanduser()
            if not py.exists():
                raise FileNotFoundError(f"QUEEN_UPSTREAM_PYTHON points to {py}, which doesn't exist "
                                        "(full paths start with / or ~)")
        else:
            py = model_dir / ".venv" / "bin" / "python"
            if not py.exists():
                raise FileNotFoundError(f"no upstream venv at {py}; set QUEEN_UPSTREAM_PYTHON to the "
                                        "python inside the venv that has vLLM installed")
        self.python = str(py)

    def analyze(self, req: AnalyzeRequest, *, temperature: float | None = None) -> Analysis:
        cmd = [self.python, str(self.infer), "--fen", req.fen, "--game-id", str(req.game_id)]
        if req.prompt:
            cmd += ["--prompt", req.prompt]
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]
        with tempfile.TemporaryDirectory() as tmp:
            if req.history:
                hist = Path(tmp) / "history.json"
                hist.write_text(json.dumps(req.history))
                cmd += ["--history", str(hist)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"infer.py failed ({proc.returncode}):\n{proc.stderr[-2000:]}")
        return Analysis.from_upstream(parse_upstream_stdout(proc.stdout))
