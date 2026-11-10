import chess

from queen_portable.backends.vllm_subprocess import parse_upstream_stdout
from queen_portable.providers import board_history
from queen_portable.types import Analysis


def test_parse_skips_logs():
    out = ('INFO vllm loading {weird brace}\nWARNING something\n'
           '{\n  "text": "Develop the knight.",\n  "best_move_uci": "g1f3",\n'
           '  "move_source": "model",\n  "critical_line": ["g1f3", "d7d5"]\n}\n')
    data = parse_upstream_stdout(out)
    a = Analysis.from_upstream(data)
    assert a.best_move_uci == "g1f3" and a.critical_line == ["g1f3", "d7d5"]


def test_history_excludes_current():
    b = chess.Board()
    b.push_uci("e2e4"); b.push_uci("e7e5")
    h = board_history(b)
    assert len(h) == 2 and h[0] == chess.STARTING_FEN and b.fen() not in h
