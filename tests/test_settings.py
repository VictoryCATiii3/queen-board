import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import settings  # noqa: E402


def write(tmp_path, text):
    p = tmp_path / "settings.yaml"
    p.write_text(text)
    return p


def test_shipped_file_has_only_defaults():
    values, warnings = settings.read_settings()
    assert not warnings
    assert values["cpu"] is False
    assert settings.to_env(values, environ={}) == {
        "QUEEN_MODEL": "pawn-8", "PORT": "8000", "SAVE_GAMES": "1", "STOCKFISH_TIME": "0.5"}


def test_cpu_true_and_paths(tmp_path):
    p = write(tmp_path, "cpu: TRUE   # force it\ndownload_folder: D:\\hf models\nopen_browser: no\n")
    values, warnings = settings.read_settings(p)
    assert not warnings
    env = settings.to_env(values, environ={})
    assert env == {"QUEEN_DEVICE": "cpu", "HF_HOME": "D:\\hf models", "NO_BROWSER": "1"}


def test_command_line_wins(tmp_path):
    values, _ = settings.read_settings(write(tmp_path, "cpu: true\nport: 9000\n"))
    env = settings.to_env(values, environ={"PORT": "8080", "QUEEN_DEVICE": "cuda"})
    assert env == {}


def test_bad_values_warn_not_crash(tmp_path):
    values, warnings = settings.read_settings(write(tmp_path, "cpu: maybe\nquantize: q4\ncolour: blue\nport 8000\n"))
    assert values == {} and len(warnings) == 4
