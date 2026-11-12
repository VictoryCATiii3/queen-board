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



def test_queen_config_reads_env_when_created(monkeypatch):
    """Regression: defaults used to be frozen when the module was first imported."""
    from queen_portable.config import QueenConfig
    monkeypatch.setenv("QUEEN_DEVICE", "cpu")
    assert QueenConfig().device == "cpu"
    monkeypatch.delenv("QUEEN_DEVICE")
    assert QueenConfig().device == "auto"


def test_queen_command_applies_settings_file(tmp_path, monkeypatch):
    import os
    from queen_portable.cli import _apply_settings_file
    from queen_portable.config import QueenConfig
    for k in ("QUEEN_DEVICE", "QUEEN_MODEL", "HF_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("QUEEN_MODEL", "pawn-8")   # set by the user: must win over the file
    _apply_settings_file(write(tmp_path, "cpu: true\nmodel: hce-4\ndownload_folder: /drive/hf\n"))
    cfg = QueenConfig()
    assert (cfg.device, cfg.model, os.environ["HF_HOME"]) == ("cpu", "pawn-8", "/drive/hf")



def test_default_model_is_pinned_to_a_tested_commit():
    import re
    from queen_portable.config import PINNED_REVISIONS, QueenConfig
    sha = PINNED_REVISIONS[QueenConfig(model="pawn-8").repo_id]
    assert sha and re.fullmatch(r"[0-9a-f]{40}", sha)
