"""Read setup/settings.yaml and hand the values to the launchers.

Every part of Queen Board already reads its options from environment variables;
this just turns the friendly settings file into those variables. A variable that
is already set (e.g. `PORT=8080 ./queen_runner_linux.sh`) is left alone, so the
command line always wins over the file.

    python tools/settings.py --shell     # export lines for bash (Linux/macOS launchers)
    python tools/settings.py --cmd       # set lines for Windows .bat
    python tools/settings.py             # show what would be used

Standard library only (it runs before the virtual environment exists). The file
format is the simple `key: value` subset of YAML, so no YAML library is needed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE = ROOT / "setup" / "settings.yaml"

TRUE = {"true", "yes", "on", "1"}
FALSE = {"false", "no", "off", "0"}


def _bool(key: str, value: str) -> bool:
    v = value.lower()
    if v in TRUE:
        return True
    if v in FALSE:
        return False
    raise ValueError(f"{key} should be true or false, not '{value}'")


def _choice(*allowed: str):
    def check(key: str, value: str) -> str:
        if value not in allowed:
            raise ValueError(f"{key} should be one of {', '.join(allowed)}, not '{value}'")
        return value
    return check


def _number(key: str, value: str) -> str:
    try:
        float(value)
    except ValueError:
        raise ValueError(f"{key} should be a number, not '{value}'") from None
    return value


def _text(key: str, value: str) -> str:
    return value


# setting -> (check, how it becomes environment variables)
SETTINGS = {
    "cpu":             (_bool, lambda v: {"QUEEN_DEVICE": "cpu"} if v else {}),
    "model":           (_text, lambda v: {"QUEEN_MODEL": v}),
    "quantize":        (_choice("int8", "nf4"), lambda v: {"QUEEN_QUANTIZE": v}),
    "download_folder": (_text, lambda v: {"HF_HOME": v}),
    "model_folder":    (_text, lambda v: {"QUEEN_MODEL_DIR": v}),
    "queen_url":       (_text, lambda v: {"QUEEN_URL": v}),
    "port":            (_number, lambda v: {"PORT": v}),
    "open_browser":    (_bool, lambda v: {} if v else {"NO_BROWSER": "1"}),
    "save_games":      (_bool, lambda v: {"SAVE_GAMES": "1" if v else "0"}),
    "stockfish_time":  (_number, lambda v: {"STOCKFISH_TIME": v}),
}


def read_settings(path: Path = DEFAULT_FILE) -> tuple[dict[str, object], list[str]]:
    """(settings, warnings). Missing file = no settings. Bad lines are skipped with a warning."""
    values: dict[str, object] = {}
    warnings: list[str] = []
    if not path.exists():
        return values, warnings
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.split(" #")[0].strip() if not line.lstrip().startswith("#") else ""
        if not line:
            continue
        if ":" not in line:
            warnings.append(f"line {n}: expected 'name: value', got '{line}'")
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        value = value.strip("'\"")
        key = key.lower()
        if key not in SETTINGS:
            warnings.append(f"line {n}: unknown setting '{key}' (ignored)")
            continue
        if value == "":
            continue  # blank = default
        check, _ = SETTINGS[key]
        try:
            values[key] = check(key, value)
        except ValueError as e:
            warnings.append(f"line {n}: {e} (ignored)")
    return values, warnings


def to_env(values: dict[str, object], environ=os.environ) -> dict[str, str]:
    """Environment variables to set, skipping any the user already set."""
    out: dict[str, str] = {}
    for key, value in values.items():
        for name, v in SETTINGS[key][1](value).items():
            if not environ.get(name):
                out[name] = str(v)
    return out


def _sh_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def main(argv: list[str]) -> int:
    values, warnings = read_settings()
    for w in warnings:
        print(f"setup/settings.yaml {w}", file=sys.stderr)
    env = to_env(values)
    if "--shell" in argv:
        for k, v in env.items():
            print(f"export {k}={_sh_quote(v)}")
    elif "--cmd" in argv:
        for k, v in env.items():
            print(f"{k}={v}")  # the .bat does: for /f "tokens=1* delims==" ... set "%%a=%%b"
    else:
        print(f"Settings file: {DEFAULT_FILE}")
        for k, v in values.items():
            print(f"  {k}: {v}")
        print("Environment it sets:" if env else "Environment it sets: nothing (all defaults)")
        for k, v in env.items():
            print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
