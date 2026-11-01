"""Download Stockfish (a CPU-only chess engine) into bin/stockfish/.

Called by run.bat and run.sh with the project's .venv Python. Standard library
only, nothing installed system-wide. Does nothing if Stockfish is already found.

It asks GitHub for the latest release, picks the build for this OS/CPU, and
checks that the binary actually answers the UCI handshake before keeping it.
Never fails the launcher: the board works without Stockfish.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "bin" / "stockfish"
API = "https://api.github.com/repos/official-stockfish/Stockfish/releases/latest"
FEATURES = ("avx2", "avx512", "bmi2", "vnni", "sse41", "popcnt", "avxvnni", "zen")
ARM = ("arm", "aarch64", "apple-silicon", "m1")


def has_avx2() -> bool:
    try:
        if os.name == "nt":
            import ctypes
            return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))  # PF_AVX2_INSTRUCTIONS_AVAILABLE
        if sys.platform.startswith("linux"):
            return " avx2" in Path("/proc/cpuinfo").read_text()
    except Exception:
        pass
    return False


def pick_assets(names: list[str], os_name: str, machine: str, avx2: bool) -> list[str]:
    """Best-first list of release assets to try for this machine."""
    os_words = {"nt": ("windows",), "linux": ("ubuntu", "linux"), "darwin": ("macos", "mac")}[os_name]
    is_arm = machine.lower() in ("arm64", "aarch64")
    out = []
    for n in names:
        low = n.lower()
        if not any(w in low for w in os_words):
            continue
        if not low.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            continue
        if any(a in low for a in ARM) != is_arm:
            continue
        out.append(n)

    def rank(n: str) -> int:
        low = n.lower()
        feats = [f for f in FEATURES if f in low]
        if not feats:
            return 0  # universal / generic build
        if feats == ["avx2"]:
            return 1 if avx2 else 9
        if "sse41" in feats or "popcnt" in feats:
            return 2
        return 5  # avx512, bmi2, ... only if nothing else

    return sorted(out, key=rank)


def find_binary(folder: Path) -> Path | None:
    for p in sorted(folder.rglob("stockfish*")):
        if not p.is_file():
            continue
        if os.name == "nt" and p.suffix.lower() == ".exe":
            return p
        if os.name != "nt" and p.suffix == "":
            p.chmod(p.stat().st_mode | 0o111)
            return p
    return None


def works(exe: Path | str) -> bool:
    try:
        r = subprocess.run([str(exe)], input="uci\nquit\n", capture_output=True, text=True, timeout=20)
        return "uciok" in r.stdout
    except Exception:
        return False


def extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest, filter="data")
            except TypeError:  # Python without extraction filters
                t.extractall(dest)


def already_installed() -> str | None:
    sys.path.insert(0, str(ROOT))
    try:
        from server.engines.stockfish import _find_binary
        return _find_binary()
    except Exception:
        return None


def main() -> int:
    found = already_installed()
    if found:
        print(f"Stockfish: using {found}")
        return 0
    if os.environ.get("SKIP_STOCKFISH") == "1":
        return 0
    os_name = "nt" if os.name == "nt" else sys.platform if sys.platform == "darwin" else "linux"
    try:
        with urllib.request.urlopen(urllib.request.Request(API, headers={"User-Agent": "queen-board"}), timeout=30) as r:
            release = json.load(r)
    except Exception as e:
        print(f"Stockfish: couldn't reach GitHub ({e}). Skipping; the board still works.")
        return 0
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
    choices = pick_assets(list(assets), os_name, platform.machine(), has_avx2())
    if not choices:
        print(f"Stockfish: no build found for this machine in {release.get('tag_name')}. Assets: {list(assets)}")
        return 0
    for name in choices:
        print(f"Stockfish: downloading {release.get('tag_name')} / {name} ...")
        with tempfile.TemporaryDirectory() as tmp:
            arc = Path(tmp) / name
            try:
                with urllib.request.urlopen(assets[name], timeout=300) as r, open(arc, "wb") as f:
                    shutil.copyfileobj(r, f)
                staging = Path(tmp) / "x"
                extract(arc, staging)
            except Exception as e:
                print(f"  failed: {e}")
                continue
            exe = find_binary(staging)
            if exe and works(exe):
                if DEST.exists():
                    shutil.rmtree(DEST)
                DEST.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging), str(DEST))
                print(f"Stockfish: installed to {find_binary(DEST)}")
                return 0
            print("  that build doesn't run on this CPU, trying the next one")
    print("Stockfish: no working build found. Skipping; the board still works.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
