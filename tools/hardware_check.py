"""Look at this computer and decide how QUEEN should run, before it loads.

QUEEN needs about 9 GB in its normal (bfloat16) form. This picks:
  - the NVIDIA GPU as-is if it has room (10 GB+),
  - the NVIDIA GPU with the model shrunk to 4-bit (nf4) if it has 5.5-10 GB,
  - the CPU if there's no usable GPU,
and warns when free memory is too low for the choice.

Anything the user already chose (cpu: true, quantize: ..., QUEEN_DEVICE, ...)
is respected, never overridden.

    python tools/hardware_check.py --shell   # export lines for bash launchers
    python tools/hardware_check.py --cmd     # NAME=value lines for the .bat
    python tools/hardware_check.py           # just explain

Messages for people go to stderr; only settings go to stdout.
"""
from __future__ import annotations

import ctypes
import os
import sys

FULL_GB = 10.0     # GPU memory to run QUEEN as-is (8.4 GB weights + working room)
NF4_GB = 5.5       # GPU memory to run the 4-bit version
CPU_RAM_GB = 10.0  # free RAM to run on the CPU without heavy swapping


def free_ram_gb() -> float | None:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except Exception:
        pass
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPhys / 2**30
        return None
    try:  # Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def cuda_gpu() -> tuple[str, float] | None:
    """(name, total GB) of the first NVIDIA GPU PyTorch can use, else None."""
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return p.name, p.total_memory / 2**30
    except Exception:
        pass
    return None


def decide(env: dict, gpu: tuple[str, float] | None, ram_gb: float | None,
           has_bnb: bool) -> tuple[dict[str, str], list[str]]:
    """Pure decision logic (tested without hardware). Returns (settings, messages)."""
    out: dict[str, str] = {}
    msgs: list[str] = []
    if env.get("QUEEN_URL"):
        return out, msgs
    device = (env.get("QUEEN_DEVICE") or "auto").lower()
    quantize = env.get("QUEEN_QUANTIZE")

    if device in ("auto", "cuda") and gpu:
        name, vram = gpu
        if quantize:
            msgs.append(f"GPU: {name} ({vram:.0f} GB), QUEEN shrunk to {quantize} as set.")
        elif vram >= FULL_GB:
            msgs.append(f"GPU: {name} ({vram:.0f} GB). QUEEN will run on it at full size.")
        elif vram >= NF4_GB:
            out["QUEEN_QUANTIZE"] = "nf4"
            quantize = "nf4"
            msgs.append(f"GPU: {name} has {vram:.0f} GB, less than the ~{FULL_GB:.0f} GB QUEEN needs at full "
                        "size, so it will run shrunk to 4-bit (nf4). This is experimental: if moves or "
                        "explanations look broken, set cpu: true in setup/settings.yaml.")
        else:
            out["QUEEN_DEVICE"] = "cpu"
            device = "cpu"
            msgs.append(f"GPU: {name} has only {vram:.0f} GB, too small for QUEEN even shrunk. "
                        "Using the CPU instead.")
        if quantize and not has_bnb:
            out["QUEEN_NEEDS_BNB"] = "1"
        if device != "cpu":
            return out, msgs

    if device == "cuda" and not gpu:
        msgs.append("No usable NVIDIA GPU found (or the driver is too old), but the GPU was requested.")
    if device in ("auto", "cuda") and not gpu:
        if sys.platform == "darwin":
            return out, msgs  # the Mac launcher handles Apple GPUs
        out["QUEEN_DEVICE"] = "cpu"
        msgs.append("No usable NVIDIA GPU found (if you have one, its driver may be too old: update it "
                    "from nvidia.com). QUEEN will run on the CPU.")
        device = "cpu"

    if device == "cpu":
        if quantize:
            out["QUEEN_QUANTIZE"] = ""
            msgs.append("Note: quantize only works on NVIDIA GPUs; ignoring it on the CPU.")
        msgs.append("On the CPU each QUEEN move can take many minutes. Stockfish and the board run normally.")
        if ram_gb is not None and ram_gb < CPU_RAM_GB:
            msgs.append(f"WARNING: only {ram_gb:.1f} GB of memory is free; QUEEN needs about {CPU_RAM_GB:.0f} GB "
                        "on the CPU. Close other programs (especially browser tabs) or it will be extremely slow.")
    return out, msgs


def main(argv: list[str]) -> int:
    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except Exception:
        has_bnb = False
    settings, msgs = decide(dict(os.environ), cuda_gpu(), free_ram_gb(), has_bnb)
    for m in msgs:
        print(m, file=sys.stderr)
    for k, v in settings.items():
        if "--shell" in argv:
            print(f"export {k}='{v}'")
        elif "--cmd" in argv:
            print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
