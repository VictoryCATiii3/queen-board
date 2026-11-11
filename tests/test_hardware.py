import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from hardware_check import decide  # noqa: E402

GPU24 = ("NVIDIA GeForce RTX 3090", 24.0)
GPU6 = ("NVIDIA GeForce RTX 3060 Laptop GPU", 6.0)
GPU4 = ("NVIDIA GeForce RTX 3050 Laptop GPU", 4.0)


def test_big_gpu_runs_full_size():
    assert decide({}, GPU24, 30, False)[0] == {}


def test_laptop_3060_gets_nf4_and_bnb_flag():
    out, msgs = decide({}, GPU6, 12, False)
    assert out == {"QUEEN_QUANTIZE": "nf4", "QUEEN_NEEDS_BNB": "1"} and "nf4" in msgs[0]


def test_tiny_gpu_falls_back_to_cpu_and_warns_low_ram():
    out, msgs = decide({}, GPU4, 2.2, True)
    assert out == {"QUEEN_DEVICE": "cpu"} and any("only 2.2 GB" in m for m in msgs)


def test_cpu_true_is_respected_even_with_a_gpu():
    out, msgs = decide({"QUEEN_DEVICE": "cpu"}, GPU24, 16, True)
    assert out == {} and any("CPU" in m for m in msgs)


def test_user_quantize_kept_on_gpu_cleared_on_cpu():
    assert decide({"QUEEN_QUANTIZE": "int8"}, GPU6, 16, True)[0] == {}
    assert decide({"QUEEN_QUANTIZE": "int8", "QUEEN_DEVICE": "cpu"}, None, 16, True)[0] == {"QUEEN_QUANTIZE": ""}


def test_no_gpu_means_cpu_except_mac_and_remote(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    assert decide({}, None, 16, False)[0] == {"QUEEN_DEVICE": "cpu"}
    monkeypatch.setattr(sys, "platform", "darwin")
    assert decide({}, None, 16, False)[0] == {}
    assert decide({"QUEEN_URL": "http://x:8765"}, None, 1, False) == ({}, [])
