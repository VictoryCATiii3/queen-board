"""Backends turn a position into an Analysis. Swap them without touching anything else."""
from __future__ import annotations

from typing import Protocol

from ..config import QueenConfig
from ..types import Analysis, AnalyzeRequest


class Backend(Protocol):
    def analyze(self, req: AnalyzeRequest) -> Analysis: ...


def load_backend(cfg: QueenConfig) -> Backend:
    if cfg.backend == "vllm":
        from .vllm_subprocess import VllmSubprocessBackend
        return VllmSubprocessBackend(cfg)
    if cfg.backend == "torch":
        from .torch_native import TorchNativeBackend
        return TorchNativeBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r}")
