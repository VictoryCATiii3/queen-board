"""Download a checkpoint ahead of time (optional: it also downloads on first use).

    queen fetch                  # pawn-8
    queen fetch --model hce-4
"""
from __future__ import annotations

from .config import QueenConfig


def fetch(cfg: QueenConfig) -> str:
    return str(cfg.resolve_model_dir(download=True))
