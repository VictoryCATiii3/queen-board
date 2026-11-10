"""Plain data types shared by every piece. No heavy imports here on purpose."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class AnalyzeRequest:
    fen: str
    history: list[str] = field(default_factory=list)  # prior FENs, oldest first, excluding `fen`
    game_id: int = 0
    # Custom prompt (e.g. a user's question). None = the evaluation prompt QUEEN was
    # trained/evaluated with. Free-form prompts are experimental.
    prompt: str | None = None


@dataclass
class Analysis:
    best_move_uci: str | None      # None only if the model's answer had no legal move
    text: str                      # the explanation shown to people
    move_source: str = "best_move" # upstream labels random-fallback moves here
    critical_line: Any = None      # upstream's parsed critical-line dict
    raw: dict[str, Any] = field(default_factory=dict)  # full upstream-format JSON

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_upstream(cls, data: dict[str, Any]) -> "Analysis":
        return cls(
            best_move_uci=data.get("best_move_uci"),
            text=data.get("text", ""),
            move_source=data.get("move_source", "best_move"),
            critical_line=data.get("critical_line"),
            raw=data,
        )
