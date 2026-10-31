from .base import Analysis, Engine
from .queen import QueenEngine
from .random_engine import RandomEngine
from .stockfish import StockfishEngine

ENGINES: dict[str, type[Engine]] = {
    "random": RandomEngine,
    "stockfish": StockfishEngine,
    "queen": QueenEngine,
}

__all__ = ["Analysis", "Engine", "ENGINES", "QueenEngine", "RandomEngine", "StockfishEngine"]
