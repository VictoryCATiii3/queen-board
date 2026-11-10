"""Which checkpoint, which device, which backend. Everything else reads from here.

Zero-config by default: with nothing set, the model downloads itself into the
standard Hugging Face cache on first use and the best device is picked
automatically. Every setting can be overridden by a command-line flag, or by an
environment variable for scripts.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

# Both released checkpoints share the same code; switching is just this id.
MODELS = {
    "pawn-8": "princeton-nlp/queen_pawn-8",
    "hce-4": "princeton-nlp/queen_hce-4",
}
# Exact upstream commits we've tested against. Pinning means a future upstream
# change can't silently break installs. None = latest (until we record a sha).
PINNED_REVISIONS: dict[str, str | None] = {
    "princeton-nlp/queen_pawn-8": None,
    "princeton-nlp/queen_hce-4": None,
}


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


@dataclass
class QueenConfig:
    model: str = _env("QUEEN_MODEL") or "pawn-8"
    # A folder that already holds a checkpoint. Skips downloading entirely.
    model_dir: str | None = _env("QUEEN_MODEL_DIR")
    revision: str | None = _env("QUEEN_REVISION")
    backend: str = _env("QUEEN_BACKEND") or "torch"        # "torch" | "vllm"
    device: str = _env("QUEEN_DEVICE") or "auto"           # "auto" | "cuda" | "mps" | "cpu"
    quantize: str | None = _env("QUEEN_QUANTIZE")          # None | "int8" | "nf4" (NVIDIA only)
    # Python of upstream's own venv; only the vllm reference backend needs it.
    upstream_python: str | None = _env("QUEEN_UPSTREAM_PYTHON")

    @property
    def repo_id(self) -> str:
        return MODELS.get(self.model, self.model)  # also accepts a full repo id

    def resolve_model_dir(self, download: bool = True) -> Path:
        """Local folder holding the checkpoint, downloading it on first use."""
        if self.model_dir:
            path = Path(self.model_dir).expanduser()
            if not path.is_dir():
                raise FileNotFoundError(f"model folder {path} doesn't exist")
            return path
        from huggingface_hub import snapshot_download
        revision = self.revision or PINNED_REVISIONS.get(self.repo_id)
        try:  # already downloaded? no network needed
            return Path(snapshot_download(self.repo_id, revision=revision, local_files_only=True))
        except Exception:
            if not download:
                raise
        print(f"Downloading {self.repo_id} (~8.4 GiB, first run only)...")
        return Path(snapshot_download(self.repo_id, revision=revision))


def pick_device(preferred: str = "auto") -> str:
    if preferred != "auto":
        return preferred
    import torch  # lazy so non-model pieces stay light
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def add_config_args(parser) -> None:
    """Shared command-line flags. Flags win over environment variables."""
    g = parser.add_argument_group("model")
    g.add_argument("--model", help="pawn-8 (default), hce-4, or a Hugging Face repo id")
    g.add_argument("--model-dir", help="use a checkpoint folder you already have")
    g.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    g.add_argument("--quantize", choices=["int8", "nf4"], help="NVIDIA only; for GPUs under ~10 GB")
    g.add_argument("--backend", choices=["torch", "vllm"], help=argparse.SUPPRESS)


def config_from_args(args) -> QueenConfig:
    cfg = QueenConfig()
    for name in ("model", "model_dir", "device", "quantize", "backend"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    return cfg
