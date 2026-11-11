"""Load a Hugging Face-style model folder without transformers' weight loader.

Why: QUEEN's Leela encoder stores unused features as *empty* tensors (e.g.
smolgen `compress_w` with shape [0, 1024]). Some platform/library combinations
(seen on Windows) crash slicing those in `from_pretrained`:
    Error during slicing [:, :] with shape [0, 1024]: index 0 out of bounds ...
Empty tensors hold no data, so we never read them: shapes come from the file's
JSON header, and only non-empty tensors are read.

Generic: works for any PreTrainedModel subclass whose weights are in one
model.safetensors next to config.json.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path


def safetensors_shapes(path: Path) -> dict[str, list[int]]:
    """Tensor name -> shape, read from the header only (no tensor data touched)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: v["shape"] for k, v in header.items() if k != "__metadata__"}


def load_pretrained_safely(model_cls, folder: str | Path, weights: str = "model.safetensors"):
    from safetensors import safe_open

    folder = Path(folder)
    config = model_cls.config_class.from_pretrained(str(folder), local_files_only=True)
    model = model_cls(config)
    own = model.state_dict()
    prefix = getattr(model_cls, "base_model_prefix", "") or ""

    def our_name(name: str) -> str:
        if name in own:
            return name
        if prefix and name.startswith(prefix + ".") and name[len(prefix) + 1:] in own:
            return name[len(prefix) + 1:]
        return name

    shapes = {our_name(k): (k, s) for k, s in safetensors_shapes(folder / weights).items()}
    unexpected = [k for k in shapes if k not in own]
    if unexpected:
        raise ValueError(f"{folder / weights} has weights this model doesn't know: {unexpected[:5]}")
    for name, (_, shape) in shapes.items():
        if tuple(own[name].shape) != tuple(shape):
            raise ValueError(f"shape mismatch for {name}: file {shape}, model {list(own[name].shape)}")
    missing = [k for k, t in own.items() if k not in shapes and t.numel() > 0]
    if missing:
        raise ValueError(f"{folder / weights} is missing weights: {missing[:5]}")

    tensors = {}
    with safe_open(str(folder / weights), framework="pt", device="cpu") as f:
        for name, (file_name, shape) in shapes.items():
            if 0 in shape:
                continue  # empty: nothing to read, the model already has the right empty tensor
            tensors[name] = f.get_tensor(file_name)
    model.load_state_dict(tensors, strict=False)
    return model.eval()
