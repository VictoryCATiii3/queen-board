"""Plain PyTorch + transformers: no vLLM, so it runs on Mac (MPS), Windows (CUDA,
optionally quantized), and Linux. The model stays loaded between calls.

Pipeline (same as upstream's vLLM runner):
  FEN + history -> upstream lc0_planes  -> Leela BT5 encoder -> (1, 16, 64, 1024)
  states        -> XAttnBridge.set_board (K/V computed once per position)
  prompt        -> chat template        -> SmolLM3.generate() with bridges hooked in
  raw text      -> upstream choose_move -> best move, source, critical line

Upstream's helper modules (encoder, planes, move parsing, text translation) are
imported from the downloaded checkpoint folder rather than copied here.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

from ..config import QueenConfig, pick_device
from ..types import Analysis, AnalyzeRequest
from ..xattn import XAttnBridge

IM_END = "<|im_end|>"


def request_seed(base_seed: int, game_id: int, ply: int) -> int:
    """Identical to upstream infer.py, so seeds line up with their evaluation."""
    digest = hashlib.blake2b(f"{base_seed}|{game_id}|{ply}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def _import_upstream(model_dir: Path):
    """Make upstream's `models` and `utils` packages importable (they're plain torch/chess)."""
    path = str(model_dir.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
    from models.encoder import Lc0Bt4HFModel
    from utils.lc0_planes import encode_fen_batch
    from utils.utils import encode_planes, turn_tensor
    from utils.eval_policy import choose_move
    from utils.translate_helpers import Translator
    return Lc0Bt4HFModel, encode_fen_batch, encode_planes, turn_tensor, choose_move, Translator


class TorchNativeBackend:
    def __init__(self, cfg: QueenConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        model_dir = cfg.resolve_model_dir()
        if not (model_dir / "xattn.pt").exists():
            raise FileNotFoundError(f"{model_dir} has no xattn.pt; is it a QUEEN checkpoint folder?")
        (Lc0, self._encode_fen_batch, self._encode_planes, self._turn_tensor,
         self._choose_move, self._Translator) = _import_upstream(model_dir)

        self.settings = json.loads((model_dir / "inference_config.json").read_text())
        self.device = pick_device(cfg.device)
        self.dtype = getattr(torch, self.settings.get("dtype", "bfloat16"))

        # --- decoder (SmolLM3), optionally quantized ---------------------------------
        load_kw: dict = dict(dtype=self.dtype, device_map={"": self.device})
        if cfg.quantize:
            if self.device != "cuda":
                raise ValueError("QUEEN_QUANTIZE needs an NVIDIA GPU (bitsandbytes); run unquantized on mps/cpu")
            from transformers import BitsAndBytesConfig
            if cfg.quantize == "int8":
                qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])
            elif cfg.quantize == "nf4":
                qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=self.dtype,
                                        llm_int8_skip_modules=["lm_head"])
            else:
                raise ValueError(f"unknown quantize mode {cfg.quantize!r} (use int8 or nf4)")
            load_kw["quantization_config"] = qc
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kw).eval()
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.im_end = self.tok.convert_tokens_to_ids(IM_END)

        # --- bridges (always bf16, never quantized) ----------------------------------
        decoder_dim = self.model.config.hidden_size
        self.bridge = XAttnBridge(encoder_dim=1024, decoder_dim=decoder_dim)
        self.bridge.load_upstream(torch.load(model_dir / "xattn.pt", map_location="cpu"))
        self.bridge.to(device=self.device, dtype=self.dtype).eval()
        self.bridge.attach(self._decoder_layers())

        # --- board encoder (Leela BT5) ------------------------------------------------
        self.encoder = Lc0.from_pretrained(str(model_dir / "lc0"), local_files_only=True)
        self.encoder.to(device=self.device, dtype=self.dtype).eval()

    def _decoder_layers(self):
        m = self.model.model
        return m.layers if hasattr(m, "layers") else m.language_model.layers

    def _prompt_ids(self, prompt: str) -> torch.Tensor:
        out = self.tok.apply_chat_template([{"role": "user", "content": prompt}],
                                           tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False)
        if hasattr(out, "keys"):          # transformers 5 returns a BatchEncoding
            out = out["input_ids"]
        if out and isinstance(out[0], (list, tuple)):
            out = out[0]
        return torch.tensor([list(out)], device=self.device)

    @torch.no_grad()
    def _encode_board(self, fen: str, history: list[str]) -> torch.Tensor:
        planes = self._encode_fen_batch([fen], [history or None]).to(self.device)
        return self._encode_planes(self.encoder, planes, self.dtype,
                                   pov=True, turn=self._turn_tensor([fen]))   # (1, 16, 64, 1024)

    @torch.no_grad()
    def generate_raw(self, fen: str, history: list[str] | None = None, *, prompt: str | None = None,
                     temperature: float | None = None, max_tokens: int | None = None,
                     seed: int | None = None) -> str:
        s = self.settings
        temperature = s["temperature"] if temperature is None else temperature
        max_tokens = s["max_output_tokens"] if max_tokens is None else max_tokens
        ids = self._prompt_ids(prompt or s["prompt"])
        self.bridge.set_board(self._encode_board(fen, history or []))
        try:
            if seed is not None:
                torch.manual_seed(seed)
            gen_kw = dict(max_new_tokens=max_tokens, eos_token_id=[self.im_end],
                          pad_token_id=self.tok.pad_token_id or self.im_end,
                          repetition_penalty=s.get("repetition_penalty", 1.0))
            if temperature > 0:
                gen_kw.update(do_sample=True, temperature=temperature,
                              top_k=s.get("top_k", 20), top_p=s.get("top_p", 0.95))
            else:
                gen_kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), **gen_kw)
        finally:
            self.bridge.clear_board()
        new = out[0, ids.shape[1]:].tolist()
        if new and new[-1] == self.im_end:
            new = new[:-1]
        return self.tok.decode(new, skip_special_tokens=False)

    def analyze(self, req: AnalyzeRequest, *, temperature: float | None = None) -> Analysis:
        import chess

        board = chess.Board(req.fen)
        if not board.is_valid():
            raise ValueError("invalid chess position")
        seed = request_seed(self.settings["seed"], req.game_id, board.ply())
        raw = self.generate_raw(req.fen, req.history, prompt=req.prompt,
                                temperature=temperature, seed=seed)
        best, source, critical = self._choose_move(board.fen(), raw, seed, self.settings["move_policy"])
        legal_best = best if best is not None and best in board.legal_moves else None
        data = {
            "fen": board.fen(),
            "text": self._Translator(board.turn).decode_absolute(raw),
            "raw_text": raw,
            "best_move_uci": legal_best.uci() if legal_best else None,
            "move_source": source,
            "critical_line": critical,
            "inference_settings": {"backend": "torch", "prompt": req.prompt or self.settings["prompt"], "device": self.device,
                                   "quantize": self.cfg.quantize, "request_seed": seed,
                                   "temperature": self.settings["temperature"] if temperature is None else temperature},
        }
        return Analysis.from_upstream(data)
