"""The QUEEN "bridge": 16 gated cross-attention sublayers that let SmolLM3 read the
Leela encoder's board representation.

Plain-PyTorch port of upstream models/vllm/flamingo.py::DenseXAttnInfer. Same
parameter names (so xattn.pt loads unchanged), same math, same placement before
decoder layers 0, 2, ..., 30. Only the plumbing differs: instead of vLLM's
flat token stream and slot routing, this works on ordinary (batch, seq, dim)
tensors, so it runs anywhere PyTorch does (CUDA, MPS, CPU).

Reusable on its own: it has no chess knowledge. Any "silent expert" that yields
(n_layers, n_tokens, dim) states can be bridged into a decoder the same way.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_XATTN = 16
N_ENC_SQUARES = 64


def xattn_positions(n_layers: int, n_xattn: int = N_XATTN) -> list[int]:
    assert 2 * n_xattn <= n_layers, f"cannot fit {n_xattn} x-attns in {n_layers} layers"
    return [2 * i for i in range(n_xattn)]


class DenseXAttn(nn.Module):
    def __init__(self, encoder_dim: int, decoder_dim: int, n_heads: int = 16, n_kv_heads: int = 16):
        super().__init__()
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.head_dim = decoder_dim // n_heads
        self.n_rep = n_heads // n_kv_heads
        self.W_Q = nn.Linear(decoder_dim, n_heads * self.head_dim, bias=False)
        self.W_K = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_V = nn.Linear(encoder_dim, n_kv_heads * self.head_dim, bias=False)
        self.W_O = nn.Linear(n_heads * self.head_dim, decoder_dim, bias=False)
        self.alpha_attn = nn.Parameter(torch.zeros(1))
        self.alpha_ffn = nn.Parameter(torch.zeros(1))
        self.norm_y = nn.LayerNorm(decoder_dim)
        self.norm_x = nn.LayerNorm(encoder_dim)
        self.norm_ffn = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Module()
        self.ffn.fc1 = nn.Linear(decoder_dim, 2 * decoder_dim, bias=False)
        self.ffn.fc2 = nn.Linear(2 * decoder_dim, decoder_dim, bias=False)

    def encode_kv(self, enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """enc (B, 64, encoder_dim) -> K, V each (B, n_kv_heads, 64, head_dim). Done once per position."""
        B, S, _ = enc.shape
        x = self.norm_x(enc)
        k = self.W_K(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_V(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        return k, v

    def forward(self, y: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """y (B, T, decoder_dim) -> same shape. Runs on prefill and every decode step."""
        B, T, _ = y.shape
        q = self.W_Q(self.norm_y(y)).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)            # (B, H, T, hd), no mask: all 64 squares visible
        attn = attn.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        y = torch.tanh(self.alpha_attn).to(y.dtype) * self.W_O(attn) + y
        h = self.norm_ffn(y)
        ffn = self.ffn.fc2(F.relu(self.ffn.fc1(h)).square())     # "relu2" activation
        return torch.tanh(self.alpha_ffn).to(y.dtype) * ffn + y


class XAttnBridge(nn.Module):
    """Holds the 16 sublayers and hooks them in front of a Hugging Face decoder's layers."""

    def __init__(self, encoder_dim: int, decoder_dim: int, n_xattn: int = N_XATTN):
        super().__init__()
        self.layers = nn.ModuleList(DenseXAttn(encoder_dim, decoder_dim) for _ in range(n_xattn))
        self._kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self._handles = []

    def load_upstream(self, state_dict: dict) -> None:
        """xattn.pt keys look like '0.W_Q.weight' (a bare ModuleList state dict)."""
        missing, unexpected = self.layers.load_state_dict(state_dict, strict=False)
        assert not unexpected, f"unexpected x-attn keys: {list(unexpected)[:6]}"
        assert not missing, f"missing x-attn keys: {list(missing)[:6]}"

    def attach(self, decoder_layers) -> None:
        for xi, pos in enumerate(xattn_positions(len(decoder_layers), len(self.layers))):
            self._handles.append(decoder_layers[pos].register_forward_pre_hook(
                self._make_hook(xi), with_kwargs=True))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @torch.no_grad()
    def set_board(self, enc_states: torch.Tensor) -> None:
        """enc_states (B, 16, 64, encoder_dim) from the Leela encoder."""
        self._kv = [layer.encode_kv(enc_states[:, xi]) for xi, layer in enumerate(self.layers)]

    def clear_board(self) -> None:
        self._kv = None

    def _make_hook(self, xi: int):
        def hook(module, args, kwargs):
            if self._kv is None:
                raise RuntimeError("QUEEN decoder ran without a board; call set_board() first")
            if args:
                h, rest = args[0], args[1:]
            else:
                h, rest = kwargs["hidden_states"], ()
            k, v = self._kv[xi]
            if k.shape[0] != h.shape[0]:            # e.g. one board, several samples
                k = k.expand(h.shape[0], -1, -1, -1)
                v = v.expand(h.shape[0], -1, -1, -1)
            y = self.layers[xi](h, k.to(h.dtype), v.to(h.dtype))
            if args:
                return (y, *rest), kwargs
            kwargs["hidden_states"] = y
            return args, kwargs
        return hook
