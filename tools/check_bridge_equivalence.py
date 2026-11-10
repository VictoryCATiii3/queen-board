"""Compare our DenseXAttn/XAttnBridge to upstream DenseXAttnInfer on random weights.
Usage (from a checkpoint folder): python tools/check_bridge_equivalence.py models/vllm/flamingo.py /path/to/queen-portable"""
import ast, sys, types
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
src = open(sys.argv[1]).read()
tree = ast.parse(src)
# pull just the upstream class + constants, skip the vllm imports
keep = [n for n in tree.body if (isinstance(n, ast.ClassDef) and n.name == "DenseXAttnInfer")
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("N_XATTN", "N_ENC_SQUARES") for t in n.targets))]
ns = {"torch": torch, "nn": nn, "F": F, "np": np}
exec(compile(ast.Module(keep, []), "upstream", "exec"), ns)
Up = ns["DenseXAttnInfer"]
sys.path.insert(0, sys.argv[2])
from queen_portable.xattn import DenseXAttn, XAttnBridge

torch.manual_seed(0)
up = Up(1024, 2048).double(); 
for p in up.parameters(): nn.init.normal_(p, std=0.02)
ours = DenseXAttn(1024, 2048).double(); ours.load_state_dict(up.state_dict())

enc = torch.randn(1, 64, 1024, dtype=torch.float64)
T = 7
y = torch.randn(1, T, 2048, dtype=torch.float64)
# upstream prefill path: per-request K/V (R,64,Dkv) + segments
k_u, v_u = up.encode_kv(enc)
ref = up(y[0], k_u, v_u, [(0, T)])
k, v = ours.encode_kv(enc)
got = ours(y, k, v)[0]
print("prefill max abs diff:", (ref - got).abs().max().item())
# upstream decode path: per-token gather (T,64,Dkv), segments None
ref_d = up(y[0, :1], k_u, v_u, None)
got_d = ours(y[:, :1], k, v)[0]
print("decode  max abs diff:", (ref_d - got_d).abs().max().item())

# hook placement: fake 36-layer decoder of identity layers
class Ident(nn.Module):
    def forward(self, hidden_states, **kw): return hidden_states
layers = nn.ModuleList(Ident() for _ in range(36))
br = XAttnBridge(1024, 2048).double()
sd = {f"{i}.{k}": v for i in range(16) for k, v in up.state_dict().items()}
br.load_upstream(sd); br.attach(layers)
br.set_board(enc.unsqueeze(1).expand(1, 16, 64, 1024))
h = y.clone()
for L in layers: h = L(h) if L is not layers[0] else L(hidden_states=h)
exp = y[0].clone()
for _ in range(16): exp = up(exp, k_u, v_u, [(0, T)])
print("16-bridge stack diff:", (h[0] - exp).abs().max().item())
