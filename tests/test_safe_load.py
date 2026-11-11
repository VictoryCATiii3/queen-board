"""The encoder loader must load real weights exactly and never read empty ones."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from safetensors.torch import save_file  # noqa: E402

from queen_portable.safe_load import load_pretrained_safely, safetensors_shapes  # noqa: E402


class TinyConfig(transformers.PretrainedConfig):
    model_type = "tiny_empty_test"


class Tiny(transformers.PreTrainedModel):
    config_class = TinyConfig
    base_model_prefix = "tiny"

    def __init__(self, config):
        super().__init__(config)
        self.lin = torch.nn.Linear(4, 3)
        self.register_buffer("unused", torch.zeros(0, 1024))   # like Leela's smolgen weights
        self.post_init()

    def _init_weights(self, module):
        pass


def make(tmp_path, drop=None):
    TinyConfig().save_pretrained(tmp_path)
    sd = {"lin.weight": torch.randn(3, 4), "lin.bias": torch.randn(3), "unused": torch.zeros(0, 1024)}
    if drop:
        sd.pop(drop)
    save_file(sd, str(tmp_path / "model.safetensors"))
    return sd


def test_loads_exactly_and_keeps_empty(tmp_path):
    sd = make(tmp_path)
    assert safetensors_shapes(tmp_path / "model.safetensors")["unused"] == [0, 1024]
    m = load_pretrained_safely(Tiny, tmp_path)
    assert torch.equal(m.lin.weight, sd["lin.weight"]) and m.unused.shape == (0, 1024)


def test_missing_real_weight_is_an_error(tmp_path):
    make(tmp_path, drop="lin.weight")
    with pytest.raises(ValueError, match="missing"):
        load_pretrained_safely(Tiny, tmp_path)
