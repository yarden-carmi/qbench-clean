"""qbench.ipu: NX8 bit-exactness against the ipu-emulator codec, the catalogue, the simulator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from qbench.ipu import ApproxSimulator, approx, quantize
from qbench.ipu.nx8 import Nx8
from qbench.schemas import Scenario

EMULATOR = Path(os.environ.get("IPU_EMULATOR", Path.home() / "ipu-emulator"))


def _ipu_isa():
    src = EMULATOR / "src" / "ipu-isa" / "src"
    if not src.is_dir():
        pytest.skip("ipu-emulator checkout not found (set IPU_EMULATOR)")
    sys.path.insert(0, str(src))
    from ipu_isa import nx8  # type: ignore[import-not-found]
    return nx8


def _rows(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(6, 128, generator=g) * torch.logspace(-3, 3, 6)[:, None]
    base[0, :5] = torch.tensor([0.0, -0.0, 1.0, -1.0, 0.5])
    base[1, 7] = 0.0
    base[2] = base[2].abs()            # all-positive row
    base[3, ::7] = base[3, 0] * 2.0     # repeated max magnitude
    return base


@pytest.mark.parametrize("fe", [1, 2, 3, 4, 5, 6, 7])
def test_static_formats_bit_exact(fe):
    nx8 = _ipu_isa()
    fmt = nx8.Nx8Format(sign=1, exp_format=fe, num_bits=8)
    for seed in range(4):
        x = _rows(seed)
        got = quantize(x, Nx8(fe=fe))
        for r in range(x.shape[0]):
            enc = nx8.encode_row([float(v) for v in x[r]], fmt)
            want = nx8.decode_row(enc.elements, fmt, enc.scale, enc.dynamic_exponent)
            assert got[r].tolist() == pytest.approx(want, rel=0, abs=0), (fe, seed, r)


def test_unsigned_and_narrow_bit_exact():
    nx8 = _ipu_isa()
    for sign, fe, bits in [(0, 4, 8), (1, 3, 6), (0, 2, 4)]:
        fmt = nx8.Nx8Format(sign=sign, exp_format=fe, num_bits=bits)
        x = _rows(7)
        got = quantize(x, Nx8(fe=fe, sign=sign, bits=bits))
        for r in range(x.shape[0]):
            enc = nx8.encode_row([float(v) for v in x[r]], fmt)
            want = nx8.decode_row(enc.elements, fmt, enc.scale, enc.dynamic_exponent)
            assert got[r].tolist() == want, (sign, fe, bits, r)


def test_dynamic_matches_mostly():
    nx8 = _ipu_isa()
    fmt = nx8.Nx8Format(sign=1, exp_format=1, num_bits=8, dynamic=True)
    x = _rows(3)
    got = quantize(x, "dynamic")
    same = 0
    for r in range(x.shape[0]):
        enc = nx8.encode_row([float(v) for v in x[r]], fmt)
        same += got[r].tolist() == nx8.decode_row(enc.elements, fmt, enc.scale, enc.dynamic_exponent)
    assert same == x.shape[0]


def test_rows_along_dim_and_padding():
    x = torch.randn(3, 200, 5)
    q = quantize(x, "e4", dim=1)
    assert q.shape == x.shape
    # rows are taken along dim 1: quantizing the moved tensor gives the same values
    assert torch.equal(q, quantize(x.movedim(1, -1), "e4").movedim(-1, 1))


@pytest.mark.parametrize("op", ["gelu", "silu", "sigmoid", "tanh"])
def test_lut_close_to_exact(op):
    x = torch.linspace(-12, 12, 10001)
    exact = approx.get(op, "exact").fn(x)
    lut = approx.get(op, "lut256").fn(x)
    assert (lut - exact).abs().max() < 5e-3
    assert approx.op_error(approx.get(op, "lut256"))["sqnr_db"] > 40


def test_softmax_methods():
    x = torch.randn(4, 300) * 3
    assert torch.allclose(approx.get("softmax", "exp2_fp32").fn(x, -1), torch.softmax(x, -1), atol=1e-6)
    y = approx.get("softmax", "exp2_nx8_e4").fn(x, -1)
    assert torch.allclose(y.sum(-1), torch.ones(4), atol=0.1)


def test_catalogue_kinds():
    kinds = {m.kind for ops in approx.CATALOGUE.values() for m in ops.values()}
    assert kinds == {"exact", "approximation", "estimation", "variation"}


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
        return torch.softmax(self.fc2(self.ln(self.act(self.fc1(x)))), -1)


def test_simulator_swaps_and_counts():
    torch.manual_seed(0)
    model = _Tiny().eval()
    scen = Scenario("s", (torch.randn(4, 16),))
    ref = model(*scen.args)
    sim = ApproxSimulator(model, {"gelu": "lut256", "softmax": "exp2_nx8_e4", "layernorm": "no_eps",
                                   "matmul": "nx8_e4"})
    out = sim.run(scen)
    assert sim.hits == {"gelu": 1, "softmax": 1, "layernorm": 1, "matmul": 2}
    assert out.shape == ref.shape and not torch.equal(out, ref)
    assert torch.allclose(out, ref, atol=0.05)
    # the reference model is untouched
    assert torch.equal(model(*scen.args), ref)


def test_simulator_exact_methods_are_exact():
    torch.manual_seed(0)
    model = _Tiny().eval()
    scen = Scenario("s", (torch.randn(4, 16),))
    sim = ApproxSimulator(model, {"gelu": "exact", "softmax": "exact", "layernorm": "exact"})
    assert torch.allclose(sim.run(scen), model(*scen.args), atol=1e-6)


def test_evaluate_with_qbench():
    from qbench import EvaluationConfig, evaluate
    from qbench.providers import DirectObjectProvider
    torch.manual_seed(0)
    model = _Tiny().eval()
    data = [torch.randn(4, 16) for _ in range(3)]
    provider = DirectObjectProvider(model, [Scenario("s", (data[0],))], loader=data)
    sim = ApproxSimulator(model, {"gelu": "lut256", "matmul": "nx8_e4"})
    rep = evaluate(model, sim, provider, EvaluationConfig(metrics="fast")).to_dict()
    flat = str(rep)
    assert "sqnr" in flat and "cosine" in flat


def test_exp2_sigmoid_is_row_stable():
    # one row mixing large negative and small inputs: the naive exp2(-x log2e) form
    # flushes the small ones (sigmoid(0.02) -> 1); the stable form must not
    x = torch.zeros(1, 128)
    x[0, 0] = -40.0
    x[0, 1] = 0.02
    y = approx.get("sigmoid", "exp2_recip").fn(x)
    assert abs(float(y[0, 1]) - 0.505) < 0.02
    assert float(y[0, 0]) < 1e-6
    for op, m in [("silu", "exp2_recip"), ("gelu_tanh", "exp2_recip"), ("sigmoid", "exp2_recip")]:
        assert approx.op_error(approx.get(op, m))["sqnr_db"] > 20


def test_inf_matches_emulator_and_nan_raises():
    nx8 = _ipu_isa()
    for fe in (2, 4, 5):
        fmt = nx8.Nx8Format(sign=1, exp_format=fe, num_bits=8)
        x = _rows(5)
        x[0, 3], x[1, 9], x[2, 0] = float("inf"), float("-inf"), float("inf")
        got = quantize(x, Nx8(fe=fe))
        for r in range(x.shape[0]):
            enc = nx8.encode_row([float(v) for v in x[r]], fmt)
            assert got[r].tolist() == nx8.decode_row(enc.elements, fmt, enc.scale, enc.dynamic_exponent), (fe, r)
    with pytest.raises(ValueError):
        quantize(torch.tensor([float("nan")]), "e4")


def test_masked_softmax_rows():
    x = torch.randn(3, 256)
    x[:, 100:] = torch.finfo(torch.float32).min      # a causal-style mask
    x[1, 50:] = float("-inf")
    for fmt in ("e4", "dynamic"):
        y = approx.get("softmax", f"exp2_nx8_{fmt}").fn(x, -1)
        ref = torch.softmax(x, -1)
        assert y[:, 100:].abs().max() < 1e-30   # an all-zero NX8 row decodes as 2^-127, as in the emulator
        assert (y - ref).abs().max() < 0.05


def test_precision_regimes():
    torch.manual_seed(0)
    model = _Tiny().eval()
    scen = Scenario("s", (torch.randn(4, 16),))
    ref = model(*scen.args)
    # FP32 with no substitutes is the model
    assert torch.allclose(ApproxSimulator(model, {}).run(scen), ref, atol=1e-6)
    # NX8 baseline rounds even with no substitutes
    base = ApproxSimulator(model, {}, precision="e4")
    yq = base.run(scen)
    assert not torch.equal(yq, ref) and base.hits["matmul"] == 2
    # the substitute runs in the regime's precision
    s32 = ApproxSimulator(model, {"gelu": "tanh_exp2"}).run(scen)
    s8 = ApproxSimulator(model, {"gelu": "tanh_exp2"}, precision="e4").run(scen)
    assert not torch.equal(s32, s8)
    assert approx.PRECISION.get() is None   # reset after the run


def test_topk_substitute_routes():
    class M(nn.Module):
        def forward(self, x):
            v, i = torch.topk(torch.softmax(x, -1), 2, dim=-1)
            return v * i
    x = torch.randn(8, 16)
    sim = ApproxSimulator(M(), {"topk": "iterative_max"})
    assert torch.equal(sim.run(Scenario("s", (x,))), M()(x)) and sim.hits["topk"] == 1


def test_fold_batchnorm_is_exact_in_fp32_and_keeps_the_activation():
    import copy

    import torch
    from qbench.ipu import approx
    from qbench.ipu.simulator import ApproxSimulator

    torch.manual_seed(0)

    class ConvNormAct(torch.nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = torch.nn.Conv2d(cin, cout, 3, padding=1, bias=False)
            self.bn = torch.nn.BatchNorm2d(cout)
            self.act = torch.nn.SiLU()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    m = torch.nn.Sequential(ConvNormAct(3, 8), ConvNormAct(8, 4)).eval()
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-1, 1)
            mod.running_var.uniform_(0.5, 2)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.5, 0.5)
    x = torch.randn(1, 3, 16, 16)
    ref = m(x)
    folded = copy.deepcopy(m)
    assert approx.fold_batchnorm(folded) == 2
    assert folded[0].conv.bias is not None
    torch.testing.assert_close(folded(x), ref, atol=1e-5, rtol=1e-5)
    sim = ApproxSimulator(m, {"batchnorm": "fold"})
    assert sim.hits["batchnorm"] == 2
    assert m[0].conv.bias is None                        # the simulator folds its own copy
