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
