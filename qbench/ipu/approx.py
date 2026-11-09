"""The catalogue of IPU realisations of each op: exact, approximation, estimation, variation.

The IPU (ipu-emulator) has, as hardware nonlinear functions, only ``identity``,
``relu``, ``relu6``, ``reciprocal``, ``rsqrt``, ``exp2`` and a 256-entry ``generic``
LUT (``ACTIVATE.QUANTIZE``), and it stores every vector as NX8. Anything else
(GELU, SiLU, sigmoid, tanh, exp) is computed from those, approximately. Each
:class:`Method` here is one way to do it, as a PyTorch function with the op's
own signature, so it can stand in for the op inside a real model
(:mod:`qbench.ipu.simulator`) and serve as a kernel's golden reference.

Kinds (shared with ipu-kernelgen's database):

``exact``          the same function, up to FP32 rounding order
``approximation``  the same function computed approximately (LUT, NX8 rounding)
``estimation``     a different, cheaper function standing in for it
``variation``      the op restructured, with a possible numerical effect

Assumption: the LUT's entry format and interpolation are unpublished (OQ-38).
``lut256`` models 256 FP32 entries over [-R, R] with linear interpolation and the
function's asymptotes outside; ``lut256_nearest`` drops the interpolation (the
pessimistic case).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from qbench.ipu.nx8 import quantize

LOG2E = 1.0 / math.log(2.0)


@dataclass(frozen=True)
class Method:
    op: str
    name: str
    kind: str
    fn: Callable[..., torch.Tensor]
    description: str
    hw: str = ""                       # IPU instructions that realise it
    on_chip: bool = True               # realisable with HARDWARE instructions only
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.op}:{self.name}"


CATALOGUE: dict[str, dict[str, Method]] = {}


def register(m: Method) -> Method:
    CATALOGUE.setdefault(m.op, {})[m.name] = m
    return m


def get(op: str, name: str) -> Method:
    try:
        return CATALOGUE[op][name]
    except KeyError:
        known = ", ".join(f"{o}:{n}" for o in CATALOGUE for n in CATALOGUE[o])
        raise KeyError(f"no method {op}:{name}; known: {known}") from None


# --------------------------------------------------------------------------- LUT
def lut(fn: Callable[[torch.Tensor], torch.Tensor], lo: Callable[[torch.Tensor], torch.Tensor],
        hi: Callable[[torch.Tensor], torch.Tensor], r: float = 8.0, entries: int = 256,
        interpolate: bool = True) -> Callable[[torch.Tensor], torch.Tensor]:
    """A table of ``fn`` over [-r, r]; ``lo``/``hi`` give the value outside it."""
    grid = torch.linspace(-r, r, entries, dtype=torch.float64)
    table = fn(grid).to(torch.float32)
    step = 2 * r / (entries - 1)

    def apply(x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        pos = ((xf.clamp(-r, r) + r) / step)
        if interpolate:
            i0 = pos.floor().clamp(0, entries - 2).long()
            t = pos - i0
            y = table[i0] * (1 - t) + table[i0 + 1] * t
        else:
            y = table[pos.round().long().clamp(0, entries - 1)]
        y = torch.where(xf < -r, lo(xf), torch.where(xf > r, hi(xf), y))
        return y.to(x.dtype)

    return apply


def _gelu(x):
    return F.gelu(x)


def _gelu_t(x):
    return F.gelu(x, approximate="tanh")


zeros = torch.zeros_like
ident = lambda x: x  # noqa: E731
ones = torch.ones_like
neg_ones = lambda x: -torch.ones_like(x)  # noqa: E731


def _q(x: torch.Tensor, fmt: str = "e4") -> torch.Tensor:
    return quantize(x, fmt)


def _exp2_recip_sigmoid(x: torch.Tensor, fmt: str = "e4") -> torch.Tensor:
    """sigmoid via exp2 and reciprocal, each stored stage NX8, in the stable form.

    e = exp2(-|x| log2e) is in (0, 1], so one row's values share a sane scale; the
    naive exp2(-x log2e) mixes huge and tiny values in a row and the row scale flushes
    the tiny ones (sigmoid(0.02) -> 1). r = 1/(1+e) = sigmoid(|x|); sigmoid(x) = r for
    x >= 0, e r for x < 0. This is the dataflow of kernels/zoo/silu_exp2_recip.
    """
    t = _q(x.abs(), fmt)
    e = _q(torch.exp2(-t * LOG2E), fmt)
    r = _q(torch.reciprocal(1.0 + e), fmt)
    return torch.where(x >= 0, r, _q(e * r, fmt))


# --------------------------------------------------------------------------- activations
for op, f, lo, hi in [
    ("gelu", _gelu, zeros, ident),
    ("gelu_tanh", _gelu_t, zeros, ident),
    ("quick_gelu", lambda x: x * torch.sigmoid(1.702 * x), zeros, ident),
    ("silu", F.silu, zeros, ident),
    ("mish", F.mish, zeros, ident),
    ("sigmoid", torch.sigmoid, zeros, ones),
    ("tanh", torch.tanh, neg_ones, ones),
    ("softplus", F.softplus, zeros, ident),
    ("elu", F.elu, neg_ones, ident),
]:
    register(Method(op, "exact", "exact", f, f"PyTorch {op}, FP32",
                    hw="ACTIVATE.EXACT (emulator only)", on_chip=False))
    register(Method(op, "lut256", "approximation", lut(f, lo, hi),
                    f"256-entry LUT over [-8, 8], linear interpolation, asymptotes outside",
                    hw="LD.LUT + ACTIVATE.QUANTIZE generic"))
    register(Method(op, "lut256_nearest", "approximation", lut(f, lo, hi, interpolate=False),
                    "256-entry LUT over [-8, 8], nearest entry (no interpolation)",
                    hw="LD.LUT + ACTIVATE.QUANTIZE generic"))

register(Method("gelu", "tanh", "estimation", _gelu_t,
                "0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))) in place of the erf form",
                hw="LUT (tanh) or ACTIVATE.QUANTIZE generic"))
register(Method("gelu", "sigmoid1702", "estimation", lambda x: x * torch.sigmoid(1.702 * x),
                "x*sigmoid(1.702x) (QuickGELU) in place of GELU", hw="exp2 + reciprocal + MULT"))
register(Method("gelu_tanh", "erf", "variation", _gelu, "the erf GELU in place of its tanh form",
                hw="LUT"))
register(Method("gelu_tanh", "sigmoid1702", "estimation", lambda x: x * torch.sigmoid(1.702 * x),
                "x*sigmoid(1.702x) in place of tanh GELU", hw="exp2 + reciprocal + MULT"))
register(Method("silu", "exp2_recip", "approximation", lambda x: x * _exp2_recip_sigmoid(x),
                "x * reciprocal(1 + exp2(-x log2e)), NX8 e4 between stages",
                hw="MULT + ACTIVATE.QUANTIZE exp2 + ACTIVATE.QUANTIZE reciprocal"))
register(Method("silu", "hardswish", "estimation", F.hardswish, "x*relu6(x+3)/6 in place of SiLU",
                hw="ACTIVATE.QUANTIZE relu6 + MULT"))
register(Method("sigmoid", "exp2_recip", "approximation", _exp2_recip_sigmoid,
                "reciprocal(1 + exp2(-x log2e)), NX8 e4 between stages",
                hw="ACTIVATE.QUANTIZE exp2 + reciprocal"))
register(Method("sigmoid", "hardsigmoid", "estimation", F.hardsigmoid, "relu6(x+3)/6 in place of sigmoid",
                hw="ACTIVATE.QUANTIZE relu6"))
register(Method("tanh", "via_sigmoid", "approximation", lambda x: 2 * _exp2_recip_sigmoid(2 * x) - 1,
                "2 sigmoid(2x) - 1 (an identity) via exp2 + reciprocal, NX8 e4 between stages",
                hw="ACTIVATE.QUANTIZE exp2 + reciprocal"))


def _gelu_tanh_exp2(x: torch.Tensor, fmt: str = "e4") -> torch.Tensor:
    # 0.5x(1 + tanh(z)) = x * sigmoid(2z), z = sqrt(2/pi)(x + 0.044715 x^3)
    z = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
    return x * _exp2_recip_sigmoid(_q(2.0 * z, fmt), fmt)


register(Method("gelu_tanh", "exp2_recip", "approximation", _gelu_tanh_exp2,
                "x * sigmoid(2 sqrt(2/pi)(x + 0.044715x^3)) via exp2 + reciprocal, NX8 e4 between stages",
                hw="MULT (cubic) + ACTIVATE.QUANTIZE exp2 + reciprocal"))
register(Method("gelu", "tanh_exp2", "estimation", _gelu_tanh_exp2,
                "the tanh form of GELU, computed via exp2 + reciprocal with NX8 e4 stages",
                hw="MULT (cubic) + ACTIVATE.QUANTIZE exp2 + reciprocal"))
register(Method("gelu", "sigmoid1702_exp2", "estimation", lambda x: x * _exp2_recip_sigmoid(1.702 * x),
                "x*sigmoid(1.702x) via exp2 + reciprocal, NX8 e4 stages", hw="MULT + exp2 + reciprocal"))
register(Method("quick_gelu", "exp2_recip", "approximation", lambda x: x * _exp2_recip_sigmoid(1.702 * x),
                "x*sigmoid(1.702x) via exp2 + reciprocal, NX8 e4 stages", hw="MULT + exp2 + reciprocal"))
for op, f in [("relu", F.relu), ("relu6", F.relu6), ("hardswish", F.hardswish),
              ("hardsigmoid", F.hardsigmoid)]:
    register(Method(op, "exact", "exact", f, f"{op}: relu/relu6 are hardware functions",
                    hw="ACTIVATE.QUANTIZE relu/relu6 (+ MULT)"))


# --------------------------------------------------------------------------- softmax
def _softmax_exact(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(x.float(), dim).to(x.dtype)


def softmax_exp2(x: torch.Tensor, dim: int, fmt: str | None = "e4") -> torch.Tensor:
    """max, (x - max) log2e, exp2, sum, reciprocal, multiply: the softmax kernels' recipe.

    With ``fmt`` every stored stage is NX8 along ``dim`` (the hardware store);
    ``None`` is the FP32 emulator path.

    Masked positions (an attention mask's ``-inf`` or ``finfo.min``, anything at or below
    -1e30) are inactive lanes, as a kernel masks them with ``R_MASK`` / the DS valid
    count: they hold 0 in every stored row, so they never set a row's scale, and their
    probability is 0. Storing them as values would let -3e38 set the scale and flush
    every real score in the row.
    """
    q = (lambda t: quantize(t, fmt, dim)) if fmt else (lambda t: t)
    xf = x.float()
    masked = xf <= -1e30
    zero = torch.zeros_like(xf)
    live = torch.where(masked, torch.full_like(xf, -torch.inf), xf)
    m = live.amax(dim, keepdim=True)
    m = torch.where(torch.isfinite(m), m, zero.sum(dim, keepdim=True))   # an all-masked row
    z = q(torch.where(masked, zero, (xf - m) * LOG2E))
    e = q(torch.where(masked, zero, torch.exp2(z)))
    s = e.sum(dim, keepdim=True)
    r = torch.reciprocal(s)        # one value per row: a MULT constant, FP32 in R_ACC
    return q(e * r).to(x.dtype)


register(Method("softmax", "exact", "exact", _softmax_exact, "torch.softmax, FP32",
                hw="ACTIVATE.EXACT exp2/reciprocal (emulator only)", on_chip=False))
register(Method("softmax", "exp2_fp32", "exact", lambda x, dim: softmax_exp2(x, dim, None),
                "exp2(x log2e) recipe in FP32 (the emulator kernels)", on_chip=False,
                hw="AGG.MAX, MULT, ACTIVATE.EXACT exp2, AGG.SUM, ACTIVATE.EXACT reciprocal"))
for fmt in ("e4", "e5", "dynamic"):
    register(Method("softmax", f"exp2_nx8_{fmt}", "approximation",
                    lambda x, dim, fmt=fmt: softmax_exp2(x, dim, fmt),
                    f"exp2(x log2e) recipe, every stored stage NX8 {fmt}",
                    hw="AGG.MAX, MULT, ACTIVATE.QUANTIZE exp2, AGG.SUM, ACTIVATE.QUANTIZE reciprocal"))


# --------------------------------------------------------------------------- norms
def layernorm(x, shape, weight, bias, eps, *, use_eps=True, fmt: str | None = None):
    dims = tuple(range(-len(shape), 0))
    xf = x.float()
    mean = xf.mean(dims, keepdim=True)
    var = ((xf - mean) ** 2).mean(dims, keepdim=True)
    rstd = torch.rsqrt(var + (eps if use_eps else 0.0))
    if not use_eps:
        rstd = torch.where(var == 0, torch.zeros_like(rstd), rstd)   # the kernels: rsqrt(0) = 0
    if fmt:
        rstd = quantize(rstd, fmt)
    y = (xf - mean) * rstd
    if fmt:
        y = quantize(y, fmt)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y.to(x.dtype), mean, rstd


register(Method("layernorm", "exact", "exact", lambda *a: layernorm(*a), "aten layer_norm, FP32",
                on_chip=False, hw="MULT/ACC mean+var, ACTIVATE.EXACT rsqrt"))
register(Method("layernorm", "no_eps", "variation", lambda *a: layernorm(*a, use_eps=False),
                "no eps (the emulator kernels, Q57); rsqrt(0) = 0", on_chip=False,
                hw="MULT/ACC mean+var, ACTIVATE.EXACT rsqrt"))
register(Method("layernorm", "rsqrt_nx8", "approximation", lambda *a: layernorm(*a, fmt="e4"),
                "rsqrt and output stored NX8 e4", hw="ACTIVATE.QUANTIZE rsqrt"))
register(Method("rsqrt", "exact", "exact", torch.rsqrt, "torch.rsqrt", hw="ACTIVATE.QUANTIZE rsqrt"))
register(Method("rsqrt", "nx8_e4", "approximation", lambda x: quantize(torch.rsqrt(x), "e4"),
                "rsqrt stored NX8 e4", hw="ACTIVATE.QUANTIZE rsqrt"))
register(Method("exp", "exp2", "exact", lambda x: torch.exp2(x * LOG2E), "exp2(x log2e)",
                hw="MULT + ACTIVATE.QUANTIZE exp2"))
register(Method("exp", "exp2_nx8", "approximation", lambda x: quantize(torch.exp2(quantize(x * LOG2E)), "e4"),
                "exp2(x log2e), stages NX8 e4", hw="MULT + ACTIVATE.QUANTIZE exp2"))


# --------------------------------------------------------------------------- numerics of the MACs
#: ``matmul`` methods quantize both operands of every mm/addmm/bmm/convolution along
#: the reduction axis (NX8 rows of 128 along k), accumulate in FP32 and, with
#: ``_out``, store the result NX8 too.
for fmt in ("e4", "e5", "dynamic"):
    register(Method("matmul", f"nx8_{fmt}", "approximation", ident,
                    f"operands NX8 {fmt} along k, FP32 accumulation",
                    hw="LD.MULT (NX8) + MULT + ACC", params={"fmt": fmt, "out": False}))
    register(Method("matmul", f"nx8_{fmt}_out", "approximation", ident,
                    f"operands and result NX8 {fmt}",
                    hw="LD.MULT + MULT + ACC + ACTIVATE.QUANTIZE", params={"fmt": fmt, "out": True}))


def summary() -> list[dict]:
    return [dict(op=m.op, method=m.name, kind=m.kind, description=m.description, hw=m.hw,
                 on_chip=m.on_chip) for ops in CATALOGUE.values() for m in ops.values()]


def op_error(m: Method, n: int = 1 << 16, seed: int = 0) -> dict[str, float]:
    """Error of ``m`` against the op's exact method on standard probe inputs."""
    g = torch.Generator().manual_seed(seed)
    if m.op == "matmul":
        a = torch.randn(64, 512, generator=g)
        b = torch.randn(512, 64, generator=g)
        fmt = m.params["fmt"]
        ref = a @ b
        got = quantize(a, fmt) @ quantize(b, fmt, dim=0)
        if m.params["out"]:
            got = quantize(got, fmt)
    elif m.op == "softmax":
        x = torch.randn(64, 1024, generator=g) * 4
        ref, got = _softmax_exact(x, -1), m.fn(x, -1)
    elif m.op == "layernorm":
        x = torch.randn(64, 768, generator=g) * 3 + 1
        args = (x, (768,), None, None, 1e-5)
        ref, got = layernorm(*args)[0], m.fn(*args)[0]
    elif m.op == "rsqrt":
        x = torch.rand(n, generator=g) * 10 + 1e-3
        ref, got = torch.rsqrt(x), m.fn(x)
    elif m.op == "exp":
        x = torch.randn(n, generator=g) * 4
        ref, got = torch.exp(x), m.fn(x)
    else:
        x = torch.randn(n, generator=g) * 3
        ref, got = CATALOGUE[m.op]["exact"].fn(x) if "exact" in CATALOGUE[m.op] else x, m.fn(x)
    err = (got.double() - ref.double())
    p = ref.double().pow(2).mean()
    nrm = err.pow(2).mean()
    return {
        "max_abs": float(err.abs().max()),
        "mean_abs": float(err.abs().mean()),
        "max_rel": float((err.abs() / ref.double().abs().clamp(min=1e-6)).max()),
        "sqnr_db": float(10 * torch.log10(p / nrm)) if nrm > 0 else float("inf"),
    }


#: What the ISA does for each op: "native" = one instruction (or MULT/ACC dataflow) computes
#: it as defined; "composed" = exact identities of native functions (exp via exp2, softmax,
#: norms) whose only error is NX8 storage; "none" = no instruction computes it, so every
#: kernel is an approximation or estimation (or uses the `generic` LUT).
ISA_SUPPORT: dict[str, tuple[str, str]] = {
    "relu": ("native", "ACTIVATE.QUANTIZE relu"),
    "relu6": ("native", "ACTIVATE.QUANTIZE relu6"),
    "reciprocal": ("native", "ACTIVATE.QUANTIZE reciprocal"),
    "rsqrt": ("native", "ACTIVATE.QUANTIZE rsqrt"),
    "matmul": ("native", "MULT + ACC"), "linear": ("native", "MULT + ACC"),
    "bmm": ("native", "MULT + ACC"), "conv2d": ("native", "MULT + ACC"),
    "add": ("native", "ACC.ADD"), "mul": ("native", "MULT"), "maxpool2d": ("native", "ACC.MAX / AGG.MAX"),
    "exp": ("composed", "exp2(x * log2e): MULT + ACTIVATE.QUANTIZE exp2"),
    "softmax": ("composed", "AGG.MAX, MULT, exp2, AGG.SUM, reciprocal"),
    "layernorm": ("composed", "MULT/ACC mean and variance, rsqrt (no eps: Q57)"),
    "rmsnorm": ("composed", "MULT/ACC mean of squares, rsqrt"),
    "hardswish": ("composed", "relu6 + MULT"), "hardsigmoid": ("composed", "relu6 + MULT"),
    "sigmoid": ("composed", "reciprocal(1 + exp2(-x log2e)) (needs |x| for exp2's domain)"),
    "silu": ("composed", "x * sigmoid(x) via exp2 + reciprocal"),
    "tanh": ("composed", "2 sigmoid(2x) - 1 via exp2 + reciprocal"),
    "quick_gelu": ("composed", "x * sigmoid(1.702x) via exp2 + reciprocal"),
    "gelu_tanh": ("composed", "x * sigmoid(2 sqrt(2/pi)(x + 0.044715x^3)) via exp2 + reciprocal"),
    "gelu": ("none", "no erf: tanh/sigmoid estimation, or the generic LUT"),
    "mish": ("none", "no log/softplus: generic LUT"), "softplus": ("none", "no log: generic LUT"),
    "elu": ("composed", "exp2 for x < 0, select by mask"),
}
