"""Substitute implementations written as code (by a person or an agent) and registered at run time.

A candidate is the source of one Python function with the op's signature (``SIGNATURES``),
written only with the primitives the IPU has (``PRIMITIVES``); ``store(t)`` marks what the
chip writes to XMEM, so it is rounded in an NX8 precision and left alone in FP32.

    from qbench.ipu import candidates
    m = candidates.register("gelu", "agent_sig2", "estimation", SOURCE, "x*sigmoid(1.6x)...")
    sim = ApproxSimulator(model, {"gelu": "agent_sig2"}, precision="e4")
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from qbench.ipu import approx
from qbench.ipu.approx import Method, _exact32, _q, lut, register as _register

#: The function a candidate defines, per op: ``def fn(<args>)`` returning what the op returns.
SIGNATURES: dict[str, str] = {
    **{op: "fn(x) -> y (elementwise)" for op in (
        "gelu", "gelu_tanh", "quick_gelu", "silu", "sigmoid", "tanh", "mish", "softplus", "elu",
        "exp", "rsqrt", "erf", "cos", "sin")},
    "softmax": "fn(x, dim) -> y (softmax along dim)",
    "layernorm": "fn(x, shape, weight, bias, eps) -> (y, mean, rstd) over the last len(shape) dims",
    "topk": "fn(x, k, dim, largest) -> (values, indices) like torch.topk",
    "div": "fn(a, b) -> a / b",
}

#: What a candidate may call besides plain tensor arithmetic (+, -, *, comparisons, where, max/min,
#: sum/amax along a dim, abs, clamp): the IPU's own functions and its store.
PRIMITIVES: dict[str, Any] = {
    "store": lambda t, dim=-1: _q(t) if dim == -1 else (approx._qf(t, approx.precision(), dim)),
    "exp2": lambda t: _exact32(torch.exp2, t),
    "reciprocal": lambda t: _exact32(torch.reciprocal, t),
    "rsqrt": lambda t: _exact32(torch.rsqrt, t),
    "relu": F.relu,
    "relu6": F.relu6,
    "lut": lut,                      # lut(fn, lo, hi, r=8.0, entries=256, interpolate=True) -> callable
    "LOG2E_HI": approx.LOG2E_HI, "LOG2E_LO": approx.LOG2E_LO, "LOG2E": approx.LOG2E,
    "math": math, "torch": torch,
}

PRIMITIVE_DOC = """\
- store(t, dim=-1): what the chip writes to memory (rounded to NX8 in a quantized run). Call it on every
  intermediate a kernel would store: the chip has ONE accumulator, so a value needed after another
  vector op must be stored and reloaded. Only FP32 accumulator sums/products chain without a store.
- exp2(t), reciprocal(t), rsqrt(t), relu(t), relu6(t): the AaQ hardware functions (exp2 needs t <= 0
  in practice: keep its input non-positive, e.g. work on -|x|).
- multiply / add / subtract / max / min of vectors and scalar constants: MULT and ACC.
- per-row sum / max along a dim (AGG.SUM / AGG.MAX).
- lut(f, lo, hi, r=8.0, entries=256): a 256-entry table of f over [-r, r] (the `generic` LUT), with
  lo(x)/hi(x) giving the value outside it.
- constants: LOG2E_HI + LOG2E_LO = log2(e) as two exactly representable constants.
- NOT available on chip: exp, log, erf, tanh, sigmoid, sin, cos, sqrt, division, sort/topk.
"""


def compile_candidate(source: str) -> Any:
    """The ``fn`` that ``source`` defines (exec in a namespace of the primitives)."""
    ns: dict[str, Any] = dict(PRIMITIVES)
    exec(compile(source, "<candidate>", "exec"), ns)  # noqa: S102 - candidates are run in a subprocess first
    fn = ns.get("fn")
    if not callable(fn):
        raise ValueError("the candidate defines no fn")
    return fn


def register(op: str, name: str, kind: str, source: str, description: str, hw: str = "") -> Method:
    if op not in SIGNATURES:
        raise ValueError(f"no candidate signature for {op}")
    if kind not in ("exact", "approximation", "estimation", "variation"):
        raise ValueError(f"kind {kind!r}")
    m = Method(op, name, kind, compile_candidate(source), description, hw=hw, params={"source": source})
    return _register(m)


def probe_inputs(op: str, seed: int = 0) -> tuple[tuple, dict]:
    g = torch.Generator().manual_seed(seed)
    if op == "softmax":
        return (torch.randn(64, 300, generator=g) * 4, -1), {}
    if op == "layernorm":
        x = torch.randn(64, 768, generator=g) * 3 + 1
        return (x, (768,), torch.rand(768, generator=g) + 0.5, torch.randn(768, generator=g) * 0.1, 1e-5), {}
    if op == "topk":
        return (torch.randn(256, 64, generator=g), 8, -1, True), {}
    if op == "div":
        return (torch.randn(4096, generator=g), torch.rand(4096, generator=g) * 4 + 0.05), {}
    if op == "rsqrt":
        return (torch.rand(1 << 15, generator=g) * 10 + 1e-3,), {}
    if op in ("cos", "sin"):
        return (torch.rand(1 << 15, generator=g) * 100,), {}
    return (torch.randn(1 << 15, generator=g) * 3,), {}


def exact_fn(op: str):
    if op == "topk":
        return lambda x, k, dim, largest: torch.topk(x, k, dim, largest)
    if op == "layernorm":
        return approx.layernorm
    if op == "div":
        return torch.div
    if op == "softmax":
        return lambda x, dim: torch.softmax(x, dim)
    return {"gelu": F.gelu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x), "silu": F.silu, "sigmoid": torch.sigmoid,
            "tanh": torch.tanh, "mish": F.mish, "softplus": F.softplus, "elu": F.elu, "exp": torch.exp,
            "rsqrt": torch.rsqrt, "erf": torch.erf, "cos": torch.cos, "sin": torch.sin}[op]


def op_error(op: str, fn, precision: str | None = None) -> dict[str, float]:
    """Error of ``fn`` against the exact op on the probe inputs, in ``precision``."""
    args, kw = probe_inputs(op)
    token = approx.PRECISION.set(precision)
    try:
        got = fn(*args, **kw)
    finally:
        approx.PRECISION.reset(token)
    ref = exact_fn(op)(*args, **kw)
    if op == "topk":
        gv, gi = got
        rv, ri = ref
        same = (torch.sort(gi, -1).values == torch.sort(ri, -1).values).all(-1).float().mean()
        err = (gv.double() - rv.double())
        return {"set_match": float(same), "max_abs": float(err.abs().max()), "sqnr_db": float("nan")}
    if op == "layernorm":
        got, ref = got[0], ref[0]
    err = got.double() - ref.double()
    if not torch.isfinite(err).all():
        return {"max_abs": float("inf"), "mean_abs": float("inf"), "sqnr_db": float("-inf")}
    p, n = ref.double().pow(2).mean(), err.pow(2).mean()
    return {"max_abs": float(err.abs().max()), "mean_abs": float(err.abs().mean()),
            "sqnr_db": float(10 * torch.log10(p / n)) if n > 0 else float("inf")}
