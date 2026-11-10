"""Run a model with chosen ops replaced by their IPU realisations (:mod:`qbench.ipu.approx`).

:class:`ApproxSimulator` is a drop-in simulator for :func:`qbench.evaluate`: it
holds a copy of the model and runs it under a ``TorchDispatchMode`` that sends
each selected ATen op to its catalogue method. Activations that a framework
writes out of elementary ops (``NewGELUActivation``, ``QuickGELUActivation``)
are replaced at the module, since no single ATen op carries them.

``methods`` maps an op of the catalogue to a method name, for example
``{"gelu": "lut256", "softmax": "exp2_nx8_e4", "matmul": "nx8_e4"}``; ops left
out run exactly.
"""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from qbench.ipu import approx
from qbench.ipu.nx8 import quantize
from qbench.schemas import Scenario

UNARY = {
    "aten::silu": "silu", "aten::sigmoid": "sigmoid", "aten::tanh": "tanh",
    "aten::hardswish": "hardswish", "aten::hardsigmoid": "hardsigmoid", "aten::relu": "relu",
    "aten::mish": "mish", "aten::softplus": "softplus", "aten::elu": "elu", "aten::rsqrt": "rsqrt",
    "aten::exp": "exp", "aten::silu_": "silu", "aten::sigmoid_": "sigmoid", "aten::tanh_": "tanh",
    "aten::hardswish_": "hardswish", "aten::relu_": "relu", "aten::elu_": "elu", "aten::exp_": "exp",
    "aten::rsqrt_": "rsqrt", "aten::hardsigmoid_": "hardsigmoid", "aten::mish_": "mish",
}
#: Activation modules written from elementary ops, by class name.
MODULE_ACTS = {
    "NewGELUActivation": "gelu_tanh", "GELUTanh": "gelu_tanh", "PytorchGELUTanh": "gelu_tanh",
    "FastGELUActivation": "gelu_tanh", "AccurateGELUActivation": "gelu_tanh",
    "QuickGELUActivation": "quick_gelu", "QuickGELU": "quick_gelu", "MishActivation": "mish",
}
#: Depending on the torch version a dispatch mode sees composite ops (``aten::linear``,
#: ``aten::layer_norm``, ``aten::softmax``) or their decompositions; both are handled.
MATMULS = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::convolution",
           "aten::_convolution", "aten::mkldnn_convolution", "aten::linear", "aten::matmul",
           "aten::conv1d", "aten::conv2d"}


class _Mode(TorchDispatchMode):
    """Routes each op: a chosen substitute (``methods``) or the exact op; in an NX8
    precision, what the chip stores is rounded either way (the quantized baseline is
    the same run with no substitutes)."""

    def __init__(self, sim: ApproxSimulator) -> None:
        super().__init__()
        self.sim = sim

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        sim = self.sim
        q = sim.precision is not None
        if name in ("aten::gelu", "aten::gelu_"):
            op = "gelu_tanh" if kwargs.get("approximate", args[1] if len(args) > 1 else "none") == "tanh" else "gelu"
            if op in sim.methods or q:
                y = sim._unary(op, args[0]) if op in sim.methods else func(*args, **kwargs)
                return sim._inplace(name, args[0], sim._store(y))
        elif name in UNARY and _plain_unary(name, args, kwargs) and (UNARY[name] in sim.methods or (q and UNARY[name] in ON_CHIP_UNARY)):
            op = UNARY[name]
            y = sim._unary(op, args[0]) if op in sim.methods else func(*args, **kwargs)
            return sim._inplace(name, args[0], sim._store(y))
        elif name in ("aten::_softmax", "aten::softmax") and ("softmax" in sim.methods or q):
            dim = args[1]
            if "softmax" in sim.methods:
                sim.hits["softmax"] += 1
                y = sim.method("softmax").fn(args[0], dim)
            else:
                y = torch.softmax(args[0].float(), dim)
            y = sim._store(y, dim)
            dtype = args[2] if name == "aten::softmax" and len(args) > 2 else kwargs.get("dtype")
            return y.to(dtype) if isinstance(dtype, torch.dtype) else y.to(args[0].dtype)
        elif name in ("aten::native_layer_norm", "aten::layer_norm") and ("layernorm" in sim.methods or q):
            x, shape = args[0], args[1]
            w = args[2] if len(args) > 2 else kwargs.get("weight")
            b = args[3] if len(args) > 3 else kwargs.get("bias")
            eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
            if "layernorm" in sim.methods:
                sim.hits["layernorm"] += 1
                y, mean, rstd = sim.method("layernorm").fn(x, shape, w, b, eps)
            else:
                y, mean, rstd = approx.layernorm(x, shape, w, b, eps)
            y = sim._store(y)
            return (y, mean, rstd) if name == "aten::native_layer_norm" else y
        elif name in MATMULS and ("matmul" in sim.methods or q):
            sim.hits["matmul"] += 1
            return sim._matmul(func, name, args, kwargs)
        elif name == "aten::topk" and "topk" in sim.methods:
            sim.hits["topk"] += 1
            k = args[1]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", -1)
            largest = args[3] if len(args) > 3 else kwargs.get("largest", True)
            return sim.method("topk").fn(args[0], k, dim, largest)
        elif name in ("aten::div", "aten::div_") and "div" in sim.methods and len(args) > 1 \
                and torch.is_tensor(args[1]) and args[1].numel() > 1 and not kwargs.get("rounding_mode"):
            sim.hits["div"] += 1
            return sim._inplace(name, args[0], sim.method("div").fn(args[0], args[1]))
        elif name in ("aten::cos", "aten::sin", "aten::erf") and name.split("::")[1] in sim.methods:
            op = name.split("::")[1]
            sim.hits[op] += 1
            return sim.method(op).fn(args[0])
        return func(*args, **kwargs)


#: Unary ops a chip kernel computes (stored NX8 in an NX8 precision even when exact).
ON_CHIP_UNARY = {"silu", "sigmoid", "tanh", "relu", "hardswish", "hardsigmoid", "exp", "rsqrt"}


def _plain_unary(name: str, args, kwargs) -> bool:
    """True when the op has its default extra arguments (softplus beta, elu alpha...)."""
    if name.startswith("aten::softplus"):
        return (len(args) < 2 or args[1] == 1) and (len(args) < 3 or args[2] == 20)
    if name.startswith("aten::elu"):
        return all(a == 1 for a in args[1:4])
    return True


class ApproxSimulator:
    def __init__(self, model: nn.Module, methods: Mapping[str, str], *, precision: str | None = None,
                 store_matmul_out: bool = False, copy_model: bool = True) -> None:
        """``precision``: None (FP32: only the substitutes differ from the model) or an NX8
        type: matmul operands, and the outputs of activations, softmax and LayerNorm, are
        rounded to it as the chip stores them; ``store_matmul_out`` rounds matmul results too."""
        for op, m in methods.items():
            approx.get(op, m)  # unknown names fail here, not mid-run
        self.methods = dict(methods)
        self.precision = None if precision in (None, "fp32") else precision
        self.store_matmul_out = store_matmul_out
        self.model = copy.deepcopy(model) if copy_model else model
        self.hits: Counter[str] = Counter()
        if self.methods.get("batchnorm") == "fold":
            self.hits["batchnorm"] += approx.fold_batchnorm(self.model)
        self._patch_modules()

    def method(self, op: str) -> approx.Method:
        return approx.get(op, self.methods[op])

    # ------------------------------------------------------------------ handlers
    def _unary(self, op: str, x: torch.Tensor) -> torch.Tensor:
        self.hits[op] += 1
        return self.method(op).fn(x)

    @staticmethod
    def _inplace(name: str, target: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if name.endswith("_"):
            return target.copy_(value)
        return value

    def _store(self, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """What an ACTIVATE.QUANTIZE store leaves: NX8 rows in an NX8 precision."""
        return quantize(y, self.precision, dim) if self.precision else y

    def _matmul(self, func, name: str, args, kwargs) -> torch.Tensor:
        if "matmul" in self.methods:
            p = self.method("matmul").params
            fmt, out = p["fmt"], p["out"]
        else:
            fmt, out = self.precision, self.store_matmul_out
        a = list(args)
        if name in ("aten::mm", "aten::bmm", "aten::matmul"):
            a[0] = quantize(a[0], fmt, -1)
            a[1] = quantize(a[1], fmt, -2 if a[1].dim() > 1 else -1)
            y, dim = func(*a, **kwargs), -1
        elif name == "aten::linear":       # linear(x, w[n, k], bias)
            a[0], a[1] = quantize(a[0], fmt, -1), quantize(a[1], fmt, -1)
            y, dim = func(*a, **kwargs), -1
        elif name in ("aten::addmm", "aten::baddbmm"):
            a[1], a[2] = quantize(a[1], fmt, -1), quantize(a[2], fmt, -2)
            y, dim = func(*a, **kwargs), -1
        else:  # convolution(x, w, bias, ...)
            x, w = a[0], a[1]
            a[0] = quantize(x, fmt, 1)
            a[1] = quantize(w.reshape(w.shape[0], -1), fmt, -1).reshape(w.shape)
            y, dim = func(*a, **kwargs), 1
        return quantize(y, fmt, dim) if out else y

    def _patch_modules(self) -> None:
        for _, mod in self.model.named_modules():
            op = MODULE_ACTS.get(type(mod).__name__)
            if op and op in self.methods:
                m = self.method(op)
                def forward(x, *_a, _m=m, _op=op, **_k):
                    self.hits[_op] += 1
                    return self._store(_m.fn(x))
                mod.forward = forward

    # ------------------------------------------------------------------ qbench simulator API
    def run(self, scenario: Scenario | Any) -> Any:
        args = tuple(a.clone() if torch.is_tensor(a) else a for a in scenario.args)
        kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scenario.kwargs.items()}
        token = approx.PRECISION.set(self.precision)
        try:
            with torch.inference_mode(), _Mode(self):
                return self.model(*args, **kwargs)
        finally:
            approx.PRECISION.reset(token)

    __call__ = run
