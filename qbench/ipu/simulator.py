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
    def __init__(self, sim: ApproxSimulator) -> None:
        super().__init__()
        self.sim = sim

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name
        sim = self.sim
        if name == "aten::gelu" or name == "aten::gelu_":
            op = "gelu_tanh" if kwargs.get("approximate", args[1] if len(args) > 1 else "none") == "tanh" else "gelu"
            if op in sim.methods:
                return sim._inplace(name, args[0], sim._unary(op, args[0]))
        elif name in UNARY and UNARY[name] in sim.methods and _plain_unary(name, args, kwargs):
            return sim._inplace(name, args[0], sim._unary(UNARY[name], args[0]))
        elif name in ("aten::_softmax", "aten::softmax") and "softmax" in sim.methods:
            sim.hits["softmax"] += 1
            y = sim.method("softmax").fn(args[0], args[1])
            dtype = args[2] if name == "aten::softmax" and len(args) > 2 else kwargs.get("dtype")
            return y.to(dtype) if isinstance(dtype, torch.dtype) else y
        elif name in ("aten::native_layer_norm", "aten::layer_norm") and "layernorm" in sim.methods:
            sim.hits["layernorm"] += 1
            x, shape = args[0], args[1]
            w = args[2] if len(args) > 2 else kwargs.get("weight")
            b = args[3] if len(args) > 3 else kwargs.get("bias")
            eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
            out = sim.method("layernorm").fn(x, shape, w, b, eps)
            return out if name == "aten::native_layer_norm" else out[0]
        elif name in MATMULS and "matmul" in sim.methods:
            sim.hits["matmul"] += 1
            return sim._matmul(func, name, args, kwargs)
        return func(*args, **kwargs)


def _plain_unary(name: str, args, kwargs) -> bool:
    """True when the op has its default extra arguments (softplus beta, elu alpha...)."""
    if name.startswith("aten::softplus"):
        return (len(args) < 2 or args[1] == 1) and (len(args) < 3 or args[2] == 20)
    if name.startswith("aten::elu"):
        return all(a == 1 for a in args[1:4])
    return True


class ApproxSimulator:
    def __init__(self, model: nn.Module, methods: Mapping[str, str], *, copy_model: bool = True) -> None:
        for op, m in methods.items():
            approx.get(op, m)  # unknown names fail here, not mid-run
        self.methods = dict(methods)
        self.model = copy.deepcopy(model) if copy_model else model
        self.hits: Counter[str] = Counter()
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

    def _matmul(self, func, name: str, args, kwargs) -> torch.Tensor:
        p = self.method("matmul").params
        fmt, out = p["fmt"], p["out"]
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
                    return _m.fn(x)
                mod.forward = forward

    # ------------------------------------------------------------------ qbench simulator API
    def run(self, scenario: Scenario | Any) -> Any:
        args = tuple(a.clone() if torch.is_tensor(a) else a for a in scenario.args)
        kwargs = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scenario.kwargs.items()}
        with torch.inference_mode(), _Mode(self):
            return self.model(*args, **kwargs)

    __call__ = run
