"""IPU numerics for qbench: NX8 fake quantization, the catalogue of IPU realisations of
each op (exact / approximation / estimation / variation), a simulator that swaps them
into a real model, and providers for Hugging Face models.

    from qbench.ipu import evaluate_methods
    report = evaluate_methods({"id": "openai-community/gpt2", "kind": "hf", "task": "causal-lm"},
                              {"gelu_tanh": "lut256", "softmax": "exp2_nx8_e4"})
"""

from __future__ import annotations

from typing import Any

from qbench.ipu import candidates
from qbench.ipu.approx import CATALOGUE, ISA_SUPPORT, Method, get, op_error, summary
from qbench.ipu.nx8 import Nx8, quantize
from qbench.ipu.simulator import ApproxSimulator

__all__ = ["CATALOGUE", "ISA_SUPPORT", "ApproxSimulator", "LayerErrors", "candidates", "Method", "Nx8", "evaluate_methods",
           "get", "op_error", "quantize", "summary"]


class LayerErrors:
    """Each leaf module's output in the simulated model against the same module's in the reference, on the
    same inputs: forward hooks on both, outputs paired in call order (the reference runs a batch, then the
    simulator), signal power, error power and the largest error accumulated per module path; nothing kept.

        with LayerErrors(reference, sim.model) as le:
            ... run both ...
        le.report()   # {path: {"sqnr_db", "max_abs", "elements", "calls", "signal", "error", "order"}}

    ``signal`` / ``error``: the summed powers (to combine modules); ``order``: the module's first call among
    all (a module's output is its last leaf's, in call order).
    """

    def __init__(self, reference: Any, simulated: Any) -> None:
        self.models = (reference, simulated)
        self.handles: list[Any] = []
        self.pending: dict[str, list[Any]] = {}
        self.stats: dict[str, list[float]] = {}
        self.order: dict[str, int] = {}

    @staticmethod
    def _first_tensor(out: Any) -> Any:
        import torch
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return next((o for o in out if isinstance(o, torch.Tensor)), None)
        if isinstance(out, dict):
            return next((o for o in out.values() if isinstance(o, torch.Tensor)), None)
        return getattr(out, "last_hidden_state", None) if hasattr(out, "last_hidden_state") else None

    def __enter__(self) -> "LayerErrors":
        for side, model in enumerate(self.models):
            for path, module in model.named_modules():
                if path and not any(module.children()):
                    self.handles.append(module.register_forward_hook(
                        lambda _m, _a, out, path=path, side=side: self._observe(path, side, out)))
        return self

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def _observe(self, path: str, side: int, out: Any) -> None:
        t = self._first_tensor(out)
        if t is None or not t.is_floating_point():
            return
        t = t.detach().float()
        if side == 0:
            self.order.setdefault(path, len(self.order))
            self.pending.setdefault(path, []).append(t)
            return
        queue = self.pending.get(path)
        if not queue:
            return
        ref = queue.pop(0)
        if ref.shape != t.shape:
            return
        err = t.to(ref.device) - ref
        st = self.stats.setdefault(path, [0.0, 0.0, 0.0, 0.0, 0.0])
        st[0] += float(ref.pow(2).sum())
        st[1] += float(err.pow(2).sum())
        st[2] = max(st[2], float(err.abs().max()) if err.numel() else 0.0)
        st[3] += ref.numel()
        st[4] += 1

    def report(self) -> dict[str, dict[str, float]]:
        import math
        out = {}
        for path, (sig, err, mx, n, calls) in self.stats.items():
            sqnr = math.inf if err == 0 else (10 * math.log10(sig / err) if sig > 0 else -math.inf)
            out[path] = {"sqnr_db": sqnr, "max_abs": mx, "elements": int(n), "calls": int(calls), "signal": sig,
                         "error": err, "order": self.order.get(path, -1)}
        return out


def evaluate_methods(spec: dict[str, Any], methods: dict[str, str], *, samples: int = 16,
                     seq_len: int = 256, provider: Any = None, precision: str | None = None,
                     store_matmul_out: bool = False, layers: bool = False) -> dict[str, Any]:
    """Accuracy of ``spec``'s model with ``methods`` swapped in, against the exact model.

    Returns qbench's evaluation report as a dict, plus ``hits`` (how many calls
    each swapped op received: 0 means the model never ran it) and ``dataset``; with ``layers``,
    also each leaf module's output error against the exact model's (``LayerErrors``).
    """
    from qbench.evaluation import EvaluationConfig, evaluate
    from qbench.ipu.providers_hf import hf_provider

    provider = provider or hf_provider(spec, samples=samples, seq_len=seq_len)
    reference = provider.build_model()
    provider.evaluation_loader()
    sim = ApproxSimulator(reference, methods, precision=precision, store_matmul_out=store_matmul_out)
    if layers:
        with LayerErrors(reference, sim.model) as le:
            report = evaluate(reference, sim, provider, EvaluationConfig(metrics="fast", task=provider.task))
    else:
        report = evaluate(reference, sim, provider, EvaluationConfig(metrics="fast", task=provider.task))
    out = report.to_dict()
    if layers:
        out["layers"] = le.report()
    out["hits"] = dict(sim.hits)
    out["dataset"] = provider.dataset
    out["task"] = provider.task
    out["precision"] = precision or "fp32"
    return out
