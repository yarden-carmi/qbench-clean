"""IPU numerics for qbench: NX8 fake quantization, the catalogue of IPU realisations of
each op (exact / approximation / estimation / variation), a simulator that swaps them
into a real model, and providers for Hugging Face models.

    from qbench.ipu import evaluate_methods
    report = evaluate_methods({"id": "openai-community/gpt2", "kind": "hf", "task": "causal-lm"},
                              {"gelu_tanh": "lut256", "softmax": "exp2_nx8_e4"})
"""

from __future__ import annotations

from typing import Any

from qbench.ipu.approx import CATALOGUE, Method, get, op_error, summary
from qbench.ipu.nx8 import Nx8, quantize
from qbench.ipu.simulator import ApproxSimulator

__all__ = ["CATALOGUE", "ApproxSimulator", "Method", "Nx8", "evaluate_methods", "get", "op_error",
           "quantize", "summary"]


def evaluate_methods(spec: dict[str, Any], methods: dict[str, str], *, samples: int = 16,
                     seq_len: int = 256, provider: Any = None) -> dict[str, Any]:
    """Accuracy of ``spec``'s model with ``methods`` swapped in, against the exact model.

    Returns qbench's evaluation report as a dict, plus ``hits`` (how many calls
    each swapped op received: 0 means the model never ran it) and ``dataset``.
    """
    from qbench.evaluation import EvaluationConfig, evaluate
    from qbench.ipu.providers_hf import hf_provider

    provider = provider or hf_provider(spec, samples=samples, seq_len=seq_len)
    reference = provider.build_model()
    provider.evaluation_loader()
    sim = ApproxSimulator(reference, methods)
    report = evaluate(reference, sim, provider, EvaluationConfig(metrics="fast", task=provider.task))
    out = report.to_dict()
    out["hits"] = dict(sim.hits)
    out["dataset"] = provider.dataset
    out["task"] = provider.task
    return out
