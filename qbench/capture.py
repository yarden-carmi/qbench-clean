"""Authoritative eager dispatcher capture for QBench."""

from __future__ import annotations

import contextvars

import copy
import inspect
import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from .schemas import OperationRecord, Scenario


#: Set by capture_scenario(track_storage=True): tensor metadata then names the tensor's storage (its
#: address, an int: no contents), so a reader can link an op's inputs to the op that wrote them.
_TRACK_STORAGE: contextvars.ContextVar[bool] = contextvars.ContextVar("qbench_track_storage", default=False)


def tensor_metadata(value: torch.Tensor) -> dict[str, Any]:
    meta = {
        "kind": "tensor",
        "shape": [int(dim) for dim in value.shape],
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": bool(value.requires_grad),
    }
    if _TRACK_STORAGE.get():
        try:
            meta["storage"] = int(value.untyped_storage().data_ptr())
        except Exception:  # noqa: BLE001 - a tensor without a storage (a meta or sparse tensor)
            pass
    return meta


def value_metadata(value: Any, *, depth: int = 0) -> Any:
    """Describe invocation values without retaining tensor contents."""
    if torch.is_tensor(value):
        return tensor_metadata(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (torch.dtype, torch.device, torch.layout)):
        return str(value)
    if depth >= 5:
        return {"kind": type(value).__name__}
    if isinstance(value, Mapping):
        items = list(value.items())[:64]
        # Dispatcher kwargs and ordinary structured model inputs use string
        # keys, which are safe to retain and are needed by argument matchers.
        # Never call ``str``/``repr`` on an arbitrary key: tensor string forms
        # contain values and user-defined ``__str__`` methods may expose other
        # private input data.  Non-string-keyed mappings therefore use an
        # explicit, privacy-safe key-metadata representation.
        if all(isinstance(key, str) for key, _item in items):
            return {key: value_metadata(item, depth=depth + 1) for key, item in items}
        return {
            "kind": "mapping",
            "items": [
                {
                    "key": value_metadata(key, depth=depth + 1),
                    "value": value_metadata(item, depth=depth + 1),
                }
                for key, item in items
            ],
        }
    if isinstance(value, (tuple, list)):
        return [value_metadata(item, depth=depth + 1) for item in value[:64]]
    return {"kind": f"{type(value).__module__}.{type(value).__qualname__}"}


def _clone_untyped_storage(value: torch.Tensor, memo: dict[Any, Any]):
    """Return one byte-for-byte storage clone shared by all aliased views."""
    storage = value.untyped_storage()
    storage_key = (
        "storage",
        value.device.type,
        value.device.index,
        int(storage._cdata),
    )
    cloned_storage = memo.get(storage_key)
    if cloned_storage is None:
        storage_bytes = int(storage.nbytes())
        # Clone the complete untyped storage as bytes so mixed-dtype views keep
        # the same raw representation and can share the cloned storage.
        flat = torch.empty(0, dtype=torch.uint8, device=value.device)
        flat.set_(storage, 0, (storage_bytes,), (1,))
        cloned_storage = flat.clone().untyped_storage()
        memo[storage_key] = cloned_storage
    return cloned_storage


def _clone_quantized_tensor(value: torch.Tensor, memo: dict[Any, Any]) -> torch.Tensor:
    """Clone a quantized tensor without discarding its quantizer metadata."""
    object_key = ("tensor", id(value))
    if object_key in memo:
        return memo[object_key]

    qscheme = value.qscheme()
    try:
        cloned_storage = _clone_untyped_storage(value, memo)
        if qscheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            clone = torch._empty_affine_quantized(
                (0,),
                scale=float(value.q_scale()),
                zero_point=int(value.q_zero_point()),
                dtype=value.dtype,
                device=value.device,
            )
        elif qscheme in {
            torch.per_channel_affine,
            torch.per_channel_symmetric,
            torch.per_channel_affine_float_qparams,
        }:
            clone = torch._empty_per_channel_affine_quantized(
                tuple(value.shape),
                scales=value.q_per_channel_scales().detach().clone(),
                zero_points=value.q_per_channel_zero_points().detach().clone(),
                axis=int(value.q_per_channel_axis()),
                dtype=value.dtype,
                device=value.device,
            )
        else:
            raise ValueError(f"unsupported quantization scheme {qscheme}")
        clone.set_(
            cloned_storage,
            int(value.storage_offset()),
            tuple(value.shape),
            tuple(value.stride()),
        )
        # The public empty factories produce affine quantizers.  Fall back to
        # PyTorch's own clone for a symmetric/future scheme rather than
        # silently changing the qscheme.
        if clone.qscheme() != qscheme:
            raise ValueError(f"could not reconstruct quantization scheme {qscheme}")
    except Exception:
        clone = value.detach().clone()
    memo[object_key] = clone
    return clone


def _clone_strided_tensor(value: torch.Tensor, memo: dict[Any, Any]) -> torch.Tensor:
    """Clone a strided tensor while preserving shared storage/view geometry."""
    object_key = ("tensor", id(value))
    if object_key in memo:
        return memo[object_key]

    if value.is_quantized:
        return _clone_quantized_tensor(value, memo)

    if value.layout != torch.strided or value.device.type == "meta":
        clone = value.detach().clone().requires_grad_(value.requires_grad)
        memo[object_key] = clone
        return clone

    cloned_storage = _clone_untyped_storage(value, memo)

    clone = torch.empty(0, dtype=value.dtype, device=value.device)
    clone.set_(
        cloned_storage,
        int(value.storage_offset()),
        tuple(value.shape),
        tuple(value.stride()),
    )
    if value.is_conj():
        clone = clone.conj()
    if getattr(value, "is_neg", lambda: False)():
        clone = clone._neg_view()
    clone.requires_grad_(value.requires_grad)
    memo[object_key] = clone
    return clone


def clone_inputs(value: Any, memo: dict[Any, Any] | None = None) -> Any:
    """Clone inputs while preserving object and tensor-storage aliases."""
    memo = {} if memo is None else memo
    if torch.is_tensor(value):
        return _clone_strided_tensor(value, memo)
    if isinstance(value, tuple):
        return (
            type(value)(*(clone_inputs(item, memo) for item in value))
            if hasattr(value, "_fields")
            else tuple(clone_inputs(item, memo) for item in value)
        )
    if isinstance(value, list):
        return [clone_inputs(item, memo) for item in value]
    if isinstance(value, Mapping):
        items = [
            (clone_inputs(key, memo), clone_inputs(item, memo))
            for key, item in value.items()
        ]
        try:
            return type(value)(items)
        except Exception:
            try:
                return copy.deepcopy(value, memo)
            except Exception:
                return dict(items)
    if value is None or isinstance(
        value,
        (
            bool,
            int,
            float,
            complex,
            str,
            bytes,
            torch.dtype,
            torch.device,
            torch.layout,
        ),
    ):
        return value
    try:
        return copy.deepcopy(value, memo)
    except Exception:
        # Trusted providers may pass opaque handles which deliberately disable
        # copying. Tensor/container inputs are still isolated above.
        return value


def clone_invocation(scenario: Scenario) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Clone one invocation while preserving aliases across args and kwargs.

    This helper is intentionally called before entering a dispatch recorder.
    Input isolation is capture setup, not model execution, and therefore must
    never add ``detach``/``clone`` operations to the authoritative ledger.
    """
    memo: dict[int, Any] = {}
    args = clone_inputs(scenario.args, memo)
    kwargs = clone_inputs(scenario.kwargs, memo)
    return tuple(args), dict(kwargs)


def invoke(model: nn.Module, scenario: Scenario) -> Any:
    args, kwargs = clone_invocation(scenario)
    return model(*args, **kwargs)


def schema_name(func: Any) -> tuple[str, str, str]:
    schema = getattr(func, "_schema", None)
    if schema is not None:
        base = str(schema.name)
        overload = str(schema.overload_name or "default")
    else:
        text = str(func)
        parts = text.split(".")
        base = ".".join(parts[:2]).replace(".", "::", 1)
        overload = parts[2] if len(parts) > 2 else "default"
    exact = base if overload == "default" else f"{base}.{overload}"
    namespace = base.split("::", 1)[0] if "::" in base else "unknown"
    return namespace, exact, overload


def _python_callsite() -> str | None:
    for frame in inspect.stack(context=0)[2:]:
        filename = os.path.abspath(frame.filename)
        normalized = filename.replace("\\", "/")
        if "/torch/" not in normalized and filename != os.path.abspath(__file__):
            return f"{filename}:{frame.lineno}"
    return None


def _uses_custom_forward(module: nn.Module | None) -> bool:
    """Cheaply distinguish user-defined forwards from stock module forwards."""
    if module is None:
        return True
    module_cls = type(module)
    # ``forward`` can be monkeypatched on one instance without changing its
    # stock torch.nn class.  Such code must receive the same callsite treatment
    # as a custom subclass and can never be accepted by class name alone.
    if "forward" in module.__dict__:
        return True
    if module_cls.__module__.startswith("torch.nn"):
        return False
    forward = getattr(module_cls, "forward", None)
    for base in module_cls.__mro__[1:]:
        if base.__module__.startswith("torch.nn") and "forward" in base.__dict__:
            return forward is not base.__dict__["forward"]
    return True


class DispatchRecorder(TorchDispatchMode):
    def __init__(
        self,
        scenario: str,
        active_modules: list[tuple[str, nn.Module]],
        *,
        capture_callsites: bool = True,
        callsite_keys: set[tuple[str, str]] | None = None,
        active_arguments: list[Any] | None = None,
        module_aliases: Mapping[int, tuple[str, ...]] | None = None,
    ):
        super().__init__()
        self.scenario = scenario
        self.active_modules = active_modules
        self.capture_callsites = capture_callsites
        self.active_arguments = active_arguments
        self.module_aliases = dict(module_aliases or {})
        self.records: list[OperationRecord] = []
        self._callsite_keys = callsite_keys if callsite_keys is not None else set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        namespace, exact, overload = schema_name(func)
        path, module = self.active_modules[-1] if self.active_modules else ("", None)
        callsite = None
        callsite_key = (path, exact)
        if (
            self.capture_callsites
            and _uses_custom_forward(module)
            and callsite_key not in self._callsite_keys
        ):
            self._callsite_keys.add(callsite_key)
            callsite = _python_callsite()
        record = OperationRecord(
            sequence=len(self.records),
            scenario=self.scenario,
            namespace=namespace,
            schema=exact,
            overload=overload,
            module_path=path,
            module_type=(
                f"{type(module).__module__}.{type(module).__qualname__}"
                if module is not None
                else None
            ),
            arguments={"args": value_metadata(args), "kwargs": value_metadata(kwargs)},
            output=None,
            module_aliases=(
                list(self.module_aliases.get(id(module), (path,)))
                if module is not None
                else []
            ),
            module_stack=[
                {
                    "path": owner_path,
                    "type": (f"{type(owner).__module__}.{type(owner).__qualname__}"),
                    "aliases": list(self.module_aliases.get(id(owner), (owner_path,))),
                }
                for owner_path, owner in self.active_modules
            ],
            module_arguments=(
                self.active_arguments[-1] if self.active_arguments else None
            ),
            callsite=callsite,
        )
        self.records.append(record)
        result = func(*args, **kwargs)
        record.output = value_metadata(result)
        return result


@contextmanager
def module_ownership(
    model: nn.Module,
    *,
    invoked: set[str] | None = None,
    active_arguments: list[Any] | None = None,
    on_enter=None,
    module_aliases: Mapping[int, tuple[str, ...]] | None = None,
):
    active: list[tuple[str, nn.Module]] = []
    handles = []
    invocation_counts: dict[str, int] = {}

    if module_aliases is None:
        alias_lists: dict[int, list[str]] = {}
        for alias_path, alias_module in model.named_modules(remove_duplicate=False):
            alias_lists.setdefault(id(alias_module), []).append(alias_path)
        module_aliases = {
            identity: tuple(paths) for identity, paths in alias_lists.items()
        }

    def pre(path, module):
        def hook(_module, _args, _kwargs=None):
            active.append((path, module))
            invocation = invocation_counts.get(path, 0)
            invocation_counts[path] = invocation + 1
            if active_arguments is not None:
                active_arguments.append(
                    {
                        "args": value_metadata(_args),
                        "kwargs": value_metadata({} if _kwargs is None else _kwargs),
                        "invocation": invocation,
                    }
                )
            if invoked is not None:
                invoked.update(module_aliases.get(id(module), (path,)))
            if on_enter is not None:
                on_enter(path, module, active)

        return hook

    def post(path, module):
        def hook(_module, _args, _output):
            for index in range(len(active) - 1, -1, -1):
                if active[index][1] is module:
                    del active[index:]
                    if active_arguments is not None:
                        del active_arguments[index:]
                    break

        return hook

    try:
        seen: set[int] = set()
        for path, module in model.named_modules(remove_duplicate=False):
            if id(module) in seen:
                continue
            seen.add(id(module))
            pre_hook = pre(path, module)
            if active_arguments is not None:
                try:
                    handles.append(
                        module.register_forward_pre_hook(
                            pre_hook,
                            with_kwargs=True,
                            prepend=True,
                        )
                    )
                except TypeError:  # older torch
                    handles.append(module.register_forward_pre_hook(pre_hook))
            else:
                try:
                    handles.append(
                        module.register_forward_pre_hook(pre_hook, prepend=True)
                    )
                except TypeError:
                    handles.append(module.register_forward_pre_hook(pre_hook))
            try:
                handles.append(
                    module.register_forward_hook(post(path, module), always_call=True)
                )
            except TypeError:  # older torch
                handles.append(module.register_forward_hook(post(path, module)))
        yield active
    finally:
        for handle in handles:
            handle.remove()


def capture_scenario(
    model: nn.Module,
    scenario: Scenario,
    *,
    capture_callsites: bool = True,
    invoked_modules: set[str] | None = None,
    operation_sink: list[OperationRecord] | None = None,
    callsite_keys: set[tuple[str, str]] | None = None,
    track_storage: bool = False,
) -> tuple[list[OperationRecord], Any]:
    """``track_storage``: tensor metadata also carries ``storage`` (its address), for data flow."""
    token = _TRACK_STORAGE.set(track_storage)
    try:
        return _capture_scenario(model, scenario, capture_callsites=capture_callsites, invoked_modules=invoked_modules,
                                 operation_sink=operation_sink, callsite_keys=callsite_keys)
    finally:
        _TRACK_STORAGE.reset(token)


def _capture_scenario(
    model: nn.Module,
    scenario: Scenario,
    *,
    capture_callsites: bool = True,
    invoked_modules: set[str] | None = None,
    operation_sink: list[OperationRecord] | None = None,
    callsite_keys: set[tuple[str, str]] | None = None,
) -> tuple[list[OperationRecord], Any]:
    args, kwargs = clone_invocation(scenario)
    active_arguments: list[Any] = []
    alias_lists: dict[int, list[str]] = {}
    for path, module in model.named_modules(remove_duplicate=False):
        alias_lists.setdefault(id(module), []).append(path)
    module_aliases = {identity: tuple(paths) for identity, paths in alias_lists.items()}
    with module_ownership(
        model,
        invoked=invoked_modules,
        active_arguments=active_arguments,
        module_aliases=module_aliases,
    ) as active:
        recorder = DispatchRecorder(
            scenario.name,
            active,
            capture_callsites=capture_callsites,
            callsite_keys=callsite_keys,
            active_arguments=active_arguments,
            module_aliases=module_aliases,
        )
        try:
            with torch.inference_mode(), recorder:
                output = model(*args, **kwargs)
        finally:
            if operation_sink is not None:
                operation_sink.extend(recorder.records)
    return recorder.records, output
