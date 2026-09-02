"""Regression tests for distributed DataProto chunking."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


class _FakeBatch:
    """Small TensorDict stand-in that records slice boundaries."""

    def __init__(self, start: int, stop: int):
        self.start = start
        self.stop = stop
        self.batch_size = (stop - start,)

    def __getitem__(self, item):
        assert isinstance(item, slice)
        start = self.start if item.start is None else self.start + item.start
        stop = self.stop if item.stop is None else self.start + item.stop
        return _FakeBatch(start, min(stop, self.stop))


def _load_protocol():
    """Load protocol.py without requiring the optional runtime stack."""
    tensor_dict_module = types.ModuleType("tensordict")
    tensor_dict_module.TensorDict = _FakeBatch
    tensor_dict_module.__version__ = "0.0.0"
    tensor_dict_module.set_lazy_legacy = lambda enabled: types.SimpleNamespace(set=lambda: None)

    verl_module = types.ModuleType("verl")
    utils_module = types.ModuleType("verl.utils")
    functional_module = types.ModuleType("verl.utils.py_functional")
    functional_module.union_two_dict = lambda left, right: {**left, **right}
    ray_module = types.ModuleType("ray")
    ray_module.ObjectRef = object

    modules = {
        "tensordict": tensor_dict_module,
        "verl": verl_module,
        "verl.utils": utils_module,
        "verl.utils.py_functional": functional_module,
        "ray": ray_module,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location(
            "simplevla_protocol_under_test", ROOT / "verl" / "protocol.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def test_chunk_returns_requested_number_for_non_divisible_batch():
    module = _load_protocol()
    batch_size = 124
    non_tensor = np.asarray(list(range(batch_size)), dtype=object)
    proto = module.DataProto(
        batch=_FakeBatch(0, batch_size),
        non_tensor_batch={"sample_id": non_tensor},
        meta_info={"tag": "test"},
    )

    chunks = proto.chunk(15)

    assert len(chunks) == 15
    assert [len(part) for part in chunks] == [9] * 4 + [8] * 11
    assert [part.non_tensor_batch["sample_id"].tolist() for part in chunks] == [
        list(range(start, start + size))
        for start, size in zip([0, 9, 18, 27, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116],
                                [9] * 4 + [8] * 11)
    ]


def test_chunk_rejects_non_positive_chunk_count():
    module = _load_protocol()
    proto = module.DataProto(batch=_FakeBatch(0, 2), meta_info={})

    with pytest.raises(ValueError, match="positive"):
        proto.chunk(0)
