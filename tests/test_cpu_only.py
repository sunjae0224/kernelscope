"""Modules on the GPU-free path must import on a CPU-only host, where torch ships without triton."""
import importlib
import sys

import pytest

import kernelscope.bench as bench


def _import_without_triton(monkeypatch, name):
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "triton.language", None)
    for attr in ("stream", "machine"):
        monkeypatch.delitem(sys.modules, f"kernelscope.bench.{attr}", raising=False)
        if hasattr(bench, attr):   # re-importing rebinds bench.<attr>; record it so teardown restores the original
            monkeypatch.setattr(bench, attr, getattr(bench, attr))
    return importlib.import_module(name)


def test_bench_machine_imports_without_triton(monkeypatch):
    machine = _import_without_triton(monkeypatch, "kernelscope.bench.machine")
    assert machine.hit_fraction(2000.0, 1000.0, 5000.0) == pytest.approx(0.5)


def test_stream_gbps_says_triton_is_needed(monkeypatch):
    stream = _import_without_triton(monkeypatch, "kernelscope.bench.stream")
    assert stream.num_chunks(1 << 20, G=128) == 8
    with pytest.raises(RuntimeError, match="triton"):
        stream.stream_gbps(1 << 30, 512)
