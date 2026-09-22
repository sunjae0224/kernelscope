"""Interchangeable paged attention policies; selection overhead is timed by Engine."""
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.model.predict import PAGED_VARIANTS, predict
from kernelscope.model.simulate import prepare_simulator
from kernelscope.workload import Workload

PAGE = 256


class Policy(Protocol):
    name: str

    def choose(self, lens: list[int], n_heads: int, n_kv_heads: int) -> int: ...


def _validate(lens, n_heads, n_kv_heads):
    if not lens or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in lens):
        raise ValueError("dispatch requires nonempty positive integer KV lengths")
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")


class FixedPolicy:
    def __init__(self, num_splits: int, name: str | None = None):
        if isinstance(num_splits, bool) or not isinstance(num_splits, int) or not 0 <= num_splits <= 128:
            raise ValueError("num_splits must be an integer in [0, 128]")
        self.num_splits = num_splits
        self.name = name or {0: "heuristic", 1: "fa2"}.get(num_splits, f"fixed{num_splits}")

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        _validate(lens, n_heads, n_kv_heads)
        return self.num_splits


def _workload(lens, n_heads, n_kv_heads):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=n_heads,
                    H_kv=n_kv_heads, d=128, dtype="bfloat16", kv_lens=tuple(lens))


class ModelPolicy:
    name = "model"

    def __init__(self, machine, params, cache_state="cold", candidates=PAGED_VARIANTS):
        if machine is None or params is None:
            raise ValueError("model policy requires --machine and --params")
        if cache_state not in params.states:
            raise ValueError(f"model parameters do not cover cache state {cache_state!r}")
        self.machine, self.params, self.cache_state = machine, params, cache_state
        self.candidates = list(candidates)
        if not self.candidates or any(not parse_variant(c).paged for c in self.candidates):
            raise ValueError("model dispatch requires at least one paged candidate")
        # Initialize outside request timing; the CLI records this startup cost
        # and backend provenance separately from every timed choose() call.
        self.simulator_backend = prepare_simulator()
        self._cache = {}
        self.predictions = 0
        self.last = []
        self.last_cache_hit = False

    def reset(self):
        self._cache.clear()
        self.predictions = 0
        self.last = []

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        _validate(lens, n_heads, n_kv_heads)
        # Sequence order matters to CTA placement. Page quantization is an explicit
        # approximation: reuse the first ranking within each page configuration.
        key = (n_heads, n_kv_heads, tuple(math.ceil(n / PAGE) for n in lens))
        self.last_cache_hit = key in self._cache
        if key not in self._cache:
            w = _workload(lens, n_heads, n_kv_heads)
            ranked = sorted(((c, float(predict(c, w, self.machine, self.params, self.cache_state).time_us))
                             for c in self.candidates), key=lambda item: item[1])
            if any(not math.isfinite(t) or t <= 0 for _, t in ranked):
                raise ValueError("surrogate produced a nonpositive or nonfinite prediction")
            self.predictions += len(self.candidates)
            self._cache[key] = (resolve_splits(parse_variant(ranked[0][0]), w, self.machine.n_sm), ranked)
        value, self.last = self._cache[key]
        return value


class TablePolicy:
    name = "table"

    def __init__(self, csv_path):
        self.path = Path(csv_path)
        table = pd.read_csv(self.path)
        if table.empty or not {"workload_key", "best_kernel"} <= set(table.columns):
            raise ValueError("dispatch table needs nonempty workload_key and best_kernel columns")
        self._rows = []
        for row in table.itertuples():
            w, v = Workload.from_key(row.workload_key), parse_variant(row.best_kernel)
            if w.phase != "decode" or w.L_q != 1 or w.d != 128 or not v.paged:
                raise ValueError("serving dispatch table must contain paged decode workloads with d=128")
            self._rows.append((w, v.num_splits))
        self._groups = {}
        for w, splits in self._rows:
            self._groups.setdefault((w.H_q, w.H_kv), []).append((w, splits))
        self._features_by_shape = {shape: np.asarray([self._features(w.lens()) for w, _ in rows])
                                   for shape, rows in self._groups.items()}
        self._cache = {}
        self.last = None
        self.last_cache_hit = False

    def reset(self):
        self._cache.clear()
        self.last = None
        self.last_cache_hit = False

    @staticmethod
    def _features(lens):
        return np.log2([len(lens), max(lens), float(np.mean(lens))])

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        _validate(lens, n_heads, n_kv_heads)
        key = (n_heads, n_kv_heads, tuple(math.ceil(n / PAGE) for n in lens))
        self.last_cache_hit = key in self._cache
        if self.last_cache_hit:
            value, self.last = self._cache[key]
            return value
        eligible = self._groups.get((n_heads, n_kv_heads), [])
        if not eligible:
            raise ValueError(f"dispatch table has no measured shape with Hq={n_heads}, Hkv={n_kv_heads}")
        query = self._features(lens)
        distances = np.square(self._features_by_shape[(n_heads, n_kv_heads)] - query).sum(axis=1)
        index = int(np.argmin(distances))
        workload, value = eligible[index]
        self.last = {"workload_key": workload.key(), "distance": float(distances[index] ** 0.5)}
        self._cache[key] = (value, self.last)
        return value


def make_policy(spec: str, machine=None, params=None, cache_state="cold") -> Policy:
    if spec == "fa2":
        return FixedPolicy(1)
    if spec == "heuristic":
        return FixedPolicy(0)
    if spec.startswith("fixed:"):
        return FixedPolicy(int(spec.split(":", 1)[1]))
    if spec == "model":
        return ModelPolicy(machine, params, cache_state=cache_state)
    if spec.startswith("table:"):
        return TablePolicy(Path(spec.split(":", 1)[1]))
    raise ValueError(f"unknown policy {spec!r}; use fa2, heuristic, fixed:N, model, table:PATH")
