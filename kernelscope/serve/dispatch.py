"""Interchangeable paged attention policies; selection overhead is timed by Engine."""
import math
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.model.hybrid import hybrid_pick, variant_for_splits
from kernelscope.model.predict import PAGED_VARIANTS, predict
from kernelscope.model.simulate import prepare_simulator
from kernelscope.workload import Workload

PAGE = 256
FLASHINFER_BACKENDS = ("flashinfer", "flashinfer_cudacore")


class Policy(Protocol):
    name: str

    def choose(self, lens: list[int], n_heads: int, n_kv_heads: int) -> int: ...


def _validate(lens, n_heads, n_kv_heads):
    if not lens or any(not isinstance(x, (int, np.integer)) or x <= 0 for x in lens):
        raise ValueError("dispatch requires nonempty positive integer KV lengths")
    if n_heads <= 0 or n_kv_heads <= 0 or n_heads % n_kv_heads:
        raise ValueError("query heads must be a positive multiple of KV heads")


class FixedPolicy:
    """A fixed split count; ``attention`` names the kernel backend the engine runs it on (``fa2`` or a
    FlashInfer variant, which schedules the batch itself and ignores the split count)."""
    attention = "fa2"

    def __init__(self, num_splits: int, name: str | None = None, attention: str = "fa2"):
        if isinstance(num_splits, bool) or not isinstance(num_splits, int) or not 0 <= num_splits <= 128:
            raise ValueError("num_splits must be an integer in [0, 128]")
        if attention != "fa2" and attention not in FLASHINFER_BACKENDS:
            raise ValueError(f"unknown attention backend {attention!r}")
        self.num_splits, self.attention = num_splits, attention
        self.name = name or {0: "heuristic", 1: "fa2"}.get(num_splits, f"fixed{num_splits}")

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        _validate(lens, n_heads, n_kv_heads)
        return self.num_splits


def _workload(lens, n_heads, n_kv_heads):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=n_heads,
                    H_kv=n_kv_heads, d=128, dtype="bfloat16", kv_lens=tuple(lens))


class ModelPolicy:
    name = "model"
    attention = "fa2"

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


# What a table policy may pick from, in tie-break order: the order idxmin sees in static_default_losses
# (FA2 split variants sort before the FlashInfer ones).
TABLE_BACKENDS = ("fa2",) + FLASHINFER_BACKENDS
_FLASHINFER_CANDIDATES = {"flashinfer": ("flashinfer_paged", "flashinfer_tc_us"),
                          "flashinfer_cudacore": ("flashinfer_paged_cudacore", "flashinfer_cc_us")}


def _finite(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool) and math.isfinite(x)


class TablePolicy:
    """Nearest measured cell, then the cheapest candidate that cell lists; ``attention`` holds the pick's backend.

    A candidate is (backend, kernel, num_splits, kernel_us). ``fa2`` is the cell's best FA2 split count
    (``fa2_best_kernel``, or ``best_kernel`` in the FA2-only tables); a FlashInfer backend needs the
    ``flashinfer_tc_us`` / ``flashinfer_cc_us`` column of ``static_default_losses`` and ignores the split count.
    Cost = n_layers * kernel_us, plus ``plan_us`` (the per-step ``plan()`` call) for a FlashInfer backend; with
    ``plan_us == 0`` that is the kernel-time-only best. ``n_layers`` is filled in by Engine from the model.
    """

    def __init__(self, csv_path, backends=("fa2",), plan_us=0.0, name="table"):
        backends = tuple(backends)
        if not backends or len(set(backends)) != len(backends) or not set(backends) <= set(TABLE_BACKENDS):
            raise ValueError(f"backends must be distinct names from {TABLE_BACKENDS}")
        if not _finite(plan_us) or plan_us < 0:
            raise ValueError("plan_us must be a nonnegative number")
        self.path, self.backends, self.plan_us, self.name = Path(csv_path), backends, float(plan_us), name
        self.attention, self.n_layers = backends[0], None
        table = pd.read_csv(self.path)
        if table.empty or not {"workload_key", "best_kernel"} <= set(table.columns):
            raise ValueError("dispatch table needs nonempty workload_key and best_kernel columns")
        if "complete" in table.columns:
            table = table[table["complete"].astype(str).str.lower() != "false"]
        self._groups = {}
        for row in table.to_dict("records"):
            w = Workload.from_key(row["workload_key"])
            fa2_kernel = str(row["fa2_best_kernel"] if isinstance(row.get("fa2_best_kernel"), str) else row["best_kernel"])
            if w.phase != "decode" or w.L_q != 1 or w.d != 128 or not fa2_kernel.endswith("_paged"):
                raise ValueError("serving dispatch table must contain paged decode workloads with d=128")
            candidates = []
            for backend in backends:
                if backend == "fa2":
                    if fa2_kernel in {kernel for kernel, _ in _FLASHINFER_CANDIDATES.values()}:
                        raise ValueError(f"{w.key()}: best_kernel {fa2_kernel} is not an FA2 variant and the table has no "
                                         "fa2_best_kernel column")
                    us = row.get("fa2_best_us") if _finite(row.get("fa2_best_us")) else row.get("best_us")
                    candidates.append((backend, fa2_kernel, parse_variant(fa2_kernel).num_splits,
                                       float(us) if _finite(us) else math.nan))
                else:
                    kernel, column = _FLASHINFER_CANDIDATES[backend]
                    if _finite(row.get(column)):
                        candidates.append((backend, kernel, 0, float(row[column])))
            if not candidates:
                raise ValueError(f"{w.key()}: no candidate among {backends} has a measured time")
            self._groups.setdefault((w.H_q, w.H_kv), []).append((w, candidates))
        if not self._groups:
            raise ValueError("dispatch table has no complete rows")
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

    def covers(self, n_heads, n_kv_heads) -> bool:
        return (n_heads, n_kv_heads) in self._groups

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        _validate(lens, n_heads, n_kv_heads)
        if self.plan_us > 0 and self.n_layers is None:
            raise ValueError("plan_us > 0 needs n_layers; Engine sets it from the model")
        key = (n_heads, n_kv_heads, tuple(math.ceil(n / PAGE) for n in lens))
        self.last_cache_hit = key in self._cache
        if self.last_cache_hit:
            value, self.attention, self.last = self._cache[key]
            return value
        eligible = self._groups.get((n_heads, n_kv_heads), [])
        if not eligible:
            raise ValueError(f"dispatch table has no measured shape with Hq={n_heads}, Hkv={n_kv_heads}")
        query = self._features(lens)
        distances = np.square(self._features_by_shape[(n_heads, n_kv_heads)] - query).sum(axis=1)
        index = int(np.argmin(distances))
        workload, candidates = eligible[index]
        # An unknown kernel time (old FA2-only tables) costs nan: reported as such, ranked last.
        costs = [(self.n_layers or 1) * us + (self.plan_us if backend != "fa2" else 0.0) for backend, _, _, us in candidates]
        pick = min(range(len(costs)), key=lambda i: costs[i] if math.isfinite(costs[i]) else math.inf)
        self.attention, kernel, splits = candidates[pick][0], candidates[pick][1], candidates[pick][2]
        self.last = {"workload_key": workload.key(), "distance": float(distances[index] ** 0.5), "kernel": kernel,
                     "attention": self.attention, "cost_us": costs[pick]}
        self._cache[key] = (splits, self.attention, self.last)
        return splits


class HybridPolicy:
    """Model ranking, but the measured table's variant wins when the model predicts it within ``delta``."""
    name = "hybrid"
    attention = "fa2"

    def __init__(self, machine, params, csv_path, delta=0.10, cache_state="cold"):
        if not (isinstance(delta, (int, float)) and delta >= 0):
            raise ValueError("hybrid delta must be a nonnegative number")
        self.model = ModelPolicy(machine, params, cache_state=cache_state)
        self.table = TablePolicy(csv_path)
        self.delta = float(delta)
        self.simulator_backend = self.model.simulator_backend
        self.last = None
        self.last_cache_hit = False

    def reset(self):
        self.model.reset()
        self.table.reset()
        self.last = None
        self.last_cache_hit = False

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        model_splits = self.model.choose(lens, n_heads, n_kv_heads)
        predicted = dict(self.model.last)
        if not self.table.covers(n_heads, n_kv_heads):          # no measured grid for this head shape
            self.last_cache_hit = self.model.last_cache_hit
            self.last = {"source": "model", "pick": min(predicted, key=predicted.get), "model": predicted, "table": None}
            return model_splits
        table_splits = self.table.choose(lens, n_heads, n_kv_heads)
        self.last_cache_hit = self.model.last_cache_hit and self.table.last_cache_hit
        table_variant = variant_for_splits(table_splits)
        pick = hybrid_pick(predicted, table_variant, self.delta)
        source = "table" if pick == table_variant else "model"
        self.last = {"source": source, "pick": pick, "model": predicted, "table": self.table.last}
        return table_splits if source == "table" else model_splits


def make_policy(spec: str, machine=None, params=None, cache_state="cold") -> Policy:
    if spec == "fa2":
        return FixedPolicy(1)
    if spec == "heuristic":
        return FixedPolicy(0)
    if spec in FLASHINFER_BACKENDS:
        return FixedPolicy(0, name=spec, attention=spec)
    if spec.startswith("fixed:"):
        return FixedPolicy(int(spec.split(":", 1)[1]))
    if spec == "model":
        return ModelPolicy(machine, params, cache_state=cache_state)
    if spec.startswith("table:"):
        return TablePolicy(Path(spec.split(":", 1)[1]))
    if spec.startswith("table_any:"):      # FA2 splits and both FlashInfer variants; optional plan() cost in us per step
        rest = spec.split(":", 1)[1]
        path, _, plan = rest.rpartition(":") if rest.count(":") else (rest, "", "")
        plan_us = float(plan) if plan else 0.0
        return TablePolicy(Path(path), backends=TABLE_BACKENDS, plan_us=plan_us,
                           name=f"table_any_p{plan_us:g}" if plan else "table_any")
    if spec.startswith("hybrid:"):
        rest = spec.split(":", 1)[1]
        path, _, delta = rest.rpartition(":") if rest.count(":") else (rest, "", "")
        return HybridPolicy(machine, params, Path(path), delta=float(delta) if delta else 0.10, cache_state=cache_state)
    raise ValueError(f"unknown policy {spec!r}; use fa2, heuristic, fixed:N, model, table:PATH, "
                     "table_any:PATH[:PLAN_US], hybrid:PATH[:DELTA], flashinfer, flashinfer_cudacore")
