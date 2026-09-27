"""Event-driven processor-sharing simulation of one kernel launch (spec §3.2).

State is kept per resident slot (n_sm * slots_per_sm of them). Between events every rate is
constant; the next event is the earliest end of a fixed phase (startup) or of a CTA's keys. All
CTAs reaching zero at that instant retire together, so equal-work waves cost one event each.

The original NumPy algorithm is retained as ``simulate_launch_reference``. An
optional CPU C implementation performs the same events without per-event NumPy
allocations. ``prepare_simulator`` explicitly reports its one-time local build or
load cost; importing this module never compiles code. No workload is preloaded.
"""
from dataclasses import dataclass
import math
import operator
import os

import numpy as np

_EPS_KEYS = 1e-6
_EPS_US = 1e-9


@dataclass(frozen=True)
class LaunchResult:
    makespan_us: float
    bw_bound_us: float       # time during which the bandwidth cap was binding
    events: int


def _simulate_launch_reference(keys, *, n_sm, slots_per_sm, sm_order, cost_us_per_key, t0_us, t_empty_us, gamma,
                               bytes_per_key, bw_bytes_per_us, max_events=2_000_000) -> LaunchResult:
    """Original algorithm; only a nonfinite-progress guard has been added."""
    keys = np.asarray(keys, dtype=np.float64)
    n = len(keys)
    if n == 0:
        return LaunchResult(0.0, 0.0, 0)
    n_slots = n_sm * slots_per_sm
    slot_sm = np.asarray(sm_order, dtype=np.int64)[np.arange(n_slots) % n_sm]
    occupied = np.zeros(n_slots, dtype=bool)
    rem_fixed = np.zeros(n_slots)
    rem_keys = np.zeros(n_slots)
    nxt = 0

    def admit(slots):
        nonlocal nxt
        m = min(len(slots), n - nxt)
        if m <= 0:
            return
        s = slots[:m]
        k = keys[nxt:nxt + m]
        nxt += m
        occupied[s] = True
        rem_keys[s] = k
        rem_fixed[s] = np.where(k > 0, t0_us, t_empty_us)

    admit(np.arange(n_slots))
    t = bw_t = 0.0
    events = 0
    while occupied.any():
        events += 1
        if events > max_events:
            raise RuntimeError(f"simulation exceeded {max_events} events")
        per_sm = np.bincount(slot_sm[occupied], minlength=n_sm)
        g = np.where(per_sm[slot_sm] >= 2, gamma, 1.0)
        fixed = occupied & (rem_fixed > _EPS_US)
        streaming = occupied & ~fixed & (rem_keys > _EPS_KEYS)
        rate = np.where(streaming, g / cost_us_per_key, 0.0)
        demand = rate.sum() * bytes_per_key
        scale = 1.0 if demand <= bw_bytes_per_us else bw_bytes_per_us / demand
        rate *= scale
        done_now = occupied & ~fixed & ~streaming
        if done_now.any():
            dt = 0.0
        else:
            dt_fixed = np.min(rem_fixed[fixed]) if fixed.any() else np.inf
            dt_keys = np.min(rem_keys[streaming] / rate[streaming]) if streaming.any() else np.inf
            dt = min(dt_fixed, dt_keys)
        if not np.isfinite(dt) or dt < 0:
            raise RuntimeError("simulation encountered nonfinite or negative progress")
        t += dt
        if scale < 1.0:
            bw_t += dt
        if not np.isfinite(t) or not np.isfinite(bw_t):
            raise RuntimeError("simulation time overflowed float64")
        rem_fixed[fixed] -= dt
        rem_keys[streaming] -= rate[streaming] * dt
        finished = occupied & (rem_fixed <= _EPS_US) & (rem_keys <= _EPS_KEYS)
        occupied[finished] = False
        admit(np.flatnonzero(finished))
    return LaunchResult(t, bw_t, events)


def _integer(value, name):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    if result > np.iinfo(np.intp).max:
        raise ValueError(f"{name} exceeds the native integer range")
    return result


def _validate(keys, parameters):
    try:
        keys = np.asarray(keys, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("keys must be a one-dimensional finite nonnegative array") from exc
    if keys.ndim != 1 or not np.isfinite(keys).all() or (keys < 0).any():
        raise ValueError("keys must be a one-dimensional finite nonnegative array")
    keys = np.ascontiguousarray(keys)
    for name in ("n_sm", "slots_per_sm", "max_events"):
        parameters[name] = _integer(parameters[name], name)
    n_sm = parameters["n_sm"]
    if n_sm * parameters["slots_per_sm"] > np.iinfo(np.intp).max // 8:
        raise ValueError("resident slot count exceeds addressable storage")
    order = np.asarray(parameters["sm_order"])
    if (order.ndim != 1 or len(order) != n_sm or not np.issubdtype(order.dtype, np.integer)
            or not np.array_equal(np.sort(order), np.arange(n_sm))):
        raise ValueError("sm_order must be an integer permutation of range(n_sm)")
    parameters["sm_order"] = np.ascontiguousarray(order, dtype=np.int64)
    for name in ("cost_us_per_key", "t0_us", "t_empty_us", "gamma", "bytes_per_key", "bw_bytes_per_us"):
        try:
            value = float(parameters[name])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        positive = name in ("cost_us_per_key", "gamma", "bw_bytes_per_us")
        if not math.isfinite(value) or (value <= 0 if positive else value < 0):
            raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
        parameters[name] = value
    if not math.isfinite(max(1.0, parameters["gamma"]) / parameters["cost_us_per_key"]):
        raise ValueError("per-CTA streaming rate overflows float64")
    return keys, parameters


def _mode(backend):
    mode = backend or os.environ.get("KERNELSCOPE_SIMULATOR", "auto")
    if mode not in ("auto", "native", "python"):
        raise ValueError("simulator backend must be auto, native, or python")
    return mode


def prepare_simulator(backend=None) -> dict:
    """Prepare a CPU backend outside measured policy decisions and report cost.

    ``auto`` uses the reference if a compiler/build is unavailable; ``native``
    fails explicitly. Set KERNELSCOPE_SIMULATOR=python to force the oracle.
    Setup does not inspect, predict, fit, or cache any workload.
    """
    mode = _mode(backend)
    if mode == "python":
        return {"backend": "python_reference", "requested_backend": mode,
                "native_available": False, "built_this_process": False,
                "cache_hit": False, "setup_us": 0.0, "cache_path": None,
                "compiler": None, "source_sha256": None, "error": None}
    from kernelscope.model._simulate_backend import prepare_native
    function, metadata = prepare_native()
    if function is None and mode == "native":
        raise RuntimeError(metadata["error"] or "native simulator is unavailable")
    return {**metadata, "requested_backend": mode}


def simulate_launch_reference(keys, *, n_sm, slots_per_sm, sm_order, cost_us_per_key,
                              t0_us, t_empty_us, gamma, bytes_per_key, bw_bytes_per_us,
                              max_events=2_000_000) -> LaunchResult:
    """Validated reference entry point for differential tests and reproducibility."""
    keys, parameters = _validate(keys, dict(n_sm=n_sm, slots_per_sm=slots_per_sm, sm_order=sm_order,
        cost_us_per_key=cost_us_per_key, t0_us=t0_us, t_empty_us=t_empty_us, gamma=gamma,
        bytes_per_key=bytes_per_key, bw_bytes_per_us=bw_bytes_per_us, max_events=max_events))
    return _simulate_launch_reference(keys, **parameters)


def simulate_launch(keys, *, n_sm, slots_per_sm, sm_order, cost_us_per_key, t0_us,
                    t_empty_us, gamma, bytes_per_key, bw_bytes_per_us,
                    max_events=2_000_000, backend=None) -> LaunchResult:
    keys, parameters = _validate(keys, dict(n_sm=n_sm, slots_per_sm=slots_per_sm, sm_order=sm_order,
        cost_us_per_key=cost_us_per_key, t0_us=t0_us, t_empty_us=t_empty_us, gamma=gamma,
        bytes_per_key=bytes_per_key, bw_bytes_per_us=bw_bytes_per_us, max_events=max_events))
    mode = _mode(backend)
    if not len(keys):
        return LaunchResult(0.0, 0.0, 0)
    if mode == "python":
        return _simulate_launch_reference(keys, **parameters)
    from kernelscope.model._simulate_backend import prepare_native
    function, metadata = prepare_native()
    if function is None:
        if mode == "native":
            raise RuntimeError(metadata["error"] or "native simulator is unavailable")
        return _simulate_launch_reference(keys, **parameters)
    output = np.empty(3, dtype=np.float64)
    status = function(keys, len(keys), parameters["n_sm"], parameters["slots_per_sm"], parameters["sm_order"],
        parameters["cost_us_per_key"], parameters["t0_us"], parameters["t_empty_us"], parameters["gamma"],
        parameters["bytes_per_key"], parameters["bw_bytes_per_us"], parameters["max_events"], output)
    if status == 1:
        raise RuntimeError(f"simulation exceeded {parameters['max_events']} events")
    if status == 2:
        raise RuntimeError("simulation encountered nonfinite arithmetic or time overflow")
    if status == 3:
        raise MemoryError("native simulator could not allocate resident slots")
    if status:
        raise RuntimeError(f"native simulator returned unknown status {status}")
    return LaunchResult(float(output[0]), float(output[1]), int(output[2]))
