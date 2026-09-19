"""Event-driven processor-sharing simulation of one kernel launch (spec §3.2).

State is kept per resident slot (n_sm * slots_per_sm of them). Between events every rate is
constant; the next event is the earliest end of a fixed phase (startup) or of a CTA's keys. All
CTAs reaching zero at that instant retire together, so equal-work waves cost one event each.
"""
from dataclasses import dataclass

import numpy as np

_EPS_KEYS = 1e-6
_EPS_US = 1e-9


@dataclass(frozen=True)
class LaunchResult:
    makespan_us: float
    bw_bound_us: float       # time during which the bandwidth cap was binding
    events: int


def simulate_launch(keys, *, n_sm, slots_per_sm, sm_order, cost_us_per_key, t0_us, t_empty_us, gamma,
                    bytes_per_key, bw_bytes_per_us, max_events=2_000_000) -> LaunchResult:
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
        t += dt
        if scale < 1.0:
            bw_t += dt
        rem_fixed[fixed] -= dt
        rem_keys[streaming] -= rate[streaming] * dt
        finished = occupied & (rem_fixed <= _EPS_US) & (rem_keys <= _EPS_KEYS)
        occupied[finished] = False
        admit(np.flatnonzero(finished))
    return LaunchResult(t, bw_t, events)
