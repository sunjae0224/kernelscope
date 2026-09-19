"""Predicted kernel time of a flash-attn decode variant, with a bottleneck breakdown."""
from dataclasses import dataclass

from kernelscope.analytic import DTYPE_BYTES
from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
from kernelscope.model.geometry import KIND_RESOURCES, build_launch, parse_variant
from kernelscope.model.simulate import simulate_launch

SPLITS = (2, 4, 8, 16, 32, 64, 128)
DENSE_VARIANTS = ["fa2", "flashdecoding"] + [f"fd_s{n}" for n in SPLITS]
PAGED_VARIANTS = [v + "_paged" for v in DENSE_VARIANTS]


def bandwidth_bytes_per_us(machine, cache_state: str, streamed_bytes: float) -> float:
    dram = machine.dram_gbps * 1e3
    l2 = machine.l2_gbps * 1e3
    if cache_state == "cold":
        return dram
    h = machine.l2_hit(streamed_bytes)
    return l2 if h >= 1.0 else min(l2, dram / (1.0 - h))


@dataclass(frozen=True)
class Prediction:
    plugin: str
    kind: str
    splits: int
    ctas: int
    slots_per_sm: int
    limiter: str
    main_us: float
    combine_us: float
    time_us: float
    bw_bound_fraction: float


def predict(plugin: str, w, machine, params, cache_state: str) -> Prediction:
    v = parse_variant(plugin)
    launch = build_launch(w, v, machine.n_sm)
    sp = params.states[cache_state]
    kp = sp.kinds[launch.kind]
    threads, regs, smem = KIND_RESOURCES[launch.kind]
    slots, limiter = blocks_per_sm_limit(threads, regs, smem, machine.props())
    bpk = 2 * w.d * DTYPE_BYTES[w.dtype]
    streamed = float(launch.keys.sum()) * bpk
    r = simulate_launch(launch.keys, n_sm=machine.n_sm, slots_per_sm=slots, sm_order=machine.sm_order(),
                        cost_us_per_key=kp.cost_us_per_key, t0_us=kp.t0_us, t_empty_us=kp.t_empty_us,
                        gamma=kp.gamma, bytes_per_key=bpk,
                        bw_bytes_per_us=bandwidth_bytes_per_us(machine, cache_state, streamed))
    main = r.makespan_us + kp.t_fixed_us
    comb = sp.comb_a_us + sp.comb_b_us * w.B * w.H_q * launch.splits if launch.splits > 1 else 0.0
    return Prediction(plugin, launch.kind, launch.splits, len(launch.keys), slots, limiter, main, comb, main + comb,
                      r.bw_bound_us / r.makespan_us if r.makespan_us > 0 else 0.0)


def rank_variants(w, plugins, machine, params, cache_state: str) -> list:
    return sorted((predict(p, w, machine, params, cache_state) for p in plugins), key=lambda p: p.time_us)
