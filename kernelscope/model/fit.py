"""Fit the surrogate model's constants to measured kernel times (spec §3.2).

Machine constants are measured (machines/<gpu>.json) and never fitted. Per cache state the fit
runs in three stages so that each stage has few parameters and only rows it can identify them from.
"""
from dataclasses import dataclass, replace

import numpy as np

from kernelscope.analytic import DTYPE_BYTES
from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
from kernelscope.model.geometry import KIND_RESOURCES, build_launch, parse_variant
from kernelscope.model.params import KindParams, ModelParams, StateParams
from kernelscope.model.predict import bandwidth_bytes_per_us
from kernelscope.model.simulate import simulate_launch
from kernelscope.workload import Workload


@dataclass
class Row:
    plugin: str
    workload: Workload
    cache_state: str
    measured_us: float
    kind: str
    splits: int
    keys: np.ndarray
    slots: int
    bpk: int
    bw: float
    comb_n: int
    n_sm: int
    sm_order: np.ndarray


def prepare_rows(df, machine) -> list:
    states = df["cache_state"].fillna("warm") if "cache_state" in df.columns else "warm"
    sel = df.assign(cache_state=states)
    sel = sel[(sel.backend == "profile") & (sel.metric == "kernel_time_us")]
    med = sel.groupby(["kernel", "workload_key", "cache_state"])["value"].median()
    rows = []
    for (plugin, key, state), t in med.items():
        try:
            v = parse_variant(plugin)
            w = Workload.from_key(key)
            launch = build_launch(w, v, machine.n_sm)
        except ValueError:
            continue
        threads, regs, smem = KIND_RESOURCES[launch.kind]
        slots, _ = blocks_per_sm_limit(threads, regs, smem, machine.props())
        bpk = 2 * w.d * DTYPE_BYTES[w.dtype]
        bw = bandwidth_bytes_per_us(machine, state, float(launch.keys.sum()) * bpk)
        comb_n = w.B * w.H_q * launch.splits if launch.splits > 1 else 0
        rows.append(Row(plugin, w, state, float(t), launch.kind, launch.splits, launch.keys, slots, bpk, bw, comb_n,
                        machine.n_sm, machine.sm_order()))
    return rows


def row_time_us(row: Row, sp: StateParams) -> float:
    kp = sp.kinds[row.kind]
    r = simulate_launch(row.keys, n_sm=row.n_sm, slots_per_sm=row.slots, sm_order=row.sm_order,
                        cost_us_per_key=kp.cost_us_per_key, t0_us=kp.t0_us, t_empty_us=kp.t_empty_us,
                        gamma=kp.gamma, bytes_per_key=row.bpk, bw_bytes_per_us=row.bw)
    comb = sp.comb_a_us + sp.comb_b_us * row.comb_n if row.comb_n else 0.0
    return r.makespan_us + kp.t_fixed_us + comb


def nelder_mead(f, x0, step=0.2, maxiter=400, xtol=1e-6, ftol=1e-10):
    x0 = np.asarray(x0, dtype=float)
    n = len(x0)
    simplex = np.vstack([x0] + [x0 + step * np.eye(n)[i] for i in range(n)])
    fv = np.array([f(x) for x in simplex])
    for _ in range(maxiter):
        o = np.argsort(fv)
        simplex, fv = simplex[o], fv[o]
        if abs(fv[-1] - fv[0]) < ftol and np.max(np.abs(simplex[1:] - simplex[0])) < xtol:
            break
        c = simplex[:-1].mean(axis=0)
        xr = c + (c - simplex[-1]); fr = f(xr)
        if fv[0] <= fr < fv[-2]:
            simplex[-1], fv[-1] = xr, fr
        elif fr < fv[0]:
            xe = c + 2 * (xr - c); fe = f(xe)
            simplex[-1], fv[-1] = (xe, fe) if fe < fr else (xr, fr)
        else:
            xc = c + 0.5 * (simplex[-1] - c); fc = f(xc)
            if fc < fv[-1]:
                simplex[-1], fv[-1] = xc, fc
            else:
                simplex[1:] = simplex[0] + 0.5 * (simplex[1:] - simplex[0])
                fv[1:] = [f(x) for x in simplex[1:]]
    o = int(np.argmin(fv))
    return simplex[o], float(fv[o])


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _gamma_to_z(g):
    s = min(max((g - 0.05) / 0.95, 1e-6), 1 - 1e-6)
    return float(np.log(s / (1 - s)))


def _subsample(rows, cap):
    rows = sorted(rows, key=lambda r: (r.workload.key(), r.plugin))
    if len(rows) <= cap:
        return rows
    step = len(rows) / cap
    return [rows[int(i * step)] for i in range(cap)]


def _msle(rows, sp):
    return float(np.mean([(np.log(max(row_time_us(r, sp), 1e-9)) - np.log(r.measured_us)) ** 2 for r in rows]))


def fit_state(rows, init: StateParams, max_rows_per_group=250, maxiter=400, log=print) -> StateParams:
    sp = init
    ns = _subsample([r for r in rows if r.kind == "nonsplit"], max_rows_per_group)
    if ns:
        k = sp.kinds["nonsplit"]
        def f(x):
            kp = replace(k, cost_us_per_key=np.exp(x[0]), t0_us=np.exp(x[1]), gamma=0.05 + 0.95 * _sig(x[2]),
                         t_fixed_us=np.exp(x[3]))
            return _msle(ns, replace(sp, kinds={**sp.kinds, "nonsplit": kp}))
        x, fx = nelder_mead(f, [np.log(k.cost_us_per_key), np.log(max(k.t0_us, 1e-3)), _gamma_to_z(k.gamma),
                                np.log(max(k.t_fixed_us, 1e-3))], maxiter=maxiter)
        k = replace(k, cost_us_per_key=float(np.exp(x[0])), t0_us=float(np.exp(x[1])),
                    gamma=float(0.05 + 0.95 * _sig(x[2])), t_fixed_us=float(np.exp(x[3])))
        sp = replace(sp, kinds={**sp.kinds, "nonsplit": k})
        if log:
            log(f"nonsplit: {len(ns)} rows, msle {fx:.4g}, {k}")
    gamma = sp.kinds["nonsplit"].gamma
    for kind, with_combine in (("split", True), ("split_paged", False)):
        rs = _subsample([r for r in rows if r.kind == kind], max_rows_per_group)
        k = replace(sp.kinds[kind], gamma=gamma)
        if not rs:
            sp = replace(sp, kinds={**sp.kinds, kind: k})
            continue
        x0 = [np.log(k.cost_us_per_key), np.log(max(k.t0_us, 1e-3)), np.log(max(k.t_empty_us, 1e-3)),
              np.log(max(k.t_fixed_us, 1e-3))]
        if with_combine:
            x0 += [np.log(max(sp.comb_a_us, 1e-3)), np.log(max(sp.comb_b_us, 1e-6))]

        def build(x, k=k, kind=kind, with_combine=with_combine):
            kp = replace(k, cost_us_per_key=float(np.exp(x[0])), t0_us=float(np.exp(x[1])),
                         t_empty_us=float(np.exp(x[2])), t_fixed_us=float(np.exp(x[3])))
            new = replace(sp, kinds={**sp.kinds, kind: kp})
            if with_combine:
                new = replace(new, comb_a_us=float(np.exp(x[4])), comb_b_us=float(np.exp(x[5])))
            return new

        x, fx = nelder_mead(lambda x: _msle(rs, build(x)), x0, maxiter=maxiter)
        sp = build(x)
        if log:
            log(f"{kind}: {len(rs)} rows, msle {fx:.4g}, {sp.kinds[kind]}")
    return sp


def fit_model(rows, init: ModelParams, **kw) -> ModelParams:
    states = sorted({r.cache_state for r in rows})
    return replace(init, states={**init.states, **{
        s: fit_state([r for r in rows if r.cache_state == s], init.states[s], **kw) for s in states}})
