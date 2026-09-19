# Surrogate Performance Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A fast, calibrated model that predicts the kernel time of every flash-attn decode variant (fa2, the library heuristic, fixed splits, dense and paged) on the measured RTX 4090 and on hypothetical variants of it (fewer/more SMs, more/less DRAM bandwidth, larger/smaller L2, more shared memory), validated against held-out measurements and against real SM-count what-ifs.

**Architecture:** `kernelscope/model/` — `machine.py` (measured `MachineSpec`, scaling), `geometry.py` (variant → per-CTA work in hardware block order, including an exact port of flash-attn's `num_splits_heuristic`), `simulate.py` (vectorised event-driven processor-sharing simulation of one launch: resident slots per SM from occupancy, measured block→SM placement, per-SM throughput sharing, a global bandwidth cap), `params.py`/`predict.py` (fitted per-kernel constants → prediction with a bottleneck breakdown), `fit.py` (Nelder–Mead on measured campaign rows), `validate.py` (spec §3.3 V1/V2/V5/V6), `blocked.py` (V3: real SM-count what-if measured with the SM blocker). CLI `kernelscope model {fit,predict,validate,blocked}`.

**Tech Stack:** Python 3.11, numpy, pandas, torch 2.8 (GPU steps only). No scipy (not installed; do not install it).

**Spec:** `docs/plan/2026-09-19-design-surrogate-dispatcher.md` §3 (model) and §1 (facts F1–F20). The model form was validated by a throwaway spike on the probe data (2026-09-19): heuristic port matched 290/290 measured split counts; cold median/p90 absolute error 7–9 % / 19–21 % across in-sample, cross-shape and ragged sets; the model's chosen split had ≤ 2.2 % regret where the library heuristic reached 705 %; removing the bandwidth cap raised the ragged median error from 8 % to 56 %. This plan fixes the spike's two identified defects (co-residency fixed at CTA start; warm working set taken from allocated rather than streamed bytes).

## Global Constraints

- **Worktree:** `/home/skkai/AI_Accelerator/kernelscope-design`, branch `design-1-3`. Never edit `/home/skkai/AI_Accelerator/kernelscope` (Codex's checkout). Measurement output for Task 8 goes to `/home/skkai/AI_Accelerator/kernelscope/results/hw_4090/blocked_s1/` (git-ignored, shared).
- **Python:** `PY=/home/skkai/miniforge3/envs/gradkernel/bin/python`, always by absolute path. Run the whole suite with one `$PY -m pytest -q -p no:cacheprovider` (tests/conftest.py orders GPU tests first; see the torch.profiler note in `kernelscope/backends/realhw/kprofile.py`).
- **No package installs.** numpy/pandas only for fitting.
- **Codex-owned, never edit:** `kernelscope/backends/accelsim/**`, `_cmd_simsweep` and the `p_sim` block of `kernelscope/cli.py`, `tests/test_accelsim_*.py`, `tests/fixtures/SM*`, `docs/setup/**`, `env/setup_accelsim_4090.sh`.
- **Scope of the model:** decode, `L_q == 1`, `d == 128`, dtype `float16` or `bfloat16`, flash-attn variants only (`fa2`, `flashdecoding`, `fd_s{N}`, and their `_paged` forms). `predict` raises `ValueError` for anything else.
- **Units:** times in µs; bandwidth inside the simulator in bytes/µs (1 GB/s = 1000 bytes/µs).
- **Measured inputs:** `machines/rtx4090.json` (committed by Phase 0) and the campaign rows under `/home/skkai/AI_Accelerator/kernelscope/results/hw_4090/{uniform_s1_dense,uniform_s1_paged,uniform_s2_dense,uniform_s2_paged,ragged_s1_dense,ragged_s1_paged,ragged_s2_paged}`.
- **Kernel resources (measured, campaign profile rows):** non-split `flash_fwd_kernel` 128 threads / 255 regs / 49152 B smem (2 CTAs/SM on the 4090); `flash_fwd_splitkv_kernel` (dense and paged) 128 / 244 / 81920 (1 CTA/SM). The combine kernel is modelled as a linear cost, not simulated.
- **GPU hygiene** (Task 8, Task 9): check `nvidia-smi` first; only the `rerun` viewer (~440 MiB, 0 %) is acceptable; otherwise wait (poll 60 s, up to 20 min) then report BLOCKED. Never kill processes.
- **Commits:** `git add` new files first, then `git commit -m "<msg>" -- <paths>`; end messages with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Run your task's tests, then the whole suite; report failures in files you did not touch instead of fixing them.

## Execution order

Sequential: Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9.

---

### Task 1: MachineSpec

**Files:**
- Create: `kernelscope/model/__init__.py` (empty docstring module), `kernelscope/model/machine.py`
- Test: `tests/test_model_machine.py`

**Interfaces:**
- Produces: `MachineSpec` (frozen dataclass) with fields `name: str, n_sm: int, max_threads_sm: int, max_ctas_sm: int, regs_sm: int, smem_sm: int, reserved_smem_per_block: int, l2_bytes: int, dram_gbps: float, l2_gbps: float, l2_curve: tuple[tuple[float, float], ...], l2_curve_ref_bytes: int`; `MachineSpec.from_json(path)`; `.props() -> dict` (the kprofile props dict accepted by `blocks_per_sm_limit`); `.sm_order() -> np.ndarray`; `.l2_hit(working_set_bytes) -> float`; `.scaled(sm=1.0, dram=1.0, l2=1.0, smem=1.0) -> MachineSpec`.

- [ ] **Step 1: Write the failing tests** — create `tests/test_model_machine.py`:

```python
import json

import numpy as np
import pytest

from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
from kernelscope.model.machine import MachineSpec

SPEC = {
    "name": "NVIDIA GeForce RTX 4090", "cc": "8.9", "n_sm": 128, "max_threads_sm": 1536, "max_ctas_sm": 24,
    "regs_sm": 65536, "smem_sm": 102400, "reserved_smem_per_block": 1024, "l2_bytes": 75497472,
    "dram_gbps": 952.6, "l2_gbps": 4561.0, "cta_dram_gbps": 26.0, "cta_l2_gbps": 46.4,
    "l2_hit_curve": [{"mib": 16, "gbps": 4850, "hit": 1.0}, {"mib": 72, "gbps": 4793, "hit": 1.0},
                     {"mib": 96, "gbps": 3866, "hit": 0.754}, {"mib": 128, "gbps": 953, "hit": 0.0}],
    "tc_tflops": 168.6, "clock_mhz": 3105.0, "block_placement": None, "provenance": {},
}


@pytest.fixture
def m(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(SPEC))
    return MachineSpec.from_json(p)


def test_loads_the_measured_fields(m):
    assert m.n_sm == 128 and m.max_ctas_sm == 24 and m.smem_sm == 102400
    assert m.dram_gbps == 952.6 and m.l2_gbps == 4561.0
    assert m.l2_curve[0] == (16 * 2**20, 1.0) and m.l2_curve_ref_bytes == 75497472


def test_props_feed_the_occupancy_calculator(m):
    assert blocks_per_sm_limit(128, 244, 81920, m.props()) == (1, "smem")
    assert blocks_per_sm_limit(128, 255, 49152, m.props()) == (2, "regs")


def test_sm_order_puts_even_sms_first(m):
    o = m.sm_order()
    assert list(o[:4]) == [0, 2, 4, 6] and list(o[64:66]) == [1, 3] and sorted(o) == list(range(128))


def test_l2_hit_interpolates_and_clamps(m):
    assert m.l2_hit(1 * 2**20) == 1.0
    assert m.l2_hit(72 * 2**20) == 1.0
    assert m.l2_hit(84 * 2**20) == pytest.approx((1.0 + 0.754) / 2)
    assert m.l2_hit(1 << 30) == 0.0


def test_scaling_changes_one_resource_each(m):
    half = m.scaled(sm=0.5)
    assert half.n_sm == 64 and list(half.sm_order()[:2]) == [0, 2]
    assert m.scaled(dram=2).dram_gbps == pytest.approx(1905.2)
    big = m.scaled(l2=2)
    assert big.l2_bytes == 2 * m.l2_bytes
    assert big.l2_hit(144 * 2**20) == 1.0        # the curve scales with capacity
    assert m.scaled(smem=1.64).smem_sm == 167936
    assert blocks_per_sm_limit(128, 244, 81920, m.scaled(smem=1.64).props())[0] == 2
    assert "sm=0.5" in half.name
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_model_machine.py -q` → `ModuleNotFoundError: kernelscope.model`.

- [ ] **Step 3: Implement.** `kernelscope/model/__init__.py`:
```python
"""Surrogate performance model of flash-attn decode kernels (spec §3)."""
```
`kernelscope/model/machine.py`:
```python
"""Measured machine parameters (machines/<gpu>.json) and hypothetical variants of them.

A hypothetical machine scales one resource of the measured one: SM count, DRAM bandwidth, L2
capacity (the measured hit-rate curve is stretched along the working-set axis) or shared memory
per SM (which changes how many CTAs of a kernel fit on an SM).
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MIB = 2**20


@dataclass(frozen=True)
class MachineSpec:
    name: str
    n_sm: int
    max_threads_sm: int
    max_ctas_sm: int
    regs_sm: int
    smem_sm: int
    reserved_smem_per_block: int
    l2_bytes: int
    dram_gbps: float
    l2_gbps: float
    l2_curve: tuple            # ((working_set_bytes, hit), ...) measured at l2_curve_ref_bytes
    l2_curve_ref_bytes: int

    @classmethod
    def from_json(cls, path) -> "MachineSpec":
        d = json.loads(Path(path).read_text())
        curve = tuple((float(p["mib"]) * MIB, float(p["hit"])) for p in d["l2_hit_curve"])
        return cls(name=d["name"], n_sm=d["n_sm"], max_threads_sm=d["max_threads_sm"], max_ctas_sm=d["max_ctas_sm"],
                   regs_sm=d["regs_sm"], smem_sm=d["smem_sm"], reserved_smem_per_block=d["reserved_smem_per_block"],
                   l2_bytes=d["l2_bytes"], dram_gbps=float(d["dram_gbps"]), l2_gbps=float(d["l2_gbps"]),
                   l2_curve=curve, l2_curve_ref_bytes=d["l2_bytes"])

    def props(self) -> dict:
        return {"num_sms": self.n_sm, "max_threads_per_sm": self.max_threads_sm, "regs_per_sm": self.regs_sm,
                "smem_per_sm": self.smem_sm, "max_blocks_per_sm": self.max_ctas_sm,
                "reserved_smem_per_block": self.reserved_smem_per_block, "warp_size": 32,
                "max_warps_per_sm": self.max_threads_sm // 32}

    def sm_order(self) -> np.ndarray:
        """Order in which the first wave of blocks lands on SMs (measured: even SMs, then odd; spec F19)."""
        return np.concatenate([np.arange(0, self.n_sm, 2), np.arange(1, self.n_sm, 2)])

    def l2_hit(self, working_set_bytes: float) -> float:
        """Share of streamed bytes served by L2 for a working set, from the measured curve."""
        x = working_set_bytes * self.l2_curve_ref_bytes / self.l2_bytes
        xs = [p[0] for p in self.l2_curve]
        hs = [p[1] for p in self.l2_curve]
        return float(np.interp(x, xs, hs, left=hs[0], right=hs[-1]))

    def scaled(self, sm: float = 1.0, dram: float = 1.0, l2: float = 1.0, smem: float = 1.0) -> "MachineSpec":
        tags = [f"{k}={v:g}" for k, v in (("sm", sm), ("dram", dram), ("l2", l2), ("smem", smem)) if v != 1.0]
        return replace(self, name=self.name + (" [" + ",".join(tags) + "]" if tags else ""),
                       n_sm=max(1, round(self.n_sm * sm)), dram_gbps=self.dram_gbps * dram,
                       l2_bytes=round(self.l2_bytes * l2), smem_sm=round(self.smem_sm * smem))
```

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_model_machine.py -q` → PASS. Also: `$PY -c "from kernelscope.model.machine import MachineSpec as M; m=M.from_json('machines/rtx4090.json'); print(m.n_sm, m.dram_gbps, m.l2_gbps, m.l2_hit(96*2**20))"` → 128, ~952.6, ~4561, ~0.75.
- [ ] **Step 5: Whole suite, commit** — `git add kernelscope/model/__init__.py kernelscope/model/machine.py tests/test_model_machine.py && git commit -m "Add MachineSpec for the surrogate model" -- kernelscope/model/__init__.py kernelscope/model/machine.py tests/test_model_machine.py`

---

### Task 2: Launch geometry and the heuristic port

**Files:**
- Create: `kernelscope/model/geometry.py`
- Test: `tests/test_model_geometry.py`

**Interfaces:**
- Consumes: `Workload` (`lens()`, `B`, `H_q`, `H_kv`, `L_kv`, `d`, `dtype`, `phase`, `L_q`).
- Produces:
  - `BLOCK_N = 128`, `PAGE = 256`, `KIND_RESOURCES = {"nonsplit": (128, 255, 49152), "split": (128, 244, 81920), "split_paged": (128, 244, 81920)}`
  - `Variant(name: str, num_splits: int, paged: bool)` and `parse_variant(name) -> Variant` (`ValueError` for non-flash names)
  - `num_splits_heuristic(batch_nheads_mblocks, num_sms, num_n_blocks, max_splits=128) -> int` (exact port)
  - `capacity(w, paged) -> int`; `resolve_splits(variant, w, n_sm) -> int`
  - `Launch(kind: str, splits: int, keys: np.ndarray)`; `build_launch(w, variant, n_sm) -> Launch` — keys per CTA in hardware linear block order; `check_scope(w)` raising `ValueError` outside the model scope

- [ ] **Step 1: Write the failing tests** — `tests/test_model_geometry.py`:

```python
import numpy as np
import pytest

from kernelscope.model.geometry import (Variant, build_launch, capacity, check_scope, num_splits_heuristic,
                                        parse_variant, resolve_splits)
from kernelscope.workload import Workload


def _w(lens, H_q=32, H_kv=8):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=H_q, H_kv=H_kv, d=128, kv_lens=lens)


def test_parse_variant_covers_the_flash_family():
    assert parse_variant("fa2") == Variant("fa2", 1, False)
    assert parse_variant("flashdecoding") == Variant("flashdecoding", 0, False)
    assert parse_variant("fd_s16") == Variant("fd_s16", 16, False)
    assert parse_variant("fa2_paged") == Variant("fa2_paged", 1, True)
    assert parse_variant("flashdecoding_paged") == Variant("flashdecoding_paged", 0, True)
    assert parse_variant("fd_s4_paged") == Variant("fd_s4_paged", 4, True)
    with pytest.raises(ValueError):
        parse_variant("sdpa_flash")


@pytest.mark.parametrize("bnh, nblocks, expected", [
    (8, 8, 8),        # B1 L1K: 64 CTAs (spec STATUS)
    (8, 64, 32),      # B1 L8K: 256 CTAs (gate report)
    (128, 8, 2),      # B16 L1K
    (256, 256, 1),    # B32: at/above 0.8 * 256 -> no split (spec F15)
    (4, 64, 64),      # S2 B1 L8K
])
def test_heuristic_port_matches_measured_split_counts(bnh, nblocks, expected):
    assert num_splits_heuristic(bnh, 256, nblocks, 128) == expected


def test_paged_capacity_rounds_up_to_whole_pages():
    w = _w([1000, 300])
    assert capacity(w, paged=False) == 1000
    assert capacity(w, paged=True) == 1024


def test_resolve_splits_uses_the_heuristic_only_for_num_splits_zero():
    w = _w([8192])
    assert resolve_splits(parse_variant("flashdecoding"), w, 128) == 32
    assert resolve_splits(parse_variant("fd_s4"), w, 128) == 4
    assert resolve_splits(parse_variant("flashdecoding"), w, 64) == 16       # fewer SMs, fewer splits


def test_nonsplit_launch_orders_blocks_batch_fastest():
    L = build_launch(_w([100, 7], H_kv=2, H_q=4), parse_variant("fa2"), 128)
    assert L.kind == "nonsplit" and L.splits == 1
    assert list(L.keys) == [100, 7, 100, 7]          # linear id = b + B*h


def test_split_launch_follows_capacity_based_ranges():
    L = build_launch(_w([1000, 200], H_kv=1, H_q=1), parse_variant("fd_s4"), 128)
    # n_blocks = ceil(1000/128) = 8, per split = 2 blocks = 256 keys; linear id = s + S*(b*H_kv + h)
    assert L.kind == "split" and L.splits == 4
    assert list(L.keys) == [256, 256, 256, 232, 200, 0, 0, 0]


def test_paged_variants_always_use_the_split_kernel():
    L = build_launch(_w([512, 512]), parse_variant("fa2_paged"), 128)
    assert L.kind == "split_paged" and L.splits == 1 and len(L.keys) == 16


def test_scope_is_enforced():
    with pytest.raises(ValueError, match="d=128"):
        check_scope(Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=64))
    with pytest.raises(ValueError, match="decode"):
        check_scope(Workload(phase="prefill", B=1, L_q=64, L_kv=64, H_q=8, H_kv=2, d=128))
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `kernelscope/model/geometry.py`:**

```python
"""What a flash-attn decode variant launches: kernel kind, split count, and each CTA's key count
in hardware linear block order (sources: flash_api.cpp set_params_splitkv / num_splits_heuristic,
flash_fwd_launch_template.h grid shapes, flash_fwd_kernel.h split ranges; spec F2, F3, F15, F17).

Decode with GQA packs a KV head's query heads into one CTA, so a split launch has
S * B * H_kv CTAs, linear id = s + S * (b * H_kv + h), and the non-split launch has B * H_kv
CTAs, linear id = b + B * h. Split ranges are sized from the cache CAPACITY (dense: L_kv; paged:
whole pages), so CTAs whose range starts past a sequence's live length do no work.
"""
import math
import re
from dataclasses import dataclass

import numpy as np

BLOCK_N = 128
PAGE = 256
KIND_RESOURCES = {"nonsplit": (128, 255, 49152), "split": (128, 244, 81920), "split_paged": (128, 244, 81920)}


@dataclass(frozen=True)
class Variant:
    name: str
    num_splits: int          # 0 = the library heuristic
    paged: bool


def parse_variant(name: str) -> Variant:
    paged = name.endswith("_paged")
    base = name[: -len("_paged")] if paged else name
    if base == "fa2":
        return Variant(name, 1, paged)
    if base == "flashdecoding":
        return Variant(name, 0, paged)
    m = re.fullmatch(r"fd_s(\d+)", base)
    if m:
        return Variant(name, int(m.group(1)), paged)
    raise ValueError(f"{name!r} is not a flash-attn decode variant the model covers")


def check_scope(w) -> None:
    if w.phase != "decode" or w.L_q != 1:
        raise ValueError("the model covers decode with L_q=1 only")
    if w.d != 128 or w.dtype not in ("float16", "bfloat16"):
        raise ValueError(f"the model is calibrated for d=128 fp16/bf16 only, got d={w.d} {w.dtype}")


def num_splits_heuristic(batch_nheads_mblocks: int, num_sms: int, num_n_blocks: int, max_splits: int = 128) -> int:
    """Exact port of flash-attn 2.8.3 num_splits_heuristic (flash_api.cpp)."""
    if batch_nheads_mblocks >= 0.8 * num_sms:
        return 1
    max_splits = min(max_splits, num_sms, num_n_blocks)
    eff = [0.0] * (max_splits + 1)
    best = 0.0
    for s in range(1, max_splits + 1):
        if s > 1 and math.ceil(num_n_blocks / s) == math.ceil(num_n_blocks / (s - 1)):
            continue
        n_waves = batch_nheads_mblocks * s / num_sms
        eff[s] = n_waves / math.ceil(n_waves)
        best = max(best, eff[s])
    for s in range(1, max_splits + 1):
        if (s == 1 or math.ceil(num_n_blocks / s) != math.ceil(num_n_blocks / (s - 1))) and eff[s] >= 0.85 * best:
            return s
    return 1


def capacity(w, paged: bool) -> int:
    return math.ceil(w.L_kv / PAGE) * PAGE if paged else w.L_kv


def resolve_splits(v: Variant, w, n_sm: int) -> int:
    if v.num_splits > 0:
        return v.num_splits
    m_blocks = math.ceil((w.H_q // w.H_kv) / 64)       # the GQA-packed query rows, 64 per m-block
    return num_splits_heuristic(w.B * w.H_kv * m_blocks, 2 * n_sm, math.ceil(capacity(w, v.paged) / BLOCK_N), 128)


@dataclass(frozen=True)
class Launch:
    kind: str                # "nonsplit" | "split" | "split_paged"
    splits: int
    keys: np.ndarray         # keys per CTA, linear block order


def build_launch(w, v: Variant, n_sm: int) -> Launch:
    check_scope(w)
    lens = np.asarray(w.lens(), dtype=np.int64)
    S = resolve_splits(v, w, n_sm)
    if not v.paged and S == 1:
        ids = np.arange(w.B * w.H_kv)
        return Launch("nonsplit", 1, lens[ids % w.B])
    per_split = math.ceil(math.ceil(capacity(w, v.paged) / BLOCK_N) / S) * BLOCK_N
    ids = np.arange(S * w.B * w.H_kv)
    s = ids % S
    b = (ids // S) // w.H_kv
    lo = s * per_split
    keys = np.clip(np.minimum(lens[b], lo + per_split) - lo, 0, None)
    return Launch("split_paged" if v.paged else "split", S, keys)
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Heuristic check against the campaign** (CPU, reads shared results): write and run this check, paste its output into your report:
```bash
$PY - <<'EOF'
import pandas as pd
from kernelscope.results.store import ResultStore
from kernelscope.model.geometry import build_launch, parse_variant
from kernelscope.workload import Workload
R = "/home/skkai/AI_Accelerator/kernelscope/results/hw_4090"
bad = n = 0
for d in ("uniform_s1_dense", "uniform_s1_paged", "uniform_s2_dense", "uniform_s2_paged", "ragged_s1_dense", "ragged_s1_paged", "ragged_s2_paged"):
    df = ResultStore(f"{R}/{d}").load()
    g = df[(df.backend == "profile") & (df.metric == "grid_blocks") & (df.launch_idx == 0) & df.kernel.str.startswith("flashdecoding")]
    for (k, key), grp in g.groupby(["kernel", "workload_key"]):
        w = Workload.from_key(key)
        L = build_launch(w, parse_variant(k), 128)
        n += 1
        if int(grp.value.iloc[0]) != len(L.keys):
            bad += 1; print("MISMATCH", d, k, key, int(grp.value.iloc[0]), len(L.keys))
print(f"{n} heuristic cells, {bad} mismatches")
EOF
```
Expected: 0 mismatches. If any mismatch appears, stop and report it (do not change the port to fit).
- [ ] **Step 6: Whole suite, commit** — `git add kernelscope/model/geometry.py tests/test_model_geometry.py && git commit -m "Add launch geometry and an exact port of the flash-attn split heuristic" -- kernelscope/model/geometry.py tests/test_model_geometry.py`

---

### Task 3: The launch simulator

**Files:**
- Create: `kernelscope/model/simulate.py`
- Test: `tests/test_model_simulate.py`

**Interfaces:**
- Produces: `LaunchResult(makespan_us: float, bw_bound_us: float, events: int)`; `simulate_launch(keys, *, n_sm, slots_per_sm, sm_order, cost_us_per_key, t0_us, t_empty_us, gamma, bytes_per_key, bw_bytes_per_us, max_events=2_000_000) -> LaunchResult`.

Semantics (spec §3.2, F19): slot `j` belongs to SM `sm_order[j % n_sm]`; the first `n_sm * slots_per_sm` CTAs take slots in order; afterwards each freed slot takes the next CTA, freed slots in ascending index order. A CTA with keys spends `t0_us` (not rate-scaled) and then streams its keys; a CTA with zero keys only spends `t_empty_us`. A streaming CTA progresses at `g / cost_us_per_key` keys/µs where `g = gamma` if two or more CTAs are resident on its SM, else 1 — re-evaluated at every event, so a CTA speeds up when its partner leaves. If the streaming CTAs together ask for more than `bw_bytes_per_us`, all streaming rates are scaled down by the same factor. Events are processed in batches: every CTA finishing at the same instant is retired at once.

- [ ] **Step 1: Write the failing tests** — `tests/test_model_simulate.py`:

```python
import numpy as np
import pytest

from kernelscope.model.simulate import simulate_launch

BASE = dict(n_sm=1, slots_per_sm=1, sm_order=np.array([0]), cost_us_per_key=0.05, t0_us=0.0, t_empty_us=0.0,
            gamma=0.5, bytes_per_key=512, bw_bytes_per_us=1e12)


def run(keys, **kw):
    return simulate_launch(np.array(keys), **{**BASE, **kw})


def test_single_cta_is_startup_plus_keys_times_cost():
    assert run([1000], t0_us=2.0).makespan_us == pytest.approx(52.0)


def test_two_resident_ctas_share_their_sm():
    assert run([1000, 1000], slots_per_sm=2).makespan_us == pytest.approx(100.0)


def test_a_cta_speeds_up_when_its_partner_leaves():
    # both at 10 keys/us until the short one ends at 20 us, then 20 keys/us for the last 800 keys
    assert run([1000, 200], slots_per_sm=2).makespan_us == pytest.approx(60.0)


def test_bandwidth_cap_scales_every_streaming_cta():
    r = run([1000] * 4, n_sm=4, sm_order=np.arange(4), bw_bytes_per_us=20480.0)
    assert r.makespan_us == pytest.approx(100.0)          # 4 x 10240 B/us asked, 20480 available
    assert r.bw_bound_us == pytest.approx(100.0)


def test_waves_when_ctas_outnumber_slots():
    assert run([100] * 4, n_sm=2, sm_order=np.arange(2), cost_us_per_key=0.1).makespan_us == pytest.approx(20.0)


def test_empty_ctas_cost_only_their_fixed_time():
    assert run([0] * 10, t_empty_us=0.5, t0_us=9.0).makespan_us == pytest.approx(5.0)


def test_block_i_and_i_plus_n_sm_share_an_sm():
    # 4 SMs x 2 slots; CTAs 0 and 4 are long and land on the same SM, so they share it
    keys = [1000, 10, 10, 10, 1000, 10, 10, 10]
    r = run(keys, n_sm=4, slots_per_sm=2, sm_order=np.array([0, 2, 1, 3]), cost_us_per_key=0.01)
    assert r.makespan_us == pytest.approx(20.0)            # 1000 keys at gamma/cost = 50 keys/us


def test_many_equal_ctas_retire_in_batches():
    r = run([500] * 4096, n_sm=128, sm_order=np.arange(128))
    assert r.makespan_us == pytest.approx(32 * 25.0)
    assert r.events <= 2 * 32 + 2


def test_large_ragged_split_grid_is_fast_enough():
    import time
    keys = np.zeros(65536); keys[:256] = 128                # fd_s128-like grid, mostly empty CTAs
    t = time.perf_counter()
    run(keys, n_sm=128, sm_order=np.arange(128), t_empty_us=0.5)
    assert time.perf_counter() - t < 2.0
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `kernelscope/model/simulate.py`:**

```python
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
```

- [ ] **Step 4: Run** → PASS. If `test_many_equal_ctas_retire_in_batches` fails on the event count only, report the observed count (do not loosen it without saying so).
- [ ] **Step 5: Whole suite, commit** — `git add kernelscope/model/simulate.py tests/test_model_simulate.py && git commit -m "Add the processor-sharing launch simulator" -- kernelscope/model/simulate.py tests/test_model_simulate.py`

---

### Task 4: Parameters and prediction

**Files:**
- Create: `kernelscope/model/params.py`, `kernelscope/model/predict.py`
- Test: `tests/test_model_predict.py`

**Interfaces:**
- Consumes: `MachineSpec` (Task 1), `parse_variant`, `build_launch`, `KIND_RESOURCES` (Task 2), `simulate_launch` (Task 3), `kprofile.blocks_per_sm_limit`, `analytic.DTYPE_BYTES`.
- Produces:
  - `params.KindParams(cost_us_per_key, t0_us, t_empty_us, gamma, t_fixed_us)`; `params.StateParams(kinds: dict[str, KindParams], comb_a_us: float, comb_b_us: float)`; `params.ModelParams(machine: str, states: dict[str, StateParams])` with `.to_json(path)`, `ModelParams.from_json(path)`; `params.SPIKE_DEFAULTS: ModelParams` (initial values, below)
  - `predict.bandwidth_bytes_per_us(machine, cache_state, streamed_bytes) -> float`: cold → DRAM; warm → `h = machine.l2_hit(streamed)`, `l2` if `h >= 1` else `min(l2, dram / (1 - h))` (concurrent service, spec F6)
  - `predict.Prediction(plugin, kind, splits, ctas, slots_per_sm, limiter, main_us, combine_us, time_us, bw_bound_fraction)`
  - `predict.predict(plugin, w, machine, params, cache_state) -> Prediction`
  - `predict.rank_variants(w, plugins, machine, params, cache_state) -> list[Prediction]` sorted by `time_us`
  - `predict.DENSE_VARIANTS = ["fa2", "flashdecoding", "fd_s2", ..., "fd_s128"]`, `predict.PAGED_VARIANTS` (same with `_paged`)

- [ ] **Step 1: Write the failing tests** — `tests/test_model_predict.py`:

```python
import json

import pytest

from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import SPIKE_DEFAULTS, KindParams, ModelParams, StateParams
from kernelscope.model.predict import DENSE_VARIANTS, bandwidth_bytes_per_us, predict, rank_variants
from kernelscope.workload import Workload

M = MachineSpec(name="t", n_sm=128, max_threads_sm=1536, max_ctas_sm=24, regs_sm=65536, smem_sm=102400,
                reserved_smem_per_block=1024, l2_bytes=72 * 2**20, dram_gbps=953.0, l2_gbps=4850.0,
                l2_curve=((16 * 2**20, 1.0), (72 * 2**20, 1.0), (96 * 2**20, 0.75), (128 * 2**20, 0.0)),
                l2_curve_ref_bytes=72 * 2**20)
K = KindParams(cost_us_per_key=0.05, t0_us=2.0, t_empty_us=0.5, gamma=0.5, t_fixed_us=1.0)
P = ModelParams(machine="t", states={s: StateParams(kinds={"nonsplit": K, "split": K, "split_paged": K},
                                                     comb_a_us=5.0, comb_b_us=0.001) for s in ("cold", "warm")})
W1 = Workload(phase="decode", B=1, L_q=1, L_kv=8192, H_q=32, H_kv=8, d=128)


def test_bandwidth_by_cache_state():
    assert bandwidth_bytes_per_us(M, "cold", 1e6) == pytest.approx(953e3)
    assert bandwidth_bytes_per_us(M, "warm", 32 * 2**20) == pytest.approx(4850e3)
    assert bandwidth_bytes_per_us(M, "warm", 96 * 2**20) == pytest.approx(953e3 / 0.25)
    assert bandwidth_bytes_per_us(M, "warm", 1 << 30) == pytest.approx(953e3)


def test_fa2_single_sequence_is_eight_lonely_ctas():
    p = predict("fa2", W1, M, P, "cold")
    assert p.kind == "nonsplit" and p.ctas == 8 and p.slots_per_sm == 2 and p.limiter == "regs"
    assert p.main_us == pytest.approx(2.0 + 8192 * 0.05 + 1.0)
    assert p.combine_us == 0.0 and p.time_us == p.main_us


def test_split_variant_adds_a_combine_cost():
    p = predict("fd_s8", W1, M, P, "cold")
    assert p.kind == "split" and p.splits == 8 and p.ctas == 64 and p.slots_per_sm == 1
    assert p.combine_us == pytest.approx(5.0 + 0.001 * 1 * 32 * 8)
    assert p.time_us < predict("fa2", W1, M, P, "cold").time_us


def test_more_shared_memory_lets_the_split_kernel_pair_up():
    assert predict("fd_s8", W1, M.scaled(smem=1.64), P, "cold").slots_per_sm == 2


def test_rank_variants_sorts_by_predicted_time():
    r = rank_variants(W1, DENSE_VARIANTS, M, P, "cold")
    assert [x.time_us for x in r] == sorted(x.time_us for x in r)
    assert r[-1].plugin == "fa2"


def test_out_of_scope_workloads_raise():
    with pytest.raises(ValueError):
        predict("fa2", Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=64), M, P, "cold")


def test_params_round_trip(tmp_path):
    SPIKE_DEFAULTS.to_json(tmp_path / "p.json")
    assert ModelParams.from_json(tmp_path / "p.json") == SPIKE_DEFAULTS
    assert set(json.loads((tmp_path / "p.json").read_text())["states"]) == {"cold", "warm"}
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `kernelscope/model/params.py`:**

```python
"""Fitted constants of the surrogate model, per cache state and kernel kind (spec §3.2)."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

KINDS = ("nonsplit", "split", "split_paged")


@dataclass(frozen=True)
class KindParams:
    cost_us_per_key: float   # one CTA alone on its SM, per key of its KV head
    t0_us: float             # fixed startup of a CTA with work
    t_empty_us: float        # a CTA whose split range is past its sequence's length
    gamma: float             # per-CTA speed when two CTAs share an SM (spec F19)
    t_fixed_us: float        # per-launch constant


@dataclass(frozen=True)
class StateParams:
    kinds: dict              # kind -> KindParams
    comb_a_us: float         # combine kernel: a + b * (B * H_q * splits), only when splits > 1
    comb_b_us: float


@dataclass(frozen=True)
class ModelParams:
    machine: str
    states: dict             # "cold" | "warm" -> StateParams

    def to_json(self, path) -> None:
        d = {"machine": self.machine, "states": {s: {"kinds": {k: asdict(v) for k, v in sp.kinds.items()},
                                                     "comb_a_us": sp.comb_a_us, "comb_b_us": sp.comb_b_us}
                                                 for s, sp in self.states.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path) -> "ModelParams":
        d = json.loads(Path(path).read_text())
        return cls(machine=d["machine"], states={
            s: StateParams(kinds={k: KindParams(**v) for k, v in sp["kinds"].items()},
                           comb_a_us=sp["comb_a_us"], comb_b_us=sp["comb_b_us"])
            for s, sp in d["states"].items()})


def _state(c_ns, c_sp, t0_ns, t0_sp, gamma, comb_a, comb_b):
    ns = KindParams(c_ns, t0_ns, 0.0, gamma, 0.0)
    sp = KindParams(c_sp, t0_sp, 0.5, gamma, 0.0)
    return StateParams(kinds={"nonsplit": ns, "split": sp, "split_paged": sp}, comb_a_us=comb_a, comb_b_us=comb_b)


# Initial values from the 2026-09-19 spike fit (gamma clamped to <= 1; see spec F19).
SPIKE_DEFAULTS = ModelParams(machine="NVIDIA GeForce RTX 4090", states={
    "cold": _state(0.0544, 0.0242, 12.0, 0.83, 0.5, 14.3, 0.00104),
    "warm": _state(0.0562, 0.0282, 3.3, 0.07, 0.53, 6.95, 0.0017),
})
```

`kernelscope/model/predict.py`:
```python
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
```

- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Whole suite, commit** — `git add kernelscope/model/params.py kernelscope/model/predict.py tests/test_model_predict.py && git commit -m "Add model parameters and variant prediction" -- kernelscope/model/params.py kernelscope/model/predict.py tests/test_model_predict.py`

---

### Task 5: Fitting

**Files:**
- Create: `kernelscope/model/fit.py`
- Modify: `kernelscope/cli.py` (add `_cmd_model` and a `p_model` parser with sub-subcommand `fit` only; later tasks add `predict`, `validate`, `blocked` to the same `p_model` subparsers)
- Test: `tests/test_model_fit.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `fit.Row` (dataclass: `plugin, workload, cache_state, measured_us, kind, splits, keys, slots, bpk, bw, comb_n`) and `fit.prepare_rows(df, machine) -> list[Row]` — one row per (plugin, workload_key, cache_state) with the median `profile/kernel_time_us`; rows for plugins outside the flash family or workloads outside the scope are skipped; rows lacking `cache_state` are `warm`
  - `fit.row_time_us(row, state_params) -> float` (prediction from the precomputed fields; must equal `predict(...).time_us` for the same inputs)
  - `fit.nelder_mead(f, x0, step=0.2, maxiter=400) -> (x, fx)` (numpy)
  - `fit.fit_state(rows, init: StateParams, max_rows_per_group=250, maxiter=400, log=print) -> StateParams` — three stages: (1) `nonsplit` kind: `cost, t0, gamma, t_fixed`; (2) dense `split` kind with the combine: `cost, t0, t_empty, t_fixed, comb_a, comb_b`; (3) `split_paged`: `cost, t0, t_empty, t_fixed` with the combine from stage 2. The split kinds' `gamma` is set to the fitted non-split `gamma` (never identifiable on this GPU, where split CTAs never share an SM; documented assumption for smem what-ifs). Parameters are optimised in log space; `gamma = 0.05 + 0.95 * sigmoid(z)` (so 0.05 < gamma ≤ 1). Objective: mean squared log error. Groups with no rows keep their initial values. Rows per group are subsampled deterministically (sorted by `workload_key` then `plugin`, every k-th) to at most `max_rows_per_group`.
  - `fit.fit_model(rows, init: ModelParams, **kw) -> ModelParams` (one `fit_state` per cache state present)
  - CLI: `kernelscope model fit --results DIR [DIR ...] --machine machines/rtx4090.json --out models/rtx4090.json [--train uniform|all] [--max-rows 250] [--maxiter 400]`. `--train uniform` keeps only non-ragged workloads (used by Task 6's V6); `all` (default) keeps ragged rows whose `sum(ord(c) for c in workload_key)` is even — the other half is the ragged test set.

- [ ] **Step 1: Write the failing tests** — `tests/test_model_fit.py`:

```python
import numpy as np
import pandas as pd
import pytest

from kernelscope.model.fit import fit_state, nelder_mead, prepare_rows, row_time_us
from kernelscope.model.params import KindParams, StateParams
from kernelscope.model.predict import predict
from tests.test_model_predict import M, P

KEYS = ["decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal",
        "decode_B4_Lq1_Lkv16384_Hq32_Hkv8_d128_float16_causal",
        "decode_B32_Lq1_Lkv8192x2+1024x30_Hq32_Hkv8_d128_float16_causal",
        "decode_B64_Lq1_Lkv2048_Hq32_Hkv8_d128_float16_causal"]
PLUGINS = ["fa2", "flashdecoding", "fd_s4", "fd_s16", "fa2_paged", "fd_s8_paged"]


def _synthetic(params, noise=0.0):
    from kernelscope.workload import Workload
    rng = np.random.default_rng(0)
    rows = []
    for key in KEYS:
        for p in PLUGINS:
            t = predict(p, Workload.from_key(key), M, params, "cold").time_us * (1 + noise * rng.standard_normal())
            rows.append({"workload_key": key, "kernel": p, "backend": "profile", "metric": "kernel_time_us",
                         "unit": "us", "value": t, "launch_idx": 0, "note": None, "cache_state": "cold"})
    rows.append({"workload_key": KEYS[0], "kernel": "sdpa_flash", "backend": "profile", "metric": "kernel_time_us",
                 "unit": "us", "value": 1.0, "launch_idx": 0, "note": None, "cache_state": "cold"})
    return pd.DataFrame(rows)


def test_nelder_mead_finds_a_quadratic_minimum():
    x, fx = nelder_mead(lambda v: float(((v - np.array([1.0, -2.0])) ** 2).sum()), np.zeros(2), maxiter=500)
    assert np.allclose(x, [1.0, -2.0], atol=1e-3) and fx < 1e-6


def test_prepare_rows_keeps_flash_variants_and_precomputes_the_launch():
    rows = prepare_rows(_synthetic(P), M)
    assert len(rows) == len(KEYS) * len(PLUGINS)                 # sdpa_flash skipped
    r = [r for r in rows if r.plugin == "fd_s16" and r.workload.B == 1][0]
    assert r.kind == "split" and r.splits == 16 and r.slots == 1 and len(r.keys) == 128


def test_row_time_matches_predict():
    from kernelscope.workload import Workload
    for r in prepare_rows(_synthetic(P), M):
        assert row_time_us(r, P.states["cold"]) == pytest.approx(
            predict(r.plugin, Workload.from_key(r.workload.key()), M, P, "cold").time_us)


def test_fit_recovers_known_parameters_from_noise_free_data():
    rows = prepare_rows(_synthetic(P), M)
    k0 = KindParams(cost_us_per_key=0.03, t0_us=5.0, t_empty_us=0.2, gamma=0.8, t_fixed_us=3.0)
    init = StateParams(kinds={"nonsplit": k0, "split": k0, "split_paged": k0}, comb_a_us=10.0, comb_b_us=0.003)
    got = fit_state(rows, init, maxiter=600, log=None)
    for kind in ("nonsplit", "split", "split_paged"):
        assert got.kinds[kind].cost_us_per_key == pytest.approx(0.05, rel=0.05)
    errs = [abs(row_time_us(r, got) / r.measured_us - 1) for r in rows]
    assert np.median(errs) < 0.02
```

- [ ] **Step 2: Run to verify it fails** → `ModuleNotFoundError`.

- [ ] **Step 3: Implement `kernelscope/model/fit.py`:**

```python
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
```

`kernelscope/cli.py` — add the `model` command group:
```python
def _load_results(dirs):
    import pandas as pd
    return pd.concat([ResultStore(r).load() for r in dirs], ignore_index=True)


def _ragged_train_half(key: str) -> bool:
    return sum(map(ord, key)) % 2 == 0


def _cmd_model_fit(args):
    from kernelscope.model.fit import fit_model, prepare_rows
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import SPIKE_DEFAULTS
    m = MachineSpec.from_json(args.machine)
    rows = prepare_rows(_load_results(args.results), m)
    if args.train == "uniform":
        rows = [r for r in rows if not r.workload.is_ragged]
    else:
        rows = [r for r in rows if not r.workload.is_ragged or _ragged_train_half(r.workload.key())]
    params = fit_model(rows, SPIKE_DEFAULTS, max_rows_per_group=args.max_rows, maxiter=args.maxiter)
    params.to_json(args.out)
    print(f"wrote {args.out} ({len(rows)} training rows)")
```
parser (after `p_disp`):
```python
    p_model = sub.add_parser("model", help="surrogate performance model: fit / predict / validate / blocked")
    msub = p_model.add_subparsers(dest="model_cmd", required=True)
    p_fit = msub.add_parser("fit", help="fit model constants to measured bench results")
    p_fit.add_argument("--results", nargs="+", required=True)
    p_fit.add_argument("--machine", default="machines/rtx4090.json")
    p_fit.add_argument("--out", default="models/rtx4090.json")
    p_fit.add_argument("--train", choices=["all", "uniform"], default="all")
    p_fit.add_argument("--max-rows", type=int, default=250)
    p_fit.add_argument("--maxiter", type=int, default=400)
    p_fit.set_defaults(func=_cmd_model_fit)
```
Add `kernelscope model fit|predict|validate|blocked` to the module docstring.

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_model_fit.py -q` → PASS (the recovery test may take up to a few minutes; if it takes over 10 minutes, report the time).
- [ ] **Step 5: Whole suite, commit** — `git add kernelscope/model/fit.py tests/test_model_fit.py && git commit -m "Add staged Nelder-Mead fitting of the surrogate model" -- kernelscope/model/fit.py kernelscope/cli.py tests/test_model_fit.py`

---

### Task 6: Validation (V1, V2, V5, V6)

**Files:**
- Create: `kernelscope/model/validate.py`
- Modify: `kernelscope/cli.py` (add `model validate` to the `p_model` subparsers and `_cmd_model_validate`)
- Test: `tests/test_model_validate.py`

**Interfaces:**
- Consumes: `prepare_rows`, `row_time_us` (Task 5), `ModelParams`, `DENSE_VARIANTS`/`PAGED_VARIANTS` (Task 4).
- Produces:
  - `validate.error_stats(rows, params) -> dict` with `n, median_ape, p90_ape, worst` (worst = list of up to 5 `(key, plugin, measured, predicted)`), APE = `|pred/meas - 1|`
  - `validate.policy_stats(rows, params) -> dict` with `n, model_median_regret, model_max_regret, heuristic_median_regret, heuristic_max_regret`: for each (workload, cache_state, family) with ≥ 3 measured variants, the model picks the variant with the lowest predicted time among the measured ones; regret = measured(pick) / measured(best) − 1; the heuristic's pick is `flashdecoding` (dense) or `flashdecoding_paged` (paged)
  - `validate.SETS`: `V1` = uniform S1 (H_kv 8) rows whose `(B, L_kv)` is not in the training grid `B ∈ {1, 4, 16, 64}` × `L ∈ {512, 2048, 8192, 32768}`; `V5` = uniform S2 (H_kv 4) rows; `V6` = ragged rows whose key is in the odd half (`sum(ord) % 2 == 1`)
  - `validate.THRESHOLDS = {"V1": (0.15, 0.30), "V5": (0.225, 0.45), "V6": (0.225, 0.45)}` (median, p90 APE; V5/V6 = 1.5 × V1 per spec §3.3); policy: model median ≤ 0.05 and max ≤ 0.15
  - `validate.report(rows, params, set_name, cache_state) -> dict` and `validate.markdown(results: list[dict]) -> str`
  - CLI: `kernelscope model validate --results DIR... --machine M --params P [--params-uniform P2] --out report.md` — runs V1/V2/V5 on `--params` and V6 on `--params-uniform` when given (else on `--params`); prints and writes the markdown.

Note: the V1 training grid above must be the grid the fit used. Task 9 fits with `--train all` restricted to that grid for uniform rows. Add to `_cmd_model_fit` in this task: a flag `--uniform-train-grid` (default on) that keeps only uniform rows with `B in {1,4,16,64}` and `L_kv in {512,2048,8192,32768}` when fitting, and put the grid constants in `validate.TRAIN_B` / `validate.TRAIN_L` so both commands share them.

- [ ] **Step 1: Write the failing tests** — `tests/test_model_validate.py`:

```python
import pytest

from kernelscope.model.fit import prepare_rows
from kernelscope.model.validate import THRESHOLDS, TRAIN_B, TRAIN_L, error_stats, in_set, markdown, policy_stats
from tests.test_model_fit import _synthetic
from tests.test_model_predict import M, P


def test_perfect_model_has_zero_error_and_zero_regret():
    rows = prepare_rows(_synthetic(P), M)
    e = error_stats(rows, P)
    assert e["n"] == len(rows) and e["median_ape"] == pytest.approx(0.0, abs=1e-9)
    pol = policy_stats(rows, P)
    assert pol["model_max_regret"] == pytest.approx(0.0, abs=1e-9)
    assert pol["heuristic_max_regret"] >= 0.0


def test_sets_partition_the_rows_as_specified():
    rows = prepare_rows(_synthetic(P), M)
    v1 = [r for r in rows if in_set(r, "V1")]
    assert all(not r.workload.is_ragged and r.workload.H_kv == 8 for r in v1)
    assert all((r.workload.B, r.workload.L_kv) not in {(b, l) for b in TRAIN_B for l in TRAIN_L} for r in v1)
    v6 = [r for r in rows if in_set(r, "V6")]
    assert all(r.workload.is_ragged for r in v6)


def test_markdown_marks_pass_and_fail():
    md = markdown([{"set": "V1", "cache_state": "cold", "n": 10, "median_ape": 0.05, "p90_ape": 0.5,
                    "threshold": THRESHOLDS["V1"], "worst": []}])
    assert "FAIL" in md and "V1" in md
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `kernelscope/model/validate.py`:**

```python
"""Validation of the surrogate model against held-out measurements (spec §3.3)."""
from collections import defaultdict

import numpy as np

from kernelscope.model.fit import row_time_us

TRAIN_B = (1, 4, 16, 64)
TRAIN_L = (512, 2048, 8192, 32768)
THRESHOLDS = {"V1": (0.15, 0.30), "V5": (0.225, 0.45), "V6": (0.225, 0.45)}
POLICY = (0.05, 0.15)
HEURISTIC = {"dense": "flashdecoding", "paged": "flashdecoding_paged"}


def in_set(row, name: str) -> bool:
    w = row.workload
    if name == "V1":
        return not w.is_ragged and w.H_kv == 8 and not (w.B in TRAIN_B and w.L_kv in TRAIN_L)
    if name == "V5":
        return not w.is_ragged and w.H_kv == 4
    if name == "V6":
        return w.is_ragged and sum(map(ord, w.key())) % 2 == 1
    raise ValueError(name)


def error_stats(rows, params) -> dict:
    if not rows:
        return {"n": 0, "median_ape": float("nan"), "p90_ape": float("nan"), "worst": []}
    preds = [row_time_us(r, params.states[r.cache_state]) for r in rows]
    ape = np.array([abs(p / r.measured_us - 1) for p, r in zip(preds, rows)])
    worst = [(rows[i].workload.key(), rows[i].plugin, rows[i].measured_us, preds[i]) for i in np.argsort(-ape)[:5]]
    return {"n": len(rows), "median_ape": float(np.median(ape)), "p90_ape": float(np.percentile(ape, 90)), "worst": worst}


def policy_stats(rows, params) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r.workload.key(), r.cache_state, "paged" if r.plugin.endswith("_paged") else "dense")].append(r)
    model, heur = [], []
    for (_, state, family), rs in groups.items():
        if len(rs) < 3:
            continue
        best = min(r.measured_us for r in rs)
        pick = min(rs, key=lambda r: row_time_us(r, params.states[state]))
        model.append(pick.measured_us / best - 1)
        h = [r for r in rs if r.plugin == HEURISTIC[family]]
        if h:
            heur.append(h[0].measured_us / best - 1)
    f = lambda xs, fn: float(fn(xs)) if xs else float("nan")  # noqa: E731
    return {"n": len(model), "model_median_regret": f(model, np.median), "model_max_regret": f(model, np.max),
            "heuristic_median_regret": f(heur, np.median), "heuristic_max_regret": f(heur, np.max)}


def report(rows, params, set_name: str, cache_state: str) -> dict:
    rs = [r for r in rows if in_set(r, set_name) and r.cache_state == cache_state]
    return {"set": set_name, "cache_state": cache_state, "threshold": THRESHOLDS[set_name],
            **error_stats(rs, params), "policy": policy_stats(rs, params)}


def markdown(results: list) -> str:
    out = ["| set | cache | n | median APE | p90 APE | threshold (median / p90) | verdict |", "|---|---|---|---|---|---|---|"]
    for r in results:
        med, p90 = r["threshold"]
        ok = r["n"] > 0 and r["median_ape"] <= med and r["p90_ape"] <= p90
        out.append(f"| {r['set']} | {r['cache_state']} | {r['n']} | {r['median_ape']:.1%} | {r['p90_ape']:.1%} | "
                   f"{med:.0%} / {p90:.0%} | {'PASS' if ok else 'FAIL'} |")
    out += ["", "| set | cache | groups | model median / max regret | heuristic median / max regret | verdict (≤ 5 % / 15 %) |",
            "|---|---|---|---|---|---|"]
    for r in results:
        p = r.get("policy")
        if not p or not p["n"]:
            continue
        ok = p["model_median_regret"] <= POLICY[0] and p["model_max_regret"] <= POLICY[1]
        out.append(f"| {r['set']} | {r['cache_state']} | {p['n']} | {p['model_median_regret']:.1%} / {p['model_max_regret']:.1%} | "
                   f"{p['heuristic_median_regret']:.1%} / {p['heuristic_max_regret']:.1%} | {'PASS' if ok else 'FAIL'} |")
    for r in results:
        if r.get("worst"):
            out += ["", f"Worst cells, {r['set']} {r['cache_state']}:", ""]
            out += [f"- `{k}` {p}: measured {m:.1f} µs, predicted {q:.1f} µs" for k, p, m, q in r["worst"]]
    return "\n".join(out) + "\n"
```

CLI:
```python
def _cmd_model_validate(args):
    from kernelscope.model.fit import prepare_rows
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.validate import markdown, report
    m = MachineSpec.from_json(args.machine)
    rows = prepare_rows(_load_results(args.results), m)
    full = ModelParams.from_json(args.params)
    uni = ModelParams.from_json(args.params_uniform) if args.params_uniform else full
    states = sorted({r.cache_state for r in rows})
    results = [report(rows, full, s, c) for s in ("V1", "V5") for c in states]
    results += [report(rows, uni, "V6", c) for c in states]
    md = markdown([r for r in results if r["n"]])
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
```
parser: `p_val = msub.add_parser("validate", ...)` with `--results` (nargs +, required), `--machine` (default `machines/rtx4090.json`), `--params` (required), `--params-uniform`, `--out`; `set_defaults(func=_cmd_model_validate)`.
And in `_cmd_model_fit`: after the ragged filter, add
```python
    if args.uniform_train_grid:
        from kernelscope.model.validate import TRAIN_B, TRAIN_L
        rows = [r for r in rows if r.workload.is_ragged or (r.workload.B in TRAIN_B and r.workload.L_kv in TRAIN_L
                                                            and r.workload.H_kv == 8)]
```
with `p_fit.add_argument("--uniform-train-grid", action=argparse.BooleanOptionalAction, default=True)`. (Uniform S2 rows are then excluded from training, which is what V5 needs.)

- [ ] **Step 4: Run** → PASS; whole suite.
- [ ] **Step 5: Commit** — `git add kernelscope/model/validate.py tests/test_model_validate.py && git commit -m "Add model validation sets and report" -- kernelscope/model/validate.py kernelscope/cli.py tests/test_model_validate.py`

---

### Task 7: `model predict` and the what-if API

**Files:**
- Create: `kernelscope/model/whatif.py`
- Modify: `kernelscope/cli.py` (add `model predict`)
- Test: `tests/test_model_whatif.py`

**Interfaces:**
- Consumes: Tasks 1, 4.
- Produces:
  - `whatif.parse_scales(text: str) -> dict` (`"sm=0.5,dram=2"` → `{"sm": 0.5, "dram": 2.0}`; unknown keys → `ValueError`)
  - `whatif.table(w, family, machine, params, cache_state, scales: dict | None = None) -> list[dict]` — one dict per variant: `plugin, time_us, main_us, combine_us, kind, splits, ctas, slots_per_sm, limiter, bw_bound_fraction, heuristic_pick: bool` sorted by `time_us`; `heuristic_pick` marks the variant equal to what the library heuristic resolves to on that (possibly scaled) machine — i.e. the row whose `splits` equals the heuristic's and whose name is not `flashdecoding*` itself is not marked; mark exactly the `flashdecoding` / `flashdecoding_paged` row
  - CLI: `kernelscope model predict --workload KEY --params P [--machine M] [--family dense|paged] [--cache-state cold|warm] [--scale sm=0.5,...]` prints the table

- [ ] **Step 1: Failing tests** — `tests/test_model_whatif.py`:

```python
import time

import pytest

from kernelscope.model.whatif import parse_scales, table
from kernelscope.workload import Workload
from tests.test_model_predict import M, P

W = Workload(phase="decode", B=32, L_q=1, L_kv=32768, H_q=32, H_kv=8, d=128, kv_lens=[32768] * 2 + [1024] * 30)


def test_parse_scales():
    assert parse_scales("sm=0.5,dram=2") == {"sm": 0.5, "dram": 2.0}
    assert parse_scales("") == {}
    with pytest.raises(ValueError):
        parse_scales("clock=2")


def test_table_is_sorted_and_marks_the_heuristic():
    t = table(W, "dense", M, P, "cold")
    assert [r["time_us"] for r in t] == sorted(r["time_us"] for r in t)
    assert [r["plugin"] for r in t if r["heuristic_pick"]] == ["flashdecoding"]


def test_halving_sms_changes_what_the_heuristic_does():
    base = {r["plugin"]: r for r in table(Workload(phase="decode", B=1, L_q=1, L_kv=8192, H_q=32, H_kv=8, d=128),
                                          "dense", M, P, "cold")}
    half = {r["plugin"]: r for r in table(Workload(phase="decode", B=1, L_q=1, L_kv=8192, H_q=32, H_kv=8, d=128),
                                          "dense", M, P, "cold", {"sm": 0.5})}
    assert base["flashdecoding"]["splits"] == 32 and half["flashdecoding"]["splits"] == 16


def test_what_if_table_is_fast_enough_for_a_slider():
    t0 = time.perf_counter()
    table(W, "dense", M, P, "cold", {"dram": 2.0})
    assert time.perf_counter() - t0 < 1.0
```

- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement `kernelscope/model/whatif.py`:**

```python
"""Variant table on the measured machine or a scaled variant of it (the dashboard's what-if view)."""
from dataclasses import asdict

from kernelscope.model.predict import DENSE_VARIANTS, PAGED_VARIANTS, rank_variants

SCALES = ("sm", "dram", "l2", "smem")


def parse_scales(text: str) -> dict:
    out = {}
    for part in filter(None, (p.strip() for p in text.split(","))):
        k, v = part.split("=")
        if k not in SCALES:
            raise ValueError(f"unknown scale {k!r}; known: {', '.join(SCALES)}")
        out[k] = float(v)
    return out


def table(w, family: str, machine, params, cache_state: str, scales: dict | None = None) -> list:
    m = machine.scaled(**(scales or {}))
    plugins = DENSE_VARIANTS if family == "dense" else PAGED_VARIANTS
    heuristic = "flashdecoding" if family == "dense" else "flashdecoding_paged"
    return [{**asdict(p), "heuristic_pick": p.plugin == heuristic} for p in rank_variants(w, plugins, m, params, cache_state)]
```
CLI:
```python
def _cmd_model_predict(args):
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.whatif import parse_scales, table
    rows = table(Workload.from_key(args.workload), args.family, MachineSpec.from_json(args.machine),
                 ModelParams.from_json(args.params), args.cache_state, parse_scales(args.scale))
    for r in rows:
        print(f"{r['plugin']:22s} {r['time_us']:10.1f} us  kind={r['kind']:11s} splits={r['splits']:3d} ctas={r['ctas']:6d} "
              f"slots/SM={r['slots_per_sm']} ({r['limiter']}) bw-bound={r['bw_bound_fraction']:.0%}"
              + ("  <- library heuristic" if r["heuristic_pick"] else ""))
```
parser: `p_pred = msub.add_parser("predict", ...)`: `--workload` (required), `--params` (required), `--machine` (default `machines/rtx4090.json`), `--family` (choices dense/paged, default dense), `--cache-state` (choices cold/warm, default cold), `--scale` (default `""`).

- [ ] **Step 4: Run → PASS; whole suite. Step 5: Commit** — `git add kernelscope/model/whatif.py tests/test_model_whatif.py && git commit -m "Add what-if variant tables and model predict" -- kernelscope/model/whatif.py kernelscope/cli.py tests/test_model_whatif.py`

---

### Task 8: Real SM-count what-if measurements (V3)

**Files:**
- Modify: `kernelscope/bench/sm_blocker.py` (`time_blocked` gains `before=None`)
- Create: `kernelscope/model/blocked.py`
- Modify: `kernelscope/cli.py` (add `model blocked`)
- Test: `tests/test_model_blocked.py` (CPU part + one GPU test)

**Interfaces:**
- Consumes: `SMBlocker` (Phase 0), `IterationHooks` (Phase 0), `REGISTRY` plugins, `predict` (Task 4).
- Produces:
  - `SMBlocker.time_blocked(fn, n_sms, iters=10, warmup=3, before=None)` — `before()` runs before launching the blocker for every call (outside the timed region); used for L2 flushing
  - `blocked.CELLS`: `["decode_B1_Lq1_Lkv32768_Hq32_Hkv8_d128_float16_causal", "decode_B4_Lq1_Lkv16384_Hq32_Hkv8_d128_float16_causal", "decode_B16_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal", "decode_B64_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal", "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"]`, `blocked.VARIANTS = ["fa2", "fd_s8", "fd_s32"]`, `blocked.BLOCKED = [0, 32, 64, 96]`
  - `blocked.measure(out_dir, device="cuda", log=print) -> pd.DataFrame` — cold (flush via `IterationHooks("cuda","cold").between()` + sync as `before`), CUDA-event µs per call, rows `{workload_key, kernel, backend: "blocked", metric: "event_us", unit: "us", value, launch_idx: 0, note: None}` with extra `blocked_sms`, written through `ResultStore(out_dir)`
  - `blocked.compare(df, machine, params) -> pd.DataFrame` with `workload_key, kernel, blocked_sms, measured_ratio, predicted_ratio, ratio_error` where ratios are `t(N)/t(0)` and the prediction uses `machine.scaled(sm=(n_sm - N)/n_sm)`; `ratio_error = predicted_ratio/measured_ratio - 1`
  - CLI: `kernelscope model blocked --out DIR [--params P --machine M]` measures, then prints `compare` and the median |ratio_error| (V3 threshold 0.20)

- [ ] **Step 1: CPU test for `compare`** — `tests/test_model_blocked.py`:
```python
import pandas as pd
import pytest

from kernelscope.model.blocked import compare
from tests.test_model_predict import M, P

KEY = "decode_B1_Lq1_Lkv32768_Hq32_Hkv8_d128_float16_causal"


def test_compare_turns_measurements_into_ratios_against_the_unblocked_run():
    df = pd.DataFrame([{"workload_key": KEY, "kernel": "fd_s32", "backend": "blocked", "metric": "event_us",
                        "unit": "us", "value": v, "launch_idx": 0, "note": None, "blocked_sms": n}
                       for n, v in ((0, 100.0), (64, 180.0))])
    c = compare(df, M, P)
    row = c[c.blocked_sms == 64].iloc[0]
    assert row.measured_ratio == pytest.approx(1.8)
    assert row.predicted_ratio > 1.0
    assert row.ratio_error == pytest.approx(row.predicted_ratio / 1.8 - 1)
    assert (c.blocked_sms == 0).sum() == 0
```
- [ ] **Step 2: Implement.** In `sm_blocker.py` `time_blocked`: add `before=None`; in the timed loop call `before()` (if given) before `self.block(...)`; the two unblocked calibration calls also call `before()` first. `kernelscope/model/blocked.py`:
```python
"""V3: real SM-count what-if. Measure variants with N SMs occupied by the SM blocker and compare
the measured slowdown t(N)/t(0) with the model's prediction on a machine with n_sm - N SMs (spec §3.3).
Ratios cancel the CUDA-event launch overhead that kernel-only predictions do not include."""
import pandas as pd

from kernelscope.model.predict import predict
from kernelscope.workload import Workload

CELLS = ["decode_B1_Lq1_Lkv32768_Hq32_Hkv8_d128_float16_causal",
         "decode_B4_Lq1_Lkv16384_Hq32_Hkv8_d128_float16_causal",
         "decode_B16_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal",
         "decode_B64_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal",
         "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"]
VARIANTS = ["fa2", "fd_s8", "fd_s32"]
BLOCKED = [0, 32, 64, 96]


def measure(out_dir, device="cuda", log=print) -> pd.DataFrame:
    import torch
    from kernelscope.backends.realhw.cache import IterationHooks
    from kernelscope.bench.sm_blocker import SMBlocker
    from kernelscope.plugins.builtin import REGISTRY
    from kernelscope.results.store import ResultStore
    blocker = SMBlocker(device)
    hooks = IterationHooks(device, "cold")

    def before():
        hooks.between()
        torch.cuda.synchronize(device)

    rows = []
    for key in CELLS:
        w = Workload.from_key(key)
        for name in VARIANTS:
            plugin = REGISTRY.get(name, device=device)
            inputs = plugin.build_inputs(w)
            fn = lambda: plugin.run(inputs)  # noqa: E731
            for n in BLOCKED:
                us = blocker.time_blocked(fn, n, before=before)
                rows.append({"workload_key": key, "kernel": name, "backend": "blocked", "metric": "event_us",
                             "unit": "us", "value": us, "launch_idx": 0, "note": None, "blocked_sms": n})
                if log:
                    log(f"{name:8s} {key[:40]:40s} blocked={n:3d} {us:9.1f} us")
            del inputs
            torch.cuda.empty_cache()
    ResultStore(out_dir).write(rows, tag="blocked", extra={"runner": "model-blocked"})
    return pd.DataFrame(rows)


def compare(df, machine, params) -> pd.DataFrame:
    out = []
    for (key, kernel), g in df[df.metric == "event_us"].groupby(["workload_key", "kernel"]):
        base = g[g.blocked_sms == 0].value.median()
        w = Workload.from_key(key)
        p0 = predict(kernel, w, machine, params, "cold").time_us
        for n, gn in g[g.blocked_sms > 0].groupby("blocked_sms"):
            meas = gn.value.median() / base
            pred = predict(kernel, w, machine.scaled(sm=(machine.n_sm - n) / machine.n_sm), params, "cold").time_us / p0
            out.append({"workload_key": key, "kernel": kernel, "blocked_sms": int(n), "measured_ratio": meas,
                        "predicted_ratio": pred, "ratio_error": pred / meas - 1})
    return pd.DataFrame(out)
```
CLI: `p_blk = msub.add_parser("blocked", ...)`: `--out` (required), `--params`, `--machine` (default `machines/rtx4090.json`), `--device` (default `cuda`); `_cmd_model_blocked` runs `measure`, and when `--params` is given prints `compare(...)` and `median |ratio_error|`.
- [ ] **Step 3: GPU test** (append to `tests/test_model_blocked.py`):
```python
@pytest.mark.gpu
def test_time_blocked_runs_the_before_hook_every_call():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernelscope.bench.sm_blocker import SMBlocker
    calls = []
    a = torch.randn(1024, 1024, device="cuda")
    SMBlocker().time_blocked(lambda: a @ a, 32, iters=3, warmup=1, before=lambda: calls.append(1))
    assert len(calls) == 2 + 1 + 3          # two unblocked calibration calls, warm-up, timed
```
- [ ] **Step 4: Run tests (hygiene first for GPU) → PASS; whole suite. Step 5: Commit** — `git add kernelscope/model/blocked.py tests/test_model_blocked.py && git commit -m "Add real SM-count what-if measurement (V3)" -- kernelscope/bench/sm_blocker.py kernelscope/model/blocked.py kernelscope/cli.py tests/test_model_blocked.py`

---

### Task 9: Fit, validate, and record

**Depends on:** Tasks 1–8. CPU-heavy (fits) plus a short GPU run (V3).

**Files:**
- Create: `models/rtx4090.json`, `models/rtx4090_uniform.json` (fitted), `docs/plan/2026-09-19-p1-model-validation.md` (results note)
- Modify: `docs/STATUS.md` (append one dated entry), `README.md` (model commands in the quick start)

- [ ] **Step 1: Fit** (expect roughly 1 h of CPU time for the `all` fit; the uniform fit is shorter). `R=/home/skkai/AI_Accelerator/kernelscope/results/hw_4090`; `ALL="$R/uniform_s1_dense $R/uniform_s1_paged $R/uniform_s2_dense $R/uniform_s2_paged $R/ragged_s1_dense $R/ragged_s1_paged $R/ragged_s2_paged"`. Each Bash call is limited to 10 minutes, so run fits in the background with a log and wait for completion by polling the log:
```bash
nohup $PY -m kernelscope.cli model fit --results $ALL --out models/rtx4090.json > /tmp/claude-1000/-home-skkai-AI-Accelerator/84108947-5942-4ee4-942f-c308fb58a2e7/scratchpad/fit_all.log 2>&1 &
nohup $PY -m kernelscope.cli model fit --results $ALL --train uniform --out models/rtx4090_uniform.json > /tmp/claude-1000/-home-skkai-AI-Accelerator/84108947-5942-4ee4-942f-c308fb58a2e7/scratchpad/fit_uniform.log 2>&1 &
```
(two fits in parallel are fine: CPU only.) Record each fit's wall time and its stage log lines.
- [ ] **Step 2: Validate.** `$PY -m kernelscope.cli model validate --results $ALL --params models/rtx4090.json --params-uniform models/rtx4090_uniform.json --out docs/plan/2026-09-19-p1-model-validation.md` (the command writes the tables; you add prose around them in step 4).
- [ ] **Step 3: V3** (GPU hygiene first): `$PY -m kernelscope.cli model blocked --out $R/blocked_s1 --params models/rtx4090.json`. Record the compare table and the median |ratio_error|.
- [ ] **Step 4: Results note** — extend `docs/plan/2026-09-19-p1-model-validation.md` with: fitted parameters per state and kind (from `models/rtx4090.json`), fit wall times, the V1/V5/V6 and policy tables with PASS/FAIL, the V3 table and verdict (threshold 20 %), the five worst cells per set with a one-line hypothesis each, and an explicit list of assumptions still untested (split-kernel gamma borrowed from the non-split kernel; paged combine equal to dense). Do not tune anything to pass; report failures as observed.
- [ ] **Step 5: Docs** — README quick start: add `model fit`, `model validate`, `model predict --scale`, `model blocked`. STATUS: append `## 2026-09-19 — design-1-3: surrogate model` (≤ 10 lines, numbers and a link to the note).
- [ ] **Step 6: Commit** — `git add models/rtx4090.json models/rtx4090_uniform.json docs/plan/2026-09-19-p1-model-validation.md && git commit -m "Fit and validate the surrogate model on the RTX 4090 campaign" -- models/rtx4090.json models/rtx4090_uniform.json docs/plan/2026-09-19-p1-model-validation.md README.md docs/STATUS.md`
