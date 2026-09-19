# Phase 0 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the real-hardware track for the RTX 4090 and make it fast and cache-aware, so it can produce the measurement grid that the surrogate performance model (plan 2) and the kernel dispatcher (plan 3) are built on.

**Architecture:** Twelve tasks. Device-correct occupancy; ragged (mixed-length) decode workloads as first-class workload keys; paged-KV and fixed-split flash-attn plugin variants; a cold/warm L2 cache-state mode whose boundary kernels also make torch.profiler output robust; an in-process batch runner (`kernelscope bench`) that replaces three subprocesses per cell; cache-aware reporting; a dispatch-table analysis; a measured machine spec; a CUDA extension for SM blocking and block placement; and finally the measurement campaign itself.

**Tech Stack:** Python 3.11, torch 2.8.0+cu128, flash-attn 2.8.3.post1, Triton (bundled with torch), pandas/pyarrow, pytest. CUDA C++ via `torch.utils.cpp_extension.load_inline` with the CUDA 12.9 toolchain of the `accelsim-build` conda env.

**Spec:** `docs/plan/2026-09-19-design-surrogate-dispatcher.md` (Korean). This plan implements its §2 (items P0-1 … P0-12). Read §1 (measured facts F1–F20) before starting any task: the facts explain *why* each change exists.

## Global Constraints

- **Worktree:** all code work happens in `/home/skkai/AI_Accelerator/kernelscope-design` on branch `design-1-3`. Never edit files under `/home/skkai/AI_Accelerator/kernelscope` (another agent, Codex, works there on branch `sim-track-4090`). The only exception: Task 12 writes measurement output to `/home/skkai/AI_Accelerator/kernelscope/results/hw_4090/` (git-ignored, shared with Codex on purpose).
- **Python:** always `PY=/home/skkai/miniforge3/envs/gradkernel/bin/python` by absolute path. CPU tests: `$PY -m pytest -q -m "not gpu"`. GPU tests: `$PY -m pytest -q -m gpu <file>`.
- **Do not install, upgrade or remove packages** in any conda env.
- **Codex-owned, never edit:** `kernelscope/backends/accelsim/**`, the `_cmd_simsweep` function and the `p_sim` parser block in `kernelscope/cli.py`, `tests/test_accelsim_*.py`, `tests/fixtures/SM*`, `docs/setup/**`, `env/setup_accelsim_4090.sh`.
- **Result schema is fixed:** rows `{workload_key, kernel, backend, metric, unit, value, launch_idx, note}`; per-run context goes in extra columns through `ResultStore.write(rows, tag=..., extra={...})`. New extra column introduced by this plan: `cache_state` ∈ {`warm`, `cold`}.
- **Keys of uniform workloads must not change** (existing results and the sim track depend on them).
- **GPU hygiene before any GPU test or measurement:** run `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv` and `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`. Only the desktop viewer `rerun` (~440 MiB, 0 % util) is acceptable. If anything else is running, wait (poll every 60 s up to 20 min), then report BLOCKED. Never kill processes.
- **CUDA builds:** `CUDA_HOME=/home/skkai/miniforge3/envs/accelsim-build`, host compiler `/home/skkai/miniforge3/envs/accelsim-build/bin/x86_64-conda-linux-gnu-g++`. Never use `/usr/bin/nvcc` (CUDA 10.1, cannot target sm_89).
- **Commits:** one or more commits per task, committing only the task's own paths with the path-limited form `git commit -m "<msg>" -- <path> <path> ...` (other agents may have staged files concurrently). If git reports `index.lock`, wait 5 s and retry. End every commit message with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- **Test hygiene:** run your task's test files first, then the whole CPU suite. If a failure is in a file your task did not touch, report it instead of fixing it.
- **Hardware facts to rely on (RTX 4090, measured):** CC 8.9, 128 SMs, 1536 threads/SM, 65536 regs/SM, 102400 B shared memory/SM, 101376 B opt-in per block, 1024 B reserved per block, 24 CTAs/SM max, L2 75497472 B (72 MiB).

## Execution order

Waves (tasks in one wave touch disjoint files and may run in parallel):

1. Task 1, Task 2, Task 11
2. Task 3, Task 5
3. Task 4, Task 6
4. Task 7 → Task 8 → Task 9 → Task 10 (sequential: all edit `kernelscope/cli.py`)
5. Task 12

---

### Task 1: Device-correct occupancy limits

The occupancy estimate hardcodes A100's 32 CTAs/SM and ignores register-allocation granularity and the 1 KiB per-block shared-memory reservation. On the 4090 this matters: the split-KV kernel (80 KiB smem) fits once per SM, the fa2 kernel twice (spec F1).

**Files:**
- Modify: `kernelscope/backends/realhw/kprofile.py:14-35` (props) and `:78-104` (`occupancy_estimate`)
- Test: `tests/test_kprofile.py`

**Interfaces:**
- Produces:
  - `MAX_BLOCKS_PER_SM: dict[tuple[int, int], int]`, `SMEM_PER_SM: dict[tuple[int, int], int]`, `REG_ALLOC_UNIT = 256`
  - `A100_PROPS`, `RTX4090_PROPS: dict` with exactly the keys `name, cc, num_sms, max_threads_per_sm, regs_per_sm, smem_per_sm, max_blocks_per_sm, reserved_smem_per_block, warp_size, max_warps_per_sm, l2_bytes`
  - `props_from_torch(device="cuda") -> dict` (same keys; raises if CUDA is unavailable — no silent A100 fallback)
  - `blocks_per_sm_limit(threads: int, regs: int | None, smem_bytes: int | None, props: dict) -> tuple[int, str]` — `(limit, limiter)`, limiter ∈ `{"threads", "blocks", "regs", "smem"}`, ties resolved in that order
  - `occupancy_estimate(grid, block, regs, smem_bytes, props)` — unchanged keys, all numeric

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kprofile.py` (and add `from types import SimpleNamespace` and the new names to the imports):

```python
from types import SimpleNamespace

from kernelscope.backends.realhw.kprofile import RTX4090_PROPS, blocks_per_sm_limit, props_from_torch


def test_splitkv_kernel_fits_once_per_sm_on_ada_because_of_shared_memory():
    # flash_fwd_splitkv_kernel, d=128, measured on the 4090: 128 threads, 244 regs, 80 KiB smem
    assert blocks_per_sm_limit(128, 244, 81920, RTX4090_PROPS) == (1, "smem")


def test_splitkv_kernel_fits_twice_per_sm_on_a100():
    assert blocks_per_sm_limit(128, 244, 81920, A100_PROPS)[0] == 2


def test_fa2_decode_kernel_is_register_limited_to_two_per_sm_on_ada():
    # 255 regs * 32 lanes = 8160 -> 8192 per warp -> 8 warps/SM -> 2 blocks of 4 warps
    assert blocks_per_sm_limit(128, 255, 49152, RTX4090_PROPS) == (2, "regs")


def test_combine_kernel_limit_matches_the_profiler_on_ada():
    assert blocks_per_sm_limit(128, 52, 160, RTX4090_PROPS) == (9, "regs")


def test_register_allocation_rounds_up_to_256_per_warp():
    # 44 regs * 32 = 1408 -> 1536 per warp -> 42 warps -> 21 blocks of 2 warps (23 without rounding)
    assert blocks_per_sm_limit(64, 44, 0, RTX4090_PROPS) == (21, "regs")


def test_cta_cap_limits_tiny_blocks():
    assert blocks_per_sm_limit(32, 16, 0, RTX4090_PROPS) == (24, "blocks")


def _fake_props(**drop):
    p = dict(name="NVIDIA GeForce RTX 4090", major=8, minor=9, multi_processor_count=128,
             max_threads_per_multi_processor=1536, regs_per_multiprocessor=65536,
             shared_memory_per_multiprocessor=102400, warp_size=32, L2_cache_size=75497472)
    for k in drop:
        p.pop(k)
    return SimpleNamespace(**p)


def test_props_from_torch_reads_the_device_and_takes_the_cta_cap_from_the_compute_capability(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: _fake_props())
    assert props_from_torch("cuda") == RTX4090_PROPS


def test_props_from_torch_falls_back_to_the_cc_table_for_missing_fields(monkeypatch):
    import torch
    fake = _fake_props(regs_per_multiprocessor=1, shared_memory_per_multiprocessor=1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: fake)
    p = props_from_torch("cuda")
    assert p["smem_per_sm"] == 102400
    assert p["regs_per_sm"] == 65536
```

- [ ] **Step 2: Run to verify they fail**

Run: `$PY -m pytest tests/test_kprofile.py -q`
Expected: FAIL — `ImportError: cannot import name 'RTX4090_PROPS'`.

- [ ] **Step 3: Implement** — replace `kprofile.py` lines 14-35 with:

```python
# Per-compute-capability limits torch does not expose (CUDA C Programming Guide,
# "Technical Specifications per Compute Capability").
MAX_BLOCKS_PER_SM = {(7, 0): 32, (7, 5): 16, (8, 0): 32, (8, 6): 16, (8, 7): 16, (8, 9): 24, (9, 0): 32}
SMEM_PER_SM = {(7, 0): 98304, (7, 5): 65536, (8, 0): 167936, (8, 6): 102400, (8, 7): 167936,
               (8, 9): 102400, (9, 0): 233472}
REG_ALLOC_UNIT = 256          # registers are allocated per warp in units of 256 (CC 7.x-9.x)

A100_PROPS = {
    "name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "num_sms": 108, "max_threads_per_sm": 2048,
    "regs_per_sm": 65536, "smem_per_sm": 167936, "max_blocks_per_sm": 32,
    "reserved_smem_per_block": 1024, "warp_size": 32, "max_warps_per_sm": 64, "l2_bytes": 41943040,
}
RTX4090_PROPS = {
    "name": "NVIDIA GeForce RTX 4090", "cc": "8.9", "num_sms": 128, "max_threads_per_sm": 1536,
    "regs_per_sm": 65536, "smem_per_sm": 102400, "max_blocks_per_sm": 24,
    "reserved_smem_per_block": 1024, "warp_size": 32, "max_warps_per_sm": 48, "l2_bytes": 75497472,
}


def props_from_torch(device="cuda") -> dict:
    """Occupancy-relevant limits of the current device. Raises if CUDA is unavailable."""
    import torch
    p = torch.cuda.get_device_properties(device)
    cc = (p.major, p.minor)
    return {
        "name": p.name,
        "cc": f"{p.major}.{p.minor}",
        "num_sms": p.multi_processor_count,
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", 65536),
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", SMEM_PER_SM.get(cc, 65536)),
        "max_blocks_per_sm": MAX_BLOCKS_PER_SM.get(cc, 16),
        "reserved_smem_per_block": 1024 if p.major >= 8 else 0,
        "warp_size": p.warp_size,
        "max_warps_per_sm": p.max_threads_per_multi_processor // p.warp_size,
        "l2_bytes": p.L2_cache_size,
    }
```

and replace `occupancy_estimate` (lines 78-104) with:

```python
def blocks_per_sm_limit(threads, regs, smem_bytes, props) -> tuple[int, str]:
    """Resident CTAs per SM and the resource that limits it (CUDA occupancy-calculator rules:
    per-warp register allocation in units of 256, 1 KiB shared memory reserved per block)."""
    warp = props["warp_size"]
    warps_per_block = math.ceil(threads / warp)
    by = {"threads": props["max_threads_per_sm"] // threads, "blocks": props["max_blocks_per_sm"]}
    if regs:
        per_warp = math.ceil(regs * warp / REG_ALLOC_UNIT) * REG_ALLOC_UNIT
        by["regs"] = (props["regs_per_sm"] // per_warp) // warps_per_block
    if smem_bytes:
        by["smem"] = props["smem_per_sm"] // (smem_bytes + props.get("reserved_smem_per_block", 0))
    limiter = min(by, key=by.get)
    return int(by[limiter]), limiter


def occupancy_estimate(grid, block, regs, smem_bytes, props=A100_PROPS) -> dict:
    """First-wave occupancy from launch geometry + per-thread resources."""
    blocks = math.prod(grid)
    threads = math.prod(block)
    warps_per_block = math.ceil(threads / props["warp_size"])
    limit, _ = blocks_per_sm_limit(threads, regs, smem_bytes, props)
    num_sms = props["num_sms"]
    sm_coverage = min(1.0, blocks / num_sms)
    blocks_per_active_sm = min(limit, math.ceil(blocks / num_sms))
    warps_per_active_sm = blocks_per_active_sm * warps_per_block
    warps_per_sm_device = min(blocks, limit * num_sms) * warps_per_block / num_sms
    return {
        "blocks": blocks, "threads_per_block": threads, "warps_per_block": warps_per_block,
        "blocks_per_sm_limit": limit, "sm_coverage": sm_coverage,
        "warps_per_active_sm": warps_per_active_sm,
        "occupancy_active_sm": warps_per_active_sm / props["max_warps_per_sm"],
        "warps_per_sm_device": warps_per_sm_device,
        "occupancy_device": warps_per_sm_device / props["max_warps_per_sm"],
    }
```

- [ ] **Step 4: Run tests** — `$PY -m pytest tests/test_kprofile.py -q` → all PASS (the two pre-existing A100 occupancy tests must still pass unchanged).
- [ ] **Step 5: Verify on the real device** (GPU hygiene first): `$PY -c "from kernelscope.backends.realhw.kprofile import props_from_torch, RTX4090_PROPS as R; p=props_from_torch('cuda'); print(p); assert p == R, {k:(p[k],R[k]) for k in R if p[k]!=R[k]}"` → prints the dict, no assertion error. If torch reports different values, stop and report them (do not edit `RTX4090_PROPS` to match silently).
- [ ] **Step 6: Full CPU suite, then commit** — `$PY -m pytest -q -m "not gpu"`; `git commit -m "Take occupancy limits from the device and compute capability" -- kernelscope/backends/realhw/kprofile.py tests/test_kprofile.py`

---

### Task 2: Ragged decode workloads and underscore dtypes

Continuous batching mixes long and short requests; the heuristic's worst failures happen exactly there (spec F14–F16). A workload needs per-sequence KV lengths, encoded compactly in its key.

**Files:**
- Modify: `kernelscope/workload.py` (whole file)
- Test: `tests/test_workload.py`

**Interfaces:**
- Produces:
  - `Workload(..., kv_lens: tuple[int, ...] | None = None)` — decode with `L_q == 1` only; `len(kv_lens) == B`; `max(kv_lens) == L_kv`; an all-equal tuple is canonicalised to `None`
  - `Workload.is_ragged: bool` (property), `Workload.lens() -> tuple[int, ...]` (per-sequence lengths, uniform or not)
  - key format: uniform keys unchanged; ragged keys put the run-length lens spec in the `Lkv` field, e.g. `decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal`
  - `format_lens(lens) -> str`, `parse_lens(s: str) -> tuple[int, ...]`, `ragged_lens(B, n_long, L_long, L_short) -> str`
  - `expand_grid` accepts `lens: <spec or list of specs>` and `ragged: {B: [...], n_long: [...], L_long: [...], L_short: [...]}` (combinations with `n_long >= B` are skipped)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_workload.py` (extend the import to `from kernelscope.workload import Workload, expand_grid, format_lens, parse_lens, ragged_lens`):

```python
RAGGED = Workload(phase="decode", B=32, L_q=1, L_kv=32768, H_q=32, H_kv=8, d=128,
                  kv_lens=[32768] * 2 + [1024] * 30)


def test_ragged_key_compresses_runs_of_equal_lengths():
    assert RAGGED.key() == "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"


def test_ragged_key_round_trips():
    assert Workload.from_key(RAGGED.key()) == RAGGED
    assert Workload.from_key(RAGGED.key()).kv_lens == (32768, 32768) + (1024,) * 30


def test_uniform_kv_lens_canonicalise_to_the_uniform_key():
    w = Workload(phase="decode", B=4, L_q=1, L_kv=512, H_q=8, H_kv=2, d=64, kv_lens=[512] * 4)
    assert w.kv_lens is None
    assert not w.is_ragged
    assert w.key() == "decode_B4_Lq1_Lkv512_Hq8_Hkv2_d64_float16_causal"


def test_lens_gives_per_sequence_lengths():
    assert RAGGED.lens()[:3] == (32768, 32768, 1024)
    u = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16)
    assert u.lens() == (64, 64, 64)


@pytest.mark.parametrize("kw, msg", [
    (dict(phase="prefill", L_q=16), "decode"),
    (dict(kv_lens=[100, 50]), "B=3"),
    (dict(kv_lens=[99, 50, 7]), "max"),
    (dict(kv_lens=[100, 0, 100]), "positive"),
])
def test_rejects_invalid_ragged_workloads(kw, msg):
    base = dict(phase="decode", B=3, L_q=1, L_kv=100, H_q=8, H_kv=2, d=16, kv_lens=[100, 50, 7])
    base.update(kw)
    with pytest.raises(ValueError, match=msg):
        Workload(**base)


def test_dtype_with_underscores_round_trips():
    for causal in (True, False):
        w = Workload(phase="decode", B=1, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16,
                     dtype="float8_e4m3fn", causal=causal)
        assert Workload.from_key(w.key()) == w


def test_format_and_parse_lens():
    assert format_lens([32768, 1024, 1024]) == "32768+1024x2"
    assert parse_lens("32768+1024x2") == (32768, 1024, 1024)
    assert parse_lens(format_lens([5, 5, 7, 5])) == (5, 5, 7, 5)
    with pytest.raises(ValueError, match="lens"):
        parse_lens("12x")


def test_ragged_lens_puts_long_sequences_first():
    assert ragged_lens(B=4, n_long=1, L_long=8192, L_short=512) == "8192+512x3"


def test_expand_grid_lens_sets_B_and_L_kv():
    ws = expand_grid({"phase": "decode", "lens": ["8192+512x3", "1024x2"], "H_q": 8, "H_kv": 2, "d": 16})
    assert [(w.B, w.L_kv, w.is_ragged) for w in ws] == [(4, 8192, True), (2, 1024, False)]


def test_expand_grid_ragged_generator_skips_impossible_combinations():
    ws = expand_grid({"phase": "decode", "H_q": 8, "H_kv": 2, "d": 16,
                      "ragged": {"B": [2, 16], "n_long": [1, 2], "L_long": 4096, "L_short": 256}})
    assert sorted(w.key().split("_")[3] for w in ws) == [
        "Lkv4096+256", "Lkv4096+256x15", "Lkv4096x2+256x14"]


def test_expand_grid_rejects_lens_together_with_B():
    with pytest.raises(ValueError, match="lens"):
        expand_grid({"phase": "decode", "lens": "64x2", "B": 2, "H_q": 8, "H_kv": 2, "d": 16})
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_workload.py -q` → FAIL (`ImportError: cannot import name 'format_lens'`).

- [ ] **Step 3: Implement** — replace `kernelscope/workload.py` with:

```python
"""Logical attention workload — layout-agnostic; each plugin materializes its own tensors.

A decode workload may be *ragged*: ``kv_lens`` gives each sequence's live KV length, the way
continuous batching mixes long and short requests. ``L_kv`` is then the longest length (the
capacity a dense cache needs). Uniform workloads keep ``kv_lens=None``, so their keys are
unchanged; a ragged key puts a run-length spec in the ``Lkv`` field: ``Lkv32768x2+1024x30``.
"""
import re
from dataclasses import dataclass
from itertools import product

PHASES = ("prefill", "decode")


def format_lens(lens) -> str:
    """(32768, 32768, 1024, 1024) -> '32768x2+1024x2'; runs keep their order, 'x1' is omitted."""
    lens, parts, i = list(lens), [], 0
    while i < len(lens):
        j = i
        while j < len(lens) and lens[j] == lens[i]:
            j += 1
        parts.append(f"{lens[i]}x{j - i}" if j - i > 1 else f"{lens[i]}")
        i = j
    return "+".join(parts)


def parse_lens(s: str) -> tuple[int, ...]:
    out = []
    for part in s.split("+"):
        m = re.fullmatch(r"(\d+)(?:x(\d+))?", part)
        if not m:
            raise ValueError(f"malformed lens spec: {s!r}")
        out += [int(m.group(1))] * int(m.group(2) or 1)
    return tuple(out)


def ragged_lens(B: int, n_long: int, L_long: int, L_short: int) -> str:
    """Lens spec with ``n_long`` sequences of ``L_long`` first, then ``B - n_long`` of ``L_short``."""
    if not 0 < n_long < B:
        raise ValueError(f"need 0 < n_long < B, got n_long={n_long}, B={B}")
    return format_lens([L_long] * n_long + [L_short] * (B - n_long))


@dataclass(frozen=True)
class Workload:
    phase: str
    B: int
    L_q: int
    L_kv: int
    H_q: int
    H_kv: int
    d: int
    dtype: str = "float16"
    causal: bool = True
    kv_lens: tuple | None = None

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        for name in ("B", "L_q", "L_kv", "H_q", "H_kv", "d"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.H_q % self.H_kv:
            raise ValueError(f"H_q ({self.H_q}) must be a multiple of H_kv ({self.H_kv})")
        if self.kv_lens is not None:
            lens = tuple(int(x) for x in self.kv_lens)
            if self.phase != "decode" or self.L_q != 1:
                raise ValueError("kv_lens (ragged batch) is supported for decode with L_q=1 only")
            if len(lens) != self.B:
                raise ValueError(f"kv_lens has {len(lens)} entries but B={self.B}")
            if min(lens) <= 0:
                raise ValueError("kv_lens entries must be positive")
            if max(lens) != self.L_kv:
                raise ValueError(f"L_kv ({self.L_kv}) must equal max(kv_lens) ({max(lens)})")
            object.__setattr__(self, "kv_lens", None if len(set(lens)) == 1 else lens)

    @property
    def is_ragged(self) -> bool:
        return self.kv_lens is not None

    def lens(self) -> tuple[int, ...]:
        return self.kv_lens if self.kv_lens is not None else (self.L_kv,) * self.B

    def key(self) -> str:
        lkv = format_lens(self.kv_lens) if self.is_ragged else str(self.L_kv)
        return (
            f"{self.phase}_B{self.B}_Lq{self.L_q}_Lkv{lkv}"
            f"_Hq{self.H_q}_Hkv{self.H_kv}_d{self.d}_{self.dtype}"
            f"_{'causal' if self.causal else 'noncausal'}"
        )

    @classmethod
    def from_key(cls, key: str) -> "Workload":
        m = _KEY_RE.fullmatch(key)
        if not m:
            raise ValueError(f"malformed workload key: {key!r}")
        g = m.groupdict()
        lkv = g["L_kv"]
        lens = parse_lens(lkv) if ("x" in lkv or "+" in lkv) else None
        return cls(
            phase=g["phase"], B=int(g["B"]), L_q=int(g["L_q"]),
            L_kv=max(lens) if lens else int(lkv),
            H_q=int(g["H_q"]), H_kv=int(g["H_kv"]), d=int(g["d"]),
            dtype=g["dtype"], causal=g["causal"] == "causal", kv_lens=lens,
        )


_KEY_RE = re.compile(
    r"(?P<phase>prefill|decode)_B(?P<B>\d+)_Lq(?P<L_q>\d+)"
    r"_Lkv(?P<L_kv>\d+(?:x\d+)?(?:\+\d+(?:x\d+)?)*)"
    r"_Hq(?P<H_q>\d+)_Hkv(?P<H_kv>\d+)_d(?P<d>\d+)_(?P<dtype>[a-z0-9_]+?)_(?P<causal>causal|noncausal)"
)


def expand_grid(spec: dict) -> list[Workload]:
    """Cartesian product over list-valued fields, in declaration order.

    ``L`` is shorthand: it sets ``L_kv`` and derives ``L_q`` from the phase (1 for decode,
    ``L`` for prefill) unless those are given explicitly. ``lens`` (a lens spec or a list of
    them) sets ``B``, ``L_kv`` and ``kv_lens``. ``ragged`` is a generator
    ``{B, n_long, L_long, L_short}`` whose product becomes a ``lens`` list.
    """
    spec = dict(spec)
    if "ragged" in spec:
        if "lens" in spec:
            raise ValueError("give either ragged or lens, not both")
        r = {k: (list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in spec.pop("ragged").items()}
        missing = {"B", "n_long", "L_long", "L_short"} - set(r)
        if missing:
            raise ValueError(f"ragged needs {sorted(missing)}")
        spec["lens"] = [ragged_lens(B, n, ll, ls)
                        for B, n, ll, ls in product(r["B"], r["n_long"], r["L_long"], r["L_short"]) if n < B]
    items = [(k, list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in spec.items()]
    out = []
    for combo in product(*(vals for _, vals in items)):
        kw = dict(zip((k for k, _ in items), combo))
        if "lens" in kw:
            if {"B", "L", "L_kv"} & kw.keys():
                raise ValueError("lens sets B and L_kv; do not also give B, L or L_kv")
            lens = parse_lens(kw.pop("lens"))
            kw.update(B=len(lens), L_kv=max(lens), kv_lens=lens)
            kw.setdefault("L_q", 1)
        if "L" in kw:
            L = kw.pop("L")
            kw.setdefault("L_kv", L)
            kw.setdefault("L_q", 1 if kw["phase"] == "decode" else L)
        out.append(Workload(**kw))
    return out
```

- [ ] **Step 4: Run tests** — `$PY -m pytest tests/test_workload.py -q` → all PASS (pre-existing tests unchanged).
- [ ] **Step 5: Full CPU suite, then commit** — `git commit -m "Support ragged decode workloads and underscore dtypes in workload keys" -- kernelscope/workload.py tests/test_workload.py`

---

### Task 3: Ragged-aware analytic model, reference, correctness check and plugin base

**Depends on:** Task 2.

**Files:**
- Modify: `kernelscope/analytic.py`, `kernelscope/reference.py`, `kernelscope/check.py`, `kernelscope/plugins/base.py`, `kernelscope/plugins/builtin/sdpa.py` (`_SDPA.supports`), `kernelscope/plugins/builtin/naive_exec.py` (`supports`), `kernelscope/plugins/builtin/triton_tutorial.py` (`supports`), `tests/fake_plugins.py`
- Test: `tests/test_analytic.py`, `tests/test_reference.py`, `tests/test_check.py`, `tests/test_plugins.py`

**Interfaces:**
- Consumes: `Workload.is_ragged`, `Workload.lens()`, `Workload.kv_lens` (Task 2)
- Produces:
  - `analytic.total_attended_pairs(w) -> int`; `attention_flops` and `attention_traffic` sum over per-sequence lengths
  - `reference_attention(q, k, v, causal, kv_lens=None)` — keys at positions `>= kv_lens[b]` are masked for batch `b`; bottom-right causal alignment is relative to each sequence's length
  - `check_outputs(plugin, w, inputs, atol=1e-2) -> dict` (same result dict as `check_plugin`; used by the batch runner in Task 7 to check already-built inputs)
  - `_PluginBase.supports_ragged: bool = False`; base `supports(w)` returns False for ragged workloads unless `supports_ragged`
  - `tests.fake_plugins.FaithfulCPU.supports_ragged = True` and honours `kv_lens`

- [ ] **Step 1: Write the failing tests.**

`tests/test_analytic.py` (append):
```python
from kernelscope.analytic import attention_flops, attention_traffic, total_attended_pairs
from kernelscope.workload import Workload

RAGGED = Workload(phase="decode", B=3, L_q=1, L_kv=1000, H_q=4, H_kv=2, d=8, kv_lens=[1000, 24, 100])


def test_ragged_pairs_and_flops_sum_over_sequences():
    assert total_attended_pairs(RAGGED) == 1124
    assert attention_flops(RAGGED) == 4 * 4 * 8 * 1124


def test_ragged_kv_bytes_sum_over_sequences():
    t = attention_traffic(RAGGED)
    assert t["kv_bytes"] == 2 * 1124 * 2 * 8 * 2
    assert t["q_bytes"] == 3 * 1 * 4 * 8 * 2


def test_uniform_totals_are_unchanged():
    w = Workload(phase="decode", B=3, L_q=1, L_kv=1000, H_q=4, H_kv=2, d=8)
    assert total_attended_pairs(w) == 3000
    assert attention_traffic(w)["kv_bytes"] == 2 * 3 * 1000 * 2 * 8 * 2
```

`tests/test_reference.py` (append):
```python
def test_ragged_reference_equals_uniform_reference_on_each_truncated_sequence():
    torch.manual_seed(0)
    q = torch.randn(3, 1, 4, 8)
    k = torch.randn(3, 50, 2, 8)
    v = torch.randn(3, 50, 2, 8)
    lens = [50, 7, 23]
    out = reference_attention(q, k, v, causal=True, kv_lens=lens)
    for b, n in enumerate(lens):
        ref_b = reference_attention(q[b:b + 1], k[b:b + 1, :n], v[b:b + 1, :n], causal=True)
        assert torch.allclose(out[b:b + 1], ref_b, atol=1e-6)


def test_kv_lens_none_keeps_the_old_behaviour():
    torch.manual_seed(1)
    q, k, v = torch.randn(2, 3, 4, 8), torch.randn(2, 9, 2, 8), torch.randn(2, 9, 2, 8)
    assert torch.equal(reference_attention(q, k, v, True), reference_attention(q, k, v, True, kv_lens=None))
```
(Use the file's existing imports; add `import torch` and `from kernelscope.reference import reference_attention` if absent.)

`tests/test_check.py` (append):
```python
from kernelscope.check import check_outputs, check_plugin
from kernelscope.reference import reference_attention
from kernelscope.workload import Workload
from tests.fake_plugins import FaithfulCPU

RAGGED_W = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32",
                    kv_lens=[64, 5, 30])


class IgnoresLengths(FaithfulCPU):
    """Attends to the whole cache: wrong for a ragged batch."""
    name = "ignores_lengths"

    def run(self, inputs):
        return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"])


def test_check_passes_a_plugin_that_honours_ragged_lengths():
    r = check_plugin(FaithfulCPU(device="cpu"), RAGGED_W)
    assert r["ok"] is True, r


def test_check_fails_a_plugin_that_ignores_ragged_lengths():
    r = check_plugin(IgnoresLengths(device="cpu"), RAGGED_W)
    assert r["ok"] is False
    assert r["max_abs_diff"] > 1e-2


def test_check_outputs_uses_the_given_inputs():
    p = FaithfulCPU(device="cpu")
    inputs = p.build_inputs(RAGGED_W)
    assert check_outputs(p, RAGGED_W, inputs)["ok"] is True
```

`tests/test_plugins.py` (append):
```python
from kernelscope.workload import Workload
from tests.fake_plugins import FakeExec, FaithfulCPU

_R = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32", kv_lens=[64, 3])


def test_plugins_do_not_support_ragged_workloads_unless_they_opt_in():
    assert FaithfulCPU(device="cpu").supports(_R) is True
    assert FakeExec(device="cpu").supports(_R) is False


def test_builtin_non_flash_plugins_refuse_ragged_workloads():
    from kernelscope.plugins.builtin.sdpa import SDPAFlash
    assert SDPAFlash(device="cpu").supports(_R) is False
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_analytic.py tests/test_reference.py tests/test_check.py tests/test_plugins.py -q` → FAIL (`ImportError: total_attended_pairs`, `TypeError: unexpected keyword 'kv_lens'`, ...).

- [ ] **Step 3: Implement.**

`kernelscope/analytic.py` — add `total_attended_pairs` and use it; replace `attention_flops` and the `kv` line of `attention_traffic`:
```python
def total_attended_pairs(w: Workload) -> int:
    """(query, key) pairs over the whole batch. A ragged batch is decode with L_q=1, so each
    sequence's single query attends to its whole live cache."""
    if w.is_ragged:
        return sum(w.lens())
    return w.B * attended_pairs(w)


def attention_flops(w: Workload) -> int:
    """QK^T and PV: 2 MACs per (pair, head, dim) = 4 FLOPs."""
    return 4 * w.H_q * w.d * total_attended_pairs(w)
```
and in `attention_traffic`: `kv = 2 * sum(w.lens()) * h_kv * w.d * s`.

`kernelscope/reference.py` — new signature and masking (keep the docstring, add one sentence about `kv_lens`):
```python
def reference_attention(q, k, v, causal: bool, kv_lens=None):
    B, L_q, H_q, d = q.shape
    L_kv, H_kv = k.shape[1], k.shape[2]
    group = H_q // H_kv
    kf = k.float().repeat_interleave(group, dim=2)
    vf = v.float().repeat_interleave(group, dim=2)
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), kf) * d ** -0.5
    lens = torch.as_tensor(list(kv_lens) if kv_lens is not None else [L_kv] * B, device=q.device).view(-1, 1, 1, 1)
    j = torch.arange(L_kv, device=q.device).view(1, 1, 1, -1)
    s = s.masked_fill(j >= lens, float("-inf"))
    if causal:
        i = torch.arange(L_q, device=q.device).view(1, 1, -1, 1)
        s = s.masked_fill(j > (lens - L_q) + i, float("-inf"))
    p = torch.softmax(s, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", p, vf)
    return out.to(q.dtype)
```

`kernelscope/check.py`:
```python
def check_plugin(plugin, w, atol: float = 1e-2) -> dict:
    try:
        inputs = plugin.build_inputs(w)
    except Exception as e:  # a broken plugin must not take the sweep down
        return _result(False, math.nan, f"{type(e).__name__}: {e}")
    return check_outputs(plugin, w, inputs, atol)


def check_outputs(plugin, w, inputs, atol: float = 1e-2) -> dict:
    """Correctness of ``plugin.run(inputs)`` against the reference, on already-built inputs."""
    try:
        out = plugin.to_dense_output(plugin.run(inputs))
        q, k, v = plugin.to_dense_inputs(inputs)
        ref = reference_attention(q, k, v, w.causal, kv_lens=w.kv_lens)
        if tuple(out.shape) != tuple(ref.shape):
            return _result(False, math.nan,
                           f"shape mismatch: got {tuple(out.shape)}, expected {tuple(ref.shape)}")
        diff = (out.float() - ref.float()).abs().max().item()
        return _result(diff <= atol, diff, None)
    except Exception as e:
        return _result(False, math.nan, f"{type(e).__name__}: {e}")
```

`kernelscope/plugins/base.py` — in `_PluginBase` add the attribute and change `supports`:
```python
    supports_ragged = False       # True only if build_inputs/run honour Workload.kv_lens

    def supports(self, w: Workload) -> bool:
        return w.phase in self.phases and (self.supports_ragged or not w.is_ragged)
```

`sdpa.py` `_SDPA.supports`, `naive_exec.py` `supports`, `triton_tutorial.py` `supports` — insert as the first line of each method body:
```python
        if w.is_ragged:
            return False
```

`tests/fake_plugins.py` `FaithfulCPU`: add `supports_ragged = True`; in `build_inputs` add `"kv_lens": w.kv_lens` to the returned dict; `run` becomes
`return reference_attention(inputs["q"], inputs["k"], inputs["v"], inputs["causal"], kv_lens=inputs.get("kv_lens"))`.

- [ ] **Step 4: Run tests** — the four test files → PASS.
- [ ] **Step 5: Full CPU suite, then commit** — `git commit -m "Make the analytic model, reference and correctness check ragged-aware" -- kernelscope/analytic.py kernelscope/reference.py kernelscope/check.py kernelscope/plugins/base.py kernelscope/plugins/builtin/sdpa.py kernelscope/plugins/builtin/naive_exec.py kernelscope/plugins/builtin/triton_tutorial.py tests/fake_plugins.py tests/test_analytic.py tests/test_reference.py tests/test_check.py tests/test_plugins.py`

---

### Task 4: Paged-KV helpers and the flash-attn plugin family

**Depends on:** Task 3.

The serving loop must use a paged KV cache (a dense cache gives every sequence the longest length's capacity and a ragged batch no longer fits in 24 GB). The paged path always runs the split-KV kernel (`force_split_kernel` in `flash_api.cpp:1455-1456`), so it must be measured separately. Fixed-split variants are the dispatcher's action space.

**Files:**
- Create: `kernelscope/plugins/paged.py`
- Modify: `kernelscope/plugins/builtin/flash.py` (whole file)
- Test: `tests/test_paged.py` (new, CPU), `tests/test_flash_plugins.py` (append, GPU)

**Interfaces:**
- Consumes: `Workload.lens()`, `Workload.kv_lens`, `supports_ragged` (Tasks 2–3)
- Produces:
  - `paged.PAGE = 256`; `build_block_table(lens, page=PAGE, seed=0) -> tuple[torch.Tensor, int]` (CPU int32 table `[B, max_pages]`, number of pool pages); `gather_from_pool(pool, lens, table, L, page=PAGE) -> torch.Tensor` (`[B, L, H, d]`, zeros past each length)
  - Plugins (registry names): `fa2`, `flashdecoding` (unchanged names; now ragged-capable), `fd_s{N}` for N ∈ {2, 4, 8, 16, 32, 64, 128}, `fa2_paged`, `flashdecoding_paged`, `fd_s{N}_paged`. All but `fa2` are decode-only. `flash.SPLITS = (2, 4, 8, 16, 32, 64, 128)`.
  - inputs dict keys: `q, k, v, causal, phase, L_kv`; decode adds `lens, cache_seqlens`; paged adds `block_table` (device) and `block_table_cpu`.

- [ ] **Step 1: Write the failing CPU tests** — create `tests/test_paged.py`:

```python
import torch

from kernelscope.plugins.paged import PAGE, build_block_table, gather_from_pool


def test_block_table_gives_every_sequence_distinct_pages():
    lens = [1000, 256, 1, 513]
    table, n = build_block_table(lens)
    need = [4, 1, 1, 3]
    assert n == sum(need)
    assert table.dtype == torch.int32 and table.shape == (4, 4)
    used = [int(table[b, i]) for b, k in enumerate(need) for i in range(k)]
    assert sorted(used) == list(range(n))            # a permutation of the pool


def test_block_table_is_seeded():
    assert torch.equal(build_block_table([700, 300])[0], build_block_table([700, 300])[0])
    assert not torch.equal(build_block_table([700, 300], seed=0)[0], build_block_table([700, 300], seed=1)[0])


def test_gather_reads_token_t_of_sequence_b_from_its_page():
    lens = [600, 5]
    table, n = build_block_table(lens)
    pool = torch.arange(n * PAGE * 2 * 3, dtype=torch.float32).view(n, PAGE, 2, 3)
    dense = gather_from_pool(pool, lens, table, L=600)
    assert dense.shape == (2, 600, 2, 3)
    for b, t in [(0, 0), (0, 255), (0, 256), (0, 599), (1, 4)]:
        assert torch.equal(dense[b, t], pool[int(table[b, t // PAGE]), t % PAGE])
    assert torch.count_nonzero(dense[1, 5:]) == 0     # past the sequence length
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_paged.py -q` → FAIL (`ModuleNotFoundError: kernelscope.plugins.paged`).

- [ ] **Step 3: Implement `kernelscope/plugins/paged.py`:**

```python
"""Paged KV-cache helpers for flash-attn's ``block_table`` layout.

A pool ``[num_pages, PAGE, H_kv, d]`` holds every sequence's K (or V) in fixed-size pages;
``block_table[b, i]`` is the pool page holding tokens ``[i*PAGE, (i+1)*PAGE)`` of sequence
``b``. flash-attn requires the page size to be a multiple of 256. Pages are handed out in a
seeded random order, like a fragmented pool in a serving engine.
"""
import math

import torch

PAGE = 256


def build_block_table(lens, page: int = PAGE, seed: int = 0) -> tuple[torch.Tensor, int]:
    """CPU int32 table ``[B, max_pages]`` and the pool size. Unused slots hold page 0, which
    the kernel never reads because it stops at each sequence's length."""
    need = [math.ceil(n / page) for n in lens]
    total = sum(need)
    perm = torch.randperm(total, generator=torch.Generator().manual_seed(seed)).to(torch.int32)
    table = torch.zeros(len(lens), max(need), dtype=torch.int32)
    pos = 0
    for b, n in enumerate(need):
        table[b, :n] = perm[pos:pos + n]
        pos += n
    return table, total


def gather_from_pool(pool: torch.Tensor, lens, table: torch.Tensor, L: int, page: int = PAGE) -> torch.Tensor:
    """Dense ``[B, L, H, d]`` copy of a paged cache (for the reference check); zeros past each length."""
    out = pool.new_zeros(len(lens), L, pool.shape[2], pool.shape[3])
    for b, n in enumerate(lens):
        for i in range(math.ceil(n / page)):
            lo, hi = i * page, min(n, (i + 1) * page)
            out[b, lo:hi] = pool[int(table[b, i]), : hi - lo]
    return out
```

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_paged.py -q` → PASS.

- [ ] **Step 5: Write the failing GPU tests** — append to `tests/test_flash_plugins.py`:

```python
from kernelscope.plugins.builtin import flash as flash_mod  # noqa: E402
from kernelscope.run_kernel import profile_launches  # noqa: E402

RAGGED = Workload(phase="decode", B=4, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128, kv_lens=(1024, 300, 700, 5))
UNIFORM = Workload(phase="decode", B=2, L_q=1, L_kv=2048, H_q=32, H_kv=8, d=128)
DECODE_PLUGINS = [c for c in flash_mod.PLUGINS if "decode" in c.phases]


def test_registry_exposes_the_fixed_split_and_paged_families():
    names = set(REGISTRY.names())
    for n in flash_mod.SPLITS:
        assert {f"fd_s{n}", f"fd_s{n}_paged"} <= names
    assert {"fa2_paged", "flashdecoding_paged"} <= names


@pytest.mark.parametrize("cls", DECODE_PLUGINS, ids=lambda c: c.name)
@pytest.mark.parametrize("w", [RAGGED, UNIFORM], ids=["ragged", "uniform"])
def test_every_decode_variant_matches_the_reference(cls, w):
    r = check_plugin(cls(device="cuda"), w, atol=2e-2)
    assert r["ok"], r


def test_fixed_split_plugin_launches_exactly_that_many_splits():
    p = REGISTRY.get("fd_s8", device="cuda")
    inputs = p.build_inputs(Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128))
    for _ in range(3):
        p.run(inputs)
    s = profile_launches(p, inputs, "cuda", iters=3)
    main = [l for l in s["launches"] if "combine" not in l["name"]][0]
    assert main["grid"][1] == 8          # grid = (m_blocks, num_splits, B * H_kv)


def test_paged_fa2_runs_the_split_kernel():
    p = REGISTRY.get("fa2_paged", device="cuda")
    names = launched_kernels(p, p.build_inputs(UNIFORM), "cuda")
    assert any("splitkv" in n for n in names), names
```

- [ ] **Step 6: Implement `kernelscope/plugins/builtin/flash.py`** — replace the file:

```python
"""flash-attn 2.x plugins.

Decode goes through ``flash_attn_with_kvcache``; each sequence's live length is passed as
``cache_seqlens`` so ragged batches work. Variants (all compute the same attention):
  fa2                  num_splits=1  -> plain ``flash_fwd_kernel`` (grid = B x H_kv CTAs after GQA packing)
  flashdecoding        num_splits=0  -> the library's heuristic (``num_splits_heuristic`` in flash_api.cpp)
  fd_s{N}              num_splits=N  -> ``flash_fwd_splitkv_kernel`` + ``..._combine_kernel``
  *_paged              same, on a paged cache (``block_table``, page 256). flash-attn always
                       uses the split kernel on this path, even for num_splits=1.
Dense caches have capacity L_kv = the longest live length (never an over-allocated capacity:
the heuristic sizes splits from ``k_cache.size(1)``, spec F17).
Prefill (fa2 only) goes through ``flash_attn_func`` (bottom-right causal, like the reference).
Inputs are generated on the device (a CPU generator would take minutes and tens of GB of
host memory for the largest grid cells).
"""
import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from kernelscope.plugins.base import KernelPlugin
from kernelscope.plugins.paged import PAGE, build_block_table, gather_from_pool
from kernelscope.workload import Workload

SPLITS = (2, 4, 8, 16, 32, 64, 128)


class _Flash(KernelPlugin):
    kernel_regex = r"flash_fwd"
    num_splits = 1
    paged = False
    supports_ragged = True

    def build_inputs(self, w: Workload) -> dict:
        dt = getattr(torch, w.dtype)
        g = torch.Generator(device=self.device).manual_seed(0)

        def rand(*shape):
            return torch.randn(*shape, generator=g, device=self.device, dtype=dt)

        inputs = {"q": rand(w.B, w.L_q, w.H_q, w.d), "causal": w.causal, "phase": w.phase, "L_kv": w.L_kv}
        if w.phase == "decode":
            lens = w.lens()
            inputs["lens"] = lens
            inputs["cache_seqlens"] = torch.tensor(lens, dtype=torch.int32, device=self.device)
            if self.paged:
                table, pages = build_block_table(lens)
                inputs["k"] = rand(pages, PAGE, w.H_kv, w.d)
                inputs["v"] = rand(pages, PAGE, w.H_kv, w.d)
                inputs["block_table_cpu"] = table
                inputs["block_table"] = table.to(self.device)
                return inputs
        inputs["k"] = rand(w.B, w.L_kv, w.H_kv, w.d)
        inputs["v"] = rand(w.B, w.L_kv, w.H_kv, w.d)
        return inputs

    def run(self, inputs: dict):
        if inputs["phase"] == "decode":
            return flash_attn_with_kvcache(
                inputs["q"], inputs["k"], inputs["v"], cache_seqlens=inputs["cache_seqlens"],
                block_table=inputs.get("block_table"), causal=inputs["causal"], num_splits=self.num_splits,
            )
        return flash_attn_func(inputs["q"], inputs["k"], inputs["v"], causal=inputs["causal"])

    def to_dense_inputs(self, inputs: dict):
        if "block_table" in inputs:
            t, lens, L = inputs["block_table_cpu"], inputs["lens"], inputs["L_kv"]
            return inputs["q"], gather_from_pool(inputs["k"], lens, t, L), gather_from_pool(inputs["v"], lens, t, L)
        return inputs["q"], inputs["k"], inputs["v"]

    def to_dense_output(self, out):
        return out


class FA2(_Flash):
    name = "fa2"
    phases = frozenset({"decode", "prefill"})
    num_splits = 1


class FlashDecoding(_Flash):
    name = "flashdecoding"
    phases = frozenset({"decode"})
    num_splits = 0


def _variant(name: str, num_splits: int, paged: bool) -> type:
    return type(name, (_Flash,), {"name": name, "phases": frozenset({"decode"}),
                                  "num_splits": num_splits, "paged": paged})


FIXED = [_variant(f"fd_s{n}", n, False) for n in SPLITS]
PAGED = ([_variant("fa2_paged", 1, True), _variant("flashdecoding_paged", 0, True)]
         + [_variant(f"fd_s{n}_paged", n, True) for n in SPLITS])

PLUGINS = [FA2, FlashDecoding, *FIXED, *PAGED]
```

- [ ] **Step 7: Run GPU tests** (hygiene first) — `$PY -m pytest -q -m gpu tests/test_flash_plugins.py` → all PASS, including the pre-existing ones.
- [ ] **Step 8: Full CPU suite, then commit** — `git commit -m "Add paged-KV and fixed-split flash-attn plugin variants" -- kernelscope/plugins/paged.py kernelscope/plugins/builtin/flash.py tests/test_paged.py tests/test_flash_plugins.py`

---

### Task 5: Iteration markers, L2 flushing and marker-segmented profiling

**Depends on:** Task 1 (same file `kprofile.py`).

Every existing real-HW number is warm-cache (no flush), and in serving the attention call is cold (spec F7, F8). torch.profiler also drops the first kernel events of a profiled block in long-running processes, which breaks the `n % iters == 0` assumption. One mechanism fixes both: a named boundary kernel before every timed call.

**Files:**
- Create: `kernelscope/backends/realhw/cache.py`
- Modify: `kernelscope/backends/realhw/kprofile.py` (`summarize_launches`)
- Test: `tests/test_kprofile.py` (append, CPU), `tests/test_cache_hooks.py` (new, GPU)

**Interfaces:**
- Produces:
  - `cache.MARKER_REGEX = r"kernelscope_(l2_flush|iter_marker)"`, `cache.CACHE_STATES = ("warm", "cold")`, `cache.FLUSH_FACTOR = 4`
  - `cache.IterationHooks(device: str, cache_state: str = "warm", l2_bytes: int | None = None)` with `.active: bool` (False off-GPU), `.cache_state`, `.between() -> None` (launch the flush kernel when cold, the marker kernel when warm; no-op when inactive)
  - `kprofile.summarize_launches(events, kernel_regex, iters, marker_regex=None) -> dict` — with markers present: iterations are the matching launches between consecutive markers; iterations whose launch count differs from the most common count are dropped; the last `iters` good iterations are summarised; extra keys `iterations_used`, `iterations_dropped`. Marker kernels are never counted, even if `kernel_regex` matches them. Without markers: legacy behaviour.

- [ ] **Step 1: Write the failing CPU tests** — append to `tests/test_kprofile.py`:

```python
def _ev(name, dur, ts):
    return {"name": name, "dur_us": dur, "ts": ts, "grid": (1, 1, 1), "block": (128, 1, 1), "regs": 32, "smem_bytes": 0}


MARK = "kernelscope_iter_marker"
FLUSH = "kernelscope_l2_flush"
MRX = r"kernelscope_(l2_flush|iter_marker)"


def _iterations(durs, marker=MARK, t0=0):
    """One marker then (main, combine) per iteration; durs = [(main, combine), ...]."""
    ev, t = [], t0
    for a, b in durs:
        ev += [_ev(marker, 0.5, t), _ev("flash_fwd_splitkv_kernel", a, t + 1), _ev("flash_fwd_splitkv_combine_kernel", b, t + 2)]
        t += 10
    return ev


def test_segmented_summary_uses_the_last_iters_complete_iterations():
    ev = _iterations([(100, 9), (100, 9), (10, 1), (12, 2), (14, 3)])   # 2 padding + 3 measured
    s = summarize_launches(ev, "flash_fwd", iters=3, marker_regex=MRX)
    assert s["launches_per_iter"] == 2
    assert s["launches"][0]["dur_us_median"] == 12
    assert s["launches"][1]["dur_us_median"] == 2
    assert s["kernel_time_us_median"] == 14
    assert s["iterations_used"] == 3 and s["iterations_dropped"] == 0
    assert s["unmatched"] == []


def test_segmented_summary_drops_iterations_with_missing_events():
    ev = _iterations([(10, 1), (12, 2), (14, 3), (16, 4)])
    del ev[4]                                       # profiler lost iteration 2's main kernel
    s = summarize_launches(ev, "flash_fwd", iters=3, marker_regex=MRX)
    assert s["iterations_dropped"] == 1
    assert s["iterations_used"] == 3
    assert s["kernel_time_us_median"] == 17         # sums 11, 17, 20 -> median 17


def test_flush_kernels_are_never_counted_even_by_a_catch_all_regex():
    ev = _iterations([(10, 1), (12, 2)], marker=FLUSH)
    s = summarize_launches(ev, ".*", iters=2, marker_regex=MRX)
    assert s["launches_per_iter"] == 2
    assert all("kernelscope" not in l["name"] for l in s["launches"])


def test_launches_before_the_first_marker_are_ignored():
    ev = [_ev("flash_fwd_splitkv_kernel", 999, -5)] + _iterations([(10, 1), (12, 2)])
    s = summarize_launches(ev, "flash_fwd", iters=2, marker_regex=MRX)
    assert s["kernel_time_us_median"] == 12.5


def test_marker_regex_without_markers_in_the_trace_falls_back_to_the_legacy_path():
    ev = kernel_events_from_chrome_trace(TRACE)
    s = summarize_launches(ev, "flash_fwd", iters=2, marker_regex=MRX)
    assert s["kernel_time_us_median"] == 13.5
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_kprofile.py -q` → FAIL (`TypeError: unexpected keyword argument 'marker_regex'`).

- [ ] **Step 3: Implement `summarize_launches`** in `kprofile.py` (replace the function; `statistics` and `re` are already imported):

```python
def summarize_launches(events: list[dict], kernel_regex: str | None, iters: int, marker_regex: str | None = None) -> dict:
    """Per-launch median duration and geometry of the kernels matching ``kernel_regex``.

    With ``marker_regex`` and marker kernels in the trace, iterations are the matching launches
    between consecutive markers; iterations whose launch count differs from the most common one
    (the profiler dropped an event) are discarded and the last ``iters`` good ones summarised.
    Marker kernels are never counted. Without markers the trace must hold exactly ``iters``
    iterations (legacy behaviour).
    """
    rx = re.compile(kernel_regex) if kernel_regex else None
    mx = re.compile(marker_regex) if marker_regex else None

    def is_marker(e):
        return bool(mx and mx.search(e["name"]))

    def matches(e):
        return bool(rx and rx.search(e["name"])) and not is_marker(e)

    unmatched = list(dict.fromkeys(e["name"] for e in events if not is_marker(e) and not matches(e)))
    if mx is not None and any(is_marker(e) for e in events):
        segments, cur = [], None
        for e in events:
            if is_marker(e):
                if cur is not None:
                    segments.append(cur)
                cur = []
            elif cur is not None and matches(e):
                cur.append(e)
        if cur is not None:
            segments.append(cur)
        per = statistics.mode(len(s) for s in segments)
        good = [s for s in segments if len(s) == per]
        used = good[-iters:]
        out = _launch_summary(used, per, unmatched)
        out.update(iterations_used=len(used), iterations_dropped=len(segments) - len(good))
        return out
    matching = [e for e in events if matches(e)]
    n = len(matching)
    if n % iters:
        raise ValueError(f"{n} matching launches is not a multiple of iters={iters}; "
                         f"launch count varies between iterations?")
    per = n // iters
    return _launch_summary([matching[k * per:(k + 1) * per] for k in range(iters)], per, unmatched)


def _launch_summary(iterations: list[list[dict]], per: int, unmatched: list[str]) -> dict:
    if per == 0 or not iterations:
        return {"launches_per_iter": 0, "launches": [], "kernel_time_us_median": None, "unmatched": unmatched}
    launches = []
    for idx in range(per):
        evs = [it[idx] for it in iterations]
        launches.append({
            "idx": idx, "name": evs[0]["name"],
            "dur_us_median": statistics.median(e["dur_us"] for e in evs),
            "grid": evs[0]["grid"], "block": evs[0]["block"],
            "regs": evs[0]["regs"], "smem_bytes": evs[0]["smem_bytes"],
        })
    total = statistics.median(sum(e["dur_us"] for e in it) for it in iterations)
    return {"launches_per_iter": per, "launches": launches, "kernel_time_us_median": total, "unmatched": unmatched}
```

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_kprofile.py -q` → PASS (old and new tests).

- [ ] **Step 5: Implement `kernelscope/backends/realhw/cache.py`:**

```python
"""Iteration boundaries and L2 flushing for Nsight-free timing loops.

Before every timed call the harness launches one of two tiny Triton kernels whose names it
recognises in the profiler trace:
  kernelscope_l2_flush     cache_state="cold": writes FLUSH_FACTOR x L2 bytes, evicting the
                           previous call's data (the 4090's L2 is not LRU, so 2 x L2 can leave
                           some of it resident — spec F6)
  kernelscope_iter_marker  cache_state="warm": writes one float
Either one marks an iteration boundary, so the trace can be cut into iterations even when
torch.profiler drops events, and neither is ever counted as kernel time.
In serving, attention is cold: the other layers' weights stream through L2 between two
attention calls of the same layer (spec F8).
"""
MARKER_REGEX = r"kernelscope_(l2_flush|iter_marker)"
CACHE_STATES = ("warm", "cold")
FLUSH_FACTOR = 4
_BLOCK = 4096

try:
    import triton
    import triton.language as tl
except ImportError:          # CPU-only environments: hooks stay inactive
    triton = None

if triton is not None:
    @triton.jit
    def kernelscope_l2_flush(buf_ptr, n, val, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(buf_ptr + offs, tl.zeros((BLOCK,), tl.float32) + val, mask=offs < n)

    @triton.jit
    def kernelscope_iter_marker(buf_ptr, val):
        tl.store(buf_ptr, val)


class IterationHooks:
    def __init__(self, device: str, cache_state: str = "warm", l2_bytes: int | None = None):
        if cache_state not in CACHE_STATES:
            raise ValueError(f"cache_state must be one of {CACHE_STATES}, got {cache_state!r}")
        self.cache_state = cache_state
        self.active = str(device).startswith("cuda") and triton is not None
        self._n = 0
        if not self.active:
            return
        import torch
        if cache_state == "cold":
            if l2_bytes is None:
                l2_bytes = torch.cuda.get_device_properties(device).L2_cache_size
            self._buf = torch.empty(FLUSH_FACTOR * l2_bytes // 4, dtype=torch.float32, device=device)
        else:
            self._buf = torch.empty(1, dtype=torch.float32, device=device)

    def between(self) -> None:
        """Launch the boundary kernel that precedes the next timed call (no-op when inactive)."""
        if not self.active:
            return
        self._n += 1
        if self.cache_state == "cold":
            n = self._buf.numel()
            kernelscope_l2_flush[(triton.cdiv(n, _BLOCK),)](self._buf, n, float(self._n), BLOCK=_BLOCK)
        else:
            kernelscope_iter_marker[(1,)](self._buf, float(self._n))
```

- [ ] **Step 6: Write the GPU test** — create `tests/test_cache_hooks.py`:

```python
"""GPU-only: the boundary kernels exist under the names the profiler parser expects."""
import re

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.backends.realhw.cache import FLUSH_FACTOR, MARKER_REGEX, IterationHooks  # noqa: E402
from kernelscope.run_kernel import launched_kernels  # noqa: E402

pytestmark = pytest.mark.gpu


class _Plugin:
    """Adapter so launched_kernels() can profile a bare callable."""
    def __init__(self, fn):
        self.run = lambda inputs: fn()


@pytest.mark.parametrize("state, name", [("cold", "kernelscope_l2_flush"), ("warm", "kernelscope_iter_marker")])
def test_boundary_kernels_carry_their_names_into_the_profiler(state, name):
    h = IterationHooks("cuda", state)
    h.between()                                   # JIT compile outside the profiled region
    torch.cuda.synchronize()
    names = launched_kernels(_Plugin(h.between), None, "cuda")
    assert any(name in n for n in names), names
    assert all(re.search(MARKER_REGEX, n) for n in names if "kernelscope" in n)


def test_cold_flush_buffer_is_four_times_l2():
    h = IterationHooks("cuda", "cold")
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    assert h._buf.numel() * 4 == FLUSH_FACTOR * l2
```

- [ ] **Step 7: Run** (hygiene first) — `$PY -m pytest -q -m gpu tests/test_cache_hooks.py` → PASS.
- [ ] **Step 8: Full CPU suite, then commit** — `git commit -m "Add named L2-flush and iteration-marker kernels; segment profiler traces by them" -- kernelscope/backends/realhw/cache.py kernelscope/backends/realhw/kprofile.py tests/test_kprofile.py tests/test_cache_hooks.py`

---

### Task 6: Cache-state mode in run_kernel, the sweep, the analytic rows and the CLI

**Depends on:** Task 5.

**Files:**
- Modify: `kernelscope/backends/realhw/latency.py` (`measure_latency`), `kernelscope/run_kernel.py` (`profile_launches`, `main`), `kernelscope/backends/realhw/sweep.py` (`profile_rows`, `analytic_rows`, `RealHWSweep`), `kernelscope/cli.py` (`_cmd_sweep` and the `p_sweep` parser block only)
- Test: `tests/test_latency.py`, `tests/test_run_kernel.py`, `tests/test_sweep.py`, `tests/test_cli.py`, `tests/test_cache_state_gpu.py` (new, GPU)

**Interfaces:**
- Consumes: `IterationHooks`, `MARKER_REGEX` (Task 5); `summarize_launches(..., marker_regex=)` (Task 5)
- Produces:
  - `measure_latency(fn, warmup=10, iters=50, timer=None, before=None)` — `before()` runs before every warm-up and timed call, outside the timed region
  - `run_kernel.profile_launches(plugin, inputs, device, iters, hooks=None, pad=0) -> dict` — with active hooks runs `pad + iters` calls, each preceded by `hooks.between()`, and summarises the last `iters` marker-delimited iterations; without hooks: unchanged behaviour
  - `run_kernel` flags `--cache-state {warm,cold}` (default `warm`) and `--pad N` (default 5); every JSON output gains `"cache_state"`
  - `sweep.analytic_rows(w, plugin, kv_heads_read, kernel_time_us, ceilings, cache_state="warm")` — emits `dram_util` only for `cold`
  - `sweep.profile_rows` additionally writes `iterations_used` / `iterations_dropped` rows when present
  - `RealHWSweep(..., cache_state="warm", pad=5)`; `_extra()` includes `cache_state`; executable plugins in `cold` return status `unsupported` (binaries time themselves; no flush between their iterations)
  - CLI: `sweep --cache-state warm,cold` (comma list; one pass per state), `--pad`

- [ ] **Step 1: Write the failing tests.**

`tests/test_latency.py` (append):
```python
def test_before_hook_runs_outside_the_timed_region_before_every_call():
    order = []
    def timer(fn):
        order.append("timed"); fn(); return 1.0
    measure_latency(lambda: None, warmup=2, iters=3, timer=timer, before=lambda: order.append("before"))
    assert order == ["before", "timed"] * 5
```

`tests/test_run_kernel.py` (append):
```python
def test_profile_and_latency_modes_report_the_cache_state(capsys):
    for mode in ("latency", "profile"):
        run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", mode, "--cache-state", "cold",
                         "--warmup", "1", "--iters", "2", "--registry", REG, "--device", "cpu"])
        out = json.loads(capsys.readouterr().out)
        assert out["cache_state"] == "cold"


def test_cache_state_is_validated():
    with pytest.raises(SystemExit):
        run_kernel.main(["--plugin", "faithful_cpu", "--workload", KEY, "--mode", "latency",
                         "--cache-state", "lukewarm", "--registry", REG, "--device", "cpu"])
```

`tests/test_sweep.py` (append; `analytic_rows` and `_sweep` exist in that module/file):
```python
from kernelscope.backends.realhw.sweep import analytic_rows


def test_dram_util_is_only_reported_for_cold_measurements():
    ceil = {"hbm_gbps": 1000.0, "fp16_matmul_tflops": 100.0}
    warm = {r["metric"] for r in analytic_rows(W, "p", 2, 10.0, ceil, cache_state="warm")}
    cold = {r["metric"] for r in analytic_rows(W, "p", 2, 10.0, ceil, cache_state="cold")}
    assert "dram_util" not in warm and "achieved_gbps" in warm
    assert "dram_util" in cold


def test_sweep_passes_the_cache_state_to_subprocesses_and_records_it(tmp_path):
    s = _sweep(tmp_path, cache_state="cold")
    argv = s._run_kernel_argv("faithful_cpu", W, "profile")
    assert argv[argv.index("--cache-state") + 1] == "cold"
    assert s._extra()["cache_state"] == "cold"


def test_executables_are_unsupported_in_cold_mode(tmp_path):
    s = _sweep(tmp_path, cache_state="cold")
    assert s.run_cell("fake_exec", W)["status"] == "unsupported"
```

`tests/test_cli.py` (append):
```python
def test_sweep_runs_one_pass_per_cache_state(tmp_path, capsys):
    grid = tmp_path / "grid.yaml"
    grid.write_text(GRID_YAML)
    out = tmp_path / "res"
    cli.main(["sweep", "--grid", str(grid), "--plugins", "faithful_cpu", "--results", str(out),
              "--registry", REG, "--device", "cpu", "--warmup", "1", "--iters", "2",
              "--cache-state", "warm,cold"])
    import pandas as pd
    from kernelscope.results.store import ResultStore
    df = ResultStore(out).load()
    assert set(df["cache_state"]) == {"warm", "cold"}
```

- [ ] **Step 2: Run to verify they fail** — `$PY -m pytest tests/test_latency.py tests/test_run_kernel.py tests/test_sweep.py tests/test_cli.py -q` → FAIL.

- [ ] **Step 3: Implement.**

`latency.py` — `measure_latency` gains `before=None`:
```python
def measure_latency(fn, warmup: int = 10, iters: int = 50, timer=None, before=None) -> dict:
    if timer is None:
        timer, name = _default_timer()
    else:
        name = "custom"

    def one():
        if before is not None:
            before()
        return timer(fn)

    for _ in range(warmup):
        one()
    samples = [one() for _ in range(iters)]
    return {"median_s": statistics.median(samples), "min_s": min(samples), "iters": iters,
            "timer": name, "samples_s": samples}
```

`run_kernel.py`:
- import `from kernelscope.backends.realhw.cache import CACHE_STATES, MARKER_REGEX, IterationHooks`
- `profile_launches(plugin, inputs, device: str, iters: int, hooks=None, pad: int = 0) -> dict`: 
```python
    active = hooks is not None and hooks.active
    runs = iters + (pad if active else 0)
    with profile(activities=activities) as prof:
        for _ in range(runs):
            if active:
                hooks.between()
            plugin.run(inputs)
        _sync(device)
    ...
    summary = summarize_launches(events, plugin.kernel_regex, iters, marker_regex=MARKER_REGEX if active else None)
```
- `main`: add `ap.add_argument("--cache-state", choices=CACHE_STATES, default="warm")` and `ap.add_argument("--pad", type=int, default=5)`; `base` dict gains `"cache_state": args.cache_state`; after `inputs = plugin.build_inputs(w)` create `hooks = IterationHooks(args.device, args.cache_state)`.
  - latency: `before = (lambda: (hooks.between(), _sync(args.device))) if args.cache_state == "cold" else None`, pass `before=before`.
  - profile: warm-up loop calls `hooks.between()` before each `plugin.run(inputs)`; then `profile_launches(plugin, inputs, args.device, args.iters, hooks=hooks, pad=args.pad)`.
  - `check`, `kernels`, `ncu`, `trace` modes are unchanged apart from `cache_state` in the printed JSON.

`sweep.py`:
- `profile_rows`: after the launch loop, `for k in ("iterations_used", "iterations_dropped"): if k in prof: rows.append(_row(w, plugin, "profile", k, prof[k]))`.
- `analytic_rows(..., ceilings, cache_state="warm")`: wrap the `dram_util` append in `if cache_state == "cold":` (keep `tc_util` for both).
- `RealHWSweep.__init__(..., cache_state="warm", pad=5)`: store both; validate `cache_state in CACHE_STATES`.
- `_extra()`: add `"cache_state": self.cache_state`.
- `_run_kernel_argv`: append `"--cache-state", self.cache_state, "--pad", str(self.pad)`.
- `run_cell`: pass `cache_state=self.cache_state` to `analytic_rows`; right after the `isinstance(meta, ExecutablePlugin)` check, if `self.cache_state == "cold"` set `summary["status"] = "unsupported"`, `summary["reason"] = "executables time themselves; cold mode needs in-process flushing"` and return it (do this before `_run_executable_cell`).

`cli.py` (`p_sweep` block and `_cmd_sweep` only): add `p_sweep.add_argument("--cache-state", default="warm", help="warm, cold, or a comma list (one pass per state)")` and `p_sweep.add_argument("--pad", type=int, default=5)`; in `_cmd_sweep` loop `for state in args.cache_state.split(","):` constructing a `RealHWSweep(..., cache_state=state, pad=args.pad)` per state and calling `_run_with_log` for each.

- [ ] **Step 4: Run** the four test files → PASS; then the full CPU suite.

- [ ] **Step 5: GPU acceptance test** — create `tests/test_cache_state_gpu.py`:
```python
"""GPU-only: cold measurements really are colder than warm ones for an L2-resident working set."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.backends.realhw.cache import IterationHooks  # noqa: E402
from kernelscope.plugins.builtin import REGISTRY  # noqa: E402
from kernelscope.run_kernel import profile_launches  # noqa: E402
from kernelscope.workload import Workload  # noqa: E402

pytestmark = pytest.mark.gpu

# 64 MiB of K/V: fits in the 72 MiB L2, so warm iterations are served from L2 (spec F7)
W = Workload(phase="decode", B=16, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128)


def _kernel_time(state):
    p = REGISTRY.get("flashdecoding", device="cuda")
    inputs = p.build_inputs(W)
    hooks = IterationHooks("cuda", state)
    for _ in range(5):
        hooks.between(); p.run(inputs)
    s = profile_launches(p, inputs, "cuda", iters=20, hooks=hooks, pad=5)
    assert all("kernelscope" not in l["name"] for l in s["launches"])
    assert s["iterations_used"] == 20
    return s["kernel_time_us_median"]


def test_cold_is_slower_than_warm_for_an_l2_resident_cell():
    assert _kernel_time("cold") > 1.2 * _kernel_time("warm")
```
Run (hygiene first): `$PY -m pytest -q -m gpu tests/test_cache_state_gpu.py` → PASS. Report both times in your hand-off.

- [ ] **Step 6: Commit** — `git commit -m "Add warm/cold cache-state measurement mode" -- kernelscope/backends/realhw/latency.py kernelscope/run_kernel.py kernelscope/backends/realhw/sweep.py kernelscope/cli.py tests/test_latency.py tests/test_run_kernel.py tests/test_sweep.py tests/test_cli.py tests/test_cache_state_gpu.py`

---

### Task 7: In-process batch runner (`kernelscope bench`)

**Depends on:** Tasks 3, 4, 6.

`sweep` spawns three subprocesses per cell (~10 s per cell with Python + torch start-up); the measurement grid has thousands of cells. `bench` measures every (workload, cache state) of each plugin inside one process with the same row builders and schema. `sweep` stays for ncu, executables and crash isolation.

**Files:**
- Create: `kernelscope/backends/realhw/batch.py`
- Modify: `kernelscope/cli.py` (add `_cmd_bench` and a `p_bench` parser block; do not touch other blocks)
- Test: `tests/test_batch.py` (new, CPU + one GPU test)

**Interfaces:**
- Consumes: `check_outputs` (Task 3); `IterationHooks`, `CACHE_STATES` (Task 5); `measure_latency(before=)`, `profile_launches(hooks=, pad=)`, `profile_rows`, `analytic_rows(cache_state=)`, `_row` (Task 6)
- Produces:
  - `batch.run_bench(plugins: list, workloads: list, cache_states: list[str], store, summaries_path, *, device="cuda", warmup=10, iters=30, pad=5, ceilings=None, check_max_ref_bytes=1 << 30, atol=1e-2, resume=True, log=print) -> list[dict]` — the reference check runs only when the float32 reference's expanded K+V (`2 * B * L_kv * H_q * d * 4` bytes; the reference repeats K/V to all query heads) fits in `check_max_ref_bytes`
  - `batch.done_cells(summaries_path) -> set[tuple[str, str, str]]` — `(plugin, workload_key, cache_state)` with status `ok`
  - Summary line per (plugin, workload, cache_state): `{"status", "plugin", "workload_key", "cache_state", "kernel_time_us", "latency_us", "launches", "check_ok", "iterations_dropped", "elapsed_s"}` (+ `"error"` or `"reason"`)
  - Rows: exactly those `sweep` writes (`latency`, `profile`, `analytic`, `check`), check rows only with the first cache state; extra columns `run_id, host, device, gpu_util_at_start, other_pids_at_start, runner="bench", cache_state`
  - CLI: `kernelscope bench --grid G [--grid G2 ...] | --workload KEY --plugins a,b --results DIR [--cache-state cold,warm] [--ceilings F] [--warmup 10] [--iters 30] [--pad 5] [--check-max-ref-mib 1024] [--atol 1e-2] [--device cuda] [--registry M:A] [--no-resume]`; default `--cache-state cold`

- [ ] **Step 1: Write the failing tests** — create `tests/test_batch.py`:

```python
import json

import pytest

from kernelscope.backends.realhw import batch
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload
from tests.fake_plugins import FakeExec, FaithfulCPU

W1 = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
W2 = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32", kv_lens=[64, 9, 30])
FAKE_PROF = {"launches_per_iter": 1, "kernel_time_us_median": 12.5, "unmatched": [], "device_props": None,
             "iterations_used": 3, "iterations_dropped": 1,
             "launches": [{"idx": 0, "name": "k", "dur_us_median": 12.5, "grid": (1, 8, 2), "block": (128, 1, 1),
                           "regs": 64, "smem_bytes": 0, "occupancy": {"blocks": 16, "sm_coverage": 0.125}}]}


@pytest.fixture
def fake_profile(monkeypatch):
    monkeypatch.setattr(batch, "profile_launches", lambda *a, **k: dict(FAKE_PROF))


def _run(tmp_path, plugins, workloads, states=("warm", "cold"), **kw):
    store = ResultStore(tmp_path / "res")
    summ = tmp_path / "res" / "summaries.jsonl"
    out = batch.run_bench(plugins, workloads, list(states), store, summ, device="cpu",
                          warmup=1, iters=3, pad=1, log=None, **kw)
    return out, store.load(), summ


def test_bench_writes_one_row_set_per_cache_state_with_the_state_column(tmp_path, fake_profile):
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1, W2])
    assert [s["status"] for s in out] == ["ok"] * 4
    kt = df[(df.backend == "profile") & (df.metric == "kernel_time_us")]
    assert sorted(kt.cache_state) == ["cold", "cold", "warm", "warm"]
    assert (df.runner == "bench").all()
    checks = df[df.backend == "check"]
    assert len(checks[checks.metric == "ok"]) == 2            # once per workload, not per state
    assert set(df[df.metric == "iterations_dropped"].value) == {1.0}


def test_bench_resumes_without_remeasuring(tmp_path, fake_profile):
    _run(tmp_path, [FaithfulCPU(device="cpu")], [W1])
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1])
    assert [s["status"] for s in out] == ["skipped", "skipped"]
    assert len(df[(df.backend == "profile") & (df.metric == "kernel_time_us")]) == 2


def test_executables_and_unsupported_workloads_are_reported_not_measured(tmp_path, fake_profile):
    out, _, _ = _run(tmp_path, [FakeExec(device="cpu")], [W1], states=("cold",))
    assert out[0]["status"] == "unsupported"


class Exploding(FaithfulCPU):
    name = "exploding"

    def run(self, inputs):
        raise RuntimeError("boom")


def test_a_failing_cell_does_not_stop_the_batch(tmp_path, fake_profile):
    out, _, summ = _run(tmp_path, [Exploding(device="cpu"), FaithfulCPU(device="cpu")], [W1], states=("cold",))
    assert [s["status"] for s in out] == ["error", "ok"]
    assert "boom" in out[0]["error"]
    lines = [json.loads(l) for l in summ.read_text().splitlines()]
    assert len(lines) == 2


def test_check_is_skipped_when_the_reference_would_be_too_large(tmp_path, fake_profile):
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1], states=("cold",), check_max_ref_bytes=10)
    assert out[0]["check_ok"] is None
    assert df[df.backend == "check"].empty
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_batch.py -q` → FAIL (`ImportError`).

- [ ] **Step 3: Implement `kernelscope/backends/realhw/batch.py`:**

```python
"""In-process batch runner for the Nsight-free real-HW track (`kernelscope bench`).

`sweep` isolates every measurement in its own subprocess (needed for ncu, the NVBit tracer and
crashing kernels) at ~10 s per cell. `bench` measures every (workload, cache state) of each
plugin inside this process with the same row builders and schema, at well under a second per
cell. A Python exception in one cell is recorded and the batch continues; a hard crash
(segfault) ends the process, and `resume` skips cells already measured.
"""
import json
import os
import socket
import time
import uuid
from pathlib import Path

from kernelscope.backends.realhw.cache import IterationHooks
from kernelscope.backends.realhw.hygiene import gpu_contention, visible_device_index
from kernelscope.backends.realhw.latency import measure_latency
from kernelscope.backends.realhw.sweep import _row, analytic_rows, profile_rows
from kernelscope.check import check_outputs
from kernelscope.plugins.base import KernelPlugin
from kernelscope.run_kernel import _sync, profile_launches


def done_cells(summaries_path) -> set:
    p = Path(summaries_path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        s = json.loads(line)
        if s.get("status") == "ok":
            out.add((s["plugin"], s["workload_key"], s["cache_state"]))
    return out


def _free(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def run_bench(plugins, workloads, cache_states, store, summaries_path, *, device="cuda", warmup=10,
              iters=30, pad=5, ceilings=None, check_max_ref_bytes=1 << 30, atol=1e-2, resume=True,
              log=print) -> list[dict]:
    summaries_path = Path(summaries_path)
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_cells(summaries_path) if resume else set()
    c = gpu_contention(visible_device_index(device), my_pids={os.getpid()}) or {}
    base_extra = {"run_id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
                  "host": socket.gethostname(), "device": device,
                  "gpu_util_at_start": c.get("utilization_pct"),
                  "other_pids_at_start": ",".join(str(p) for p in c.get("other_pids", [])),
                  "runner": "bench"}
    out = []

    def emit(s):
        out.append(s)
        with open(summaries_path, "a") as f:
            f.write(json.dumps(s) + "\n")
        if log:
            log(json.dumps(s))

    for plugin in plugins:
        for w in workloads:
            todo = [st for st in cache_states if (plugin.name, w.key(), st) not in done]
            for st in cache_states:
                if st not in todo:
                    out.append({"status": "skipped", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st})
            if not todo:
                continue
            if not isinstance(plugin, KernelPlugin) or not plugin.supports(w):
                for st in todo:
                    emit({"status": "unsupported", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
                          "reason": "executable plugin (use sweep)" if not isinstance(plugin, KernelPlugin)
                                    else "plugin.supports() is False"})
                continue
            _bench_workload(plugin, w, todo, store, emit, base_extra, device=device, warmup=warmup, iters=iters,
                            pad=pad, ceilings=ceilings, check_max_ref_bytes=check_max_ref_bytes, atol=atol)
    return out


def _bench_workload(plugin, w, states, store, emit, base_extra, *, device, warmup, iters, pad, ceilings,
                    check_max_ref_bytes, atol):
    inputs = None
    check = None
    try:
        inputs = plugin.build_inputs(w)
        # the float32 reference repeats K/V to every query head over the full cache capacity
        if 2 * w.B * w.L_kv * w.H_q * w.d * 4 <= check_max_ref_bytes:
            check = check_outputs(plugin, w, inputs, atol)
    except Exception as e:
        for st in states:
            emit({"status": "error", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
                  "error": f"{type(e).__name__}: {e}"[-2000:]})
        del inputs
        _free(device)
        return
    for i, st in enumerate(states):
        t0 = time.time()
        s = {"status": "ok", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
             "check_ok": None if check is None else bool(check["ok"])}
        rows = []
        try:
            if i == 0 and check is not None:
                rows += [_row(w, plugin.name, "check", "max_abs_diff", check["max_abs_diff"], note=check.get("error")),
                         _row(w, plugin.name, "check", "ok", float(bool(check["ok"])))]
            hooks = IterationHooks(device, st)
            before = (lambda: (hooks.between(), _sync(device))) if st == "cold" else None
            lat = measure_latency(lambda: plugin.run(inputs), warmup=warmup, iters=iters, before=before)
            rows += [_row(w, plugin.name, "latency", "median_s", lat["median_s"], unit="s", note=lat["timer"]),
                     _row(w, plugin.name, "latency", "min_s", lat["min_s"], unit="s", note=lat["timer"])]
            for _ in range(warmup):
                hooks.between()
                plugin.run(inputs)
            _sync(device)
            prof = profile_launches(plugin, inputs, device, iters, hooks=hooks, pad=pad)
            kt = prof.get("kernel_time_us_median")
            rows += profile_rows(w, plugin.name, prof)
            rows += analytic_rows(w, plugin.name, plugin.kv_heads_read(w), kt, ceilings, cache_state=st)
            s.update(kernel_time_us=kt, latency_us=lat["median_s"] * 1e6, launches=prof["launches_per_iter"],
                     iterations_dropped=prof.get("iterations_dropped"))
        except Exception as e:
            s["status"] = "error"
            s["error"] = f"{type(e).__name__}: {e}"[-2000:]
        if rows:
            store.write(rows, tag=f"{plugin.name}_{w.key()}_{st}", extra={**base_extra, "cache_state": st})
        s["elapsed_s"] = round(time.time() - t0, 3)
        emit(s)
    del inputs
    _free(device)
```

`cli.py` — add (next to the other `_cmd_*` functions):
```python
def _cmd_bench(args):
    from kernelscope.backends.realhw.batch import run_bench
    if bool(args.grid) == bool(args.workload):
        raise SystemExit("bench: give --grid (repeatable) or --workload")
    workloads = [w for g in args.grid for w in load_grid(g)] if args.grid else [Workload.from_key(args.workload)]
    registry = load_registry(args.registry)
    plugins = [registry.get(n, device=args.device) for n in args.plugins.split(",")]
    ceilings = json.loads(Path(args.ceilings).read_text()) if args.ceilings else None
    results = Path(args.results)
    run_bench(plugins, workloads, args.cache_state.split(","), ResultStore(results), results / "summaries.jsonl",
              device=args.device, warmup=args.warmup, iters=args.iters, pad=args.pad, ceilings=ceilings,
              check_max_ref_bytes=args.check_max_ref_mib << 20, atol=args.atol, resume=not args.no_resume)
```
and the parser block (after `p_sweep`):
```python
    p_bench = sub.add_parser("bench", help="in-process real-HW batch: every (workload, cache state) of each plugin in one process")
    p_bench.add_argument("--grid", action="append", help="YAML workload grid (repeatable)")
    p_bench.add_argument("--workload", help="single workload key instead of --grid")
    p_bench.add_argument("--plugins", required=True)
    p_bench.add_argument("--results", required=True)
    p_bench.add_argument("--cache-state", default="cold", help="cold, warm, or a comma list")
    p_bench.add_argument("--ceilings")
    p_bench.add_argument("--warmup", type=int, default=10)
    p_bench.add_argument("--iters", type=int, default=30)
    p_bench.add_argument("--pad", type=int, default=5)
    p_bench.add_argument("--check-max-ref-mib", type=int, default=1024, help="skip the reference check when its float32 K/V would exceed this")
    p_bench.add_argument("--atol", type=float, default=1e-2)
    p_bench.add_argument("--device", default="cuda")
    p_bench.add_argument("--registry", default=DEFAULT_REGISTRY)
    p_bench.add_argument("--no-resume", action="store_true")
    p_bench.set_defaults(func=_cmd_bench)
```
Also add `bench` to the module docstring's command list.

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_batch.py -q` → PASS; full CPU suite.

- [ ] **Step 5: GPU smoke test** — append to `tests/test_batch.py`:
```python
@pytest.mark.gpu
def test_bench_on_real_flash_kernels(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernelscope.plugins.builtin import REGISTRY
    ws = [Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128),
          Workload(phase="decode", B=4, L_q=1, L_kv=2048, H_q=32, H_kv=8, d=128, kv_lens=[2048, 100, 100, 100])]
    plugins = [REGISTRY.get(n, device="cuda") for n in ("fa2", "fd_s8_paged")]
    out = batch.run_bench(plugins, ws, ["cold"], ResultStore(tmp_path), tmp_path / "s.jsonl", log=None)
    assert all(s["status"] == "ok" and s["kernel_time_us"] > 0 and s["check_ok"] for s in out), out
```
Run (hygiene first): `$PY -m pytest -q -m gpu tests/test_batch.py` → PASS.
- [ ] **Step 6: Commit** — `git commit -m "Add in-process batch runner (kernelscope bench)" -- kernelscope/backends/realhw/batch.py kernelscope/cli.py tests/test_batch.py`

---

### Task 8: Report — cache-state key and simulator clock

**Depends on:** Task 6.

The report silently converts simulator cycles with A100's 1410 MHz, and it would join simulator results (cold by construction: a trace replay starts from an empty cache) with warm hardware results.

**Files:**
- Modify: `kernelscope/analysis/report.py`, `kernelscope/cli.py` (`_cmd_report` and the `p_rep` block only)
- Test: `tests/test_report.py`

**Interfaces:**
- Produces:
  - `report.ARCH_CLOCK_MHZ = {"SM80_A100": 1410.0, "SM89_RTX4090": 2520.0}`; `CLOCK_MHZ_A100` stays exported (Codex's tests import it)
  - `report.KEY = ["kernel", "workload_key", "cache_state"]`
  - `summarize(df, clock_mhz=None, assume_cache_state="warm") -> DataFrame` indexed by `KEY`. Hardware rows without a `cache_state` get `assume_cache_state`; `sim:*` rows are always `cold`. The clock comes from `clock_mhz`, else from a single `clock_mhz` column value of the sim rows, else from a single known `arch`; otherwise `ValueError` whose message contains `--clock-mhz`. No sim rows → no clock needed.
  - CLI `report`: `--clock-mhz` default `None`; new `--assume-cache-state {warm,cold}` default `warm`

- [ ] **Step 1: Update and extend the tests.** In `tests/test_report.py`:
  - in `_rows`, give every non-`sim:` row `"cache_state": "cold"` and every `sim:` row `"arch": "SM80_A100"` (build the dict, then add the key conditionally);
  - every `s.loc[("fa2", K)]` / `s.loc[("flashdecoding", K)]` becomes `s.loc[("fa2", K, "cold")]` / `s.loc[("flashdecoding", K, "cold")]`, and `list(s.index.names) == ["kernel", "workload_key", "cache_state"]`;
  - append:
```python
from kernelscope.analysis.report import ARCH_CLOCK_MHZ


def test_legacy_rows_without_cache_state_are_warm_and_never_joined_with_the_cold_simulator():
    legacy = DF.drop(columns=["cache_state"])
    s = summarize(legacy)
    assert ("fa2", K, "warm") in s.index and ("fa2", K, "cold") in s.index
    assert pd.isna(s.loc[("fa2", K, "cold"), "kernel_time_us"])
    assert pd.isna(s.loc[("fa2", K, "warm"), "sim_cycles_base"])


def test_assume_cache_state_lets_legacy_rows_join_the_simulator():
    s = summarize(DF.drop(columns=["cache_state"]), assume_cache_state="cold")
    assert s.loc[("fa2", K, "cold"), "sim_vs_kernel_time"] == pytest.approx(68482 / 1410.0 / 49.2)


def test_clock_is_taken_from_the_simulated_arch():
    df = DF.copy()
    df.loc[df.backend.str.startswith("sim:"), "arch"] = "SM89_RTX4090"
    s = summarize(df)
    assert s.loc[("fa2", K, "cold"), "sim_us_base"] == pytest.approx(68482 / ARCH_CLOCK_MHZ["SM89_RTX4090"])


def test_explicit_clock_wins():
    assert summarize(DF, clock_mhz=1000.0).loc[("fa2", K, "cold"), "sim_us_base"] == pytest.approx(68.482)


def test_unknown_or_mixed_arch_demands_an_explicit_clock():
    df = DF.copy()
    df.loc[df.backend.str.startswith("sim:"), "arch"] = "SM75_MYSTERY"
    with pytest.raises(ValueError, match="--clock-mhz"):
        summarize(df)


def test_hardware_only_results_need_no_clock():
    only_hw = DF[~DF.backend.str.startswith("sim:")].drop(columns=["arch"])
    assert summarize(only_hw).loc[("fa2", K, "cold"), "kernel_time_us"] == 49.2
```

- [ ] **Step 2: Run to verify failures** — `$PY -m pytest tests/test_report.py -q` → FAIL.

- [ ] **Step 3: Implement** in `report.py`:
```python
CLOCK_MHZ_A100 = 1410.0
ARCH_CLOCK_MHZ = {"SM80_A100": CLOCK_MHZ_A100, "SM89_RTX4090": 2520.0}
KEY = ["kernel", "workload_key", "cache_state"]


def _with_cache_state(df: pd.DataFrame, assume: str) -> pd.DataFrame:
    """Hardware rows carry the cache state they were measured in (rows written before
    --cache-state existed get ``assume``); a simulator replay always starts cold."""
    df = df.copy()
    if "cache_state" not in df.columns:
        df["cache_state"] = None
    df["cache_state"] = df["cache_state"].astype(object)
    df.loc[df.backend.str.startswith("sim:"), "cache_state"] = "cold"
    df["cache_state"] = df["cache_state"].fillna(assume)
    return df


def _sim_clock(df: pd.DataFrame, clock_mhz) -> float | None:
    if clock_mhz:
        return float(clock_mhz)
    sim = df[df.backend.str.startswith("sim:")]
    if sim.empty:
        return None
    if "clock_mhz" in sim.columns and sim["clock_mhz"].notna().all() and sim["clock_mhz"].nunique() == 1:
        return float(sim["clock_mhz"].iloc[0])
    archs = set(sim["arch"].dropna()) if "arch" in sim.columns else set()
    if len(archs) == 1 and next(iter(archs)) in ARCH_CLOCK_MHZ:
        return ARCH_CLOCK_MHZ[next(iter(archs))]
    raise ValueError(f"cannot infer the simulator core clock (arch={sorted(archs) or 'unknown'}); pass --clock-mhz")
```
`summarize(df, clock_mhz=None, assume_cache_state="warm")`: first `df = _with_cache_state(df, assume_cache_state)` and `clock = _sim_clock(df, clock_mhz)`; `_pick` groups by the new `KEY`; replace `clock_mhz` with `clock` in the `sim_us_base` computation. `out.index.names = KEY`. `verdict`/`with_verdicts` are unchanged.

`cli.py`: `p_rep.add_argument("--clock-mhz", type=float, default=None, help="core clock to convert sim cycles to µs (default: from the simulated arch)")`; `p_rep.add_argument("--assume-cache-state", choices=["warm", "cold"], default="warm", help="cache state for hardware rows recorded before --cache-state existed")`; `_cmd_report` passes both to `summarize`.

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_report.py tests/test_cli.py -q` → PASS; full CPU suite.
- [ ] **Step 5: Reproduce Codex's table under the new rules** (read-only on the shared results):
```bash
$PY -m kernelscope.cli report --assume-cache-state cold --results \
  /home/skkai/AI_Accelerator/kernelscope/results/hw_4090_simtrack/20260918-validation2 \
  /home/skkai/AI_Accelerator/kernelscope/results/hw_4090_simtrack/20260918-resume \
  /home/skkai/AI_Accelerator/kernelscope/results/sim_4090/20260918-validation2 \
  /home/skkai/AI_Accelerator/kernelscope/results/sim_4090/20260918-resume
```
Expected: `sim_us_base` equals cycles / 2520 (e.g. fa2 B1 L1K: 170012 / 2520 = 67.47) with no `--clock-mhz`. Paste the fa2/flashdecoding rows into your hand-off.
- [ ] **Step 6: Commit** — `git commit -m "Key the report by cache state and take the simulator clock from the arch" -- kernelscope/analysis/report.py kernelscope/cli.py tests/test_report.py`

---

### Task 9: Dispatch table (`kernelscope dispatch-table`)

**Depends on:** Task 6 (the `cache_state` column), Task 4 (plugin names).

**Files:**
- Create: `kernelscope/analysis/dispatch.py`
- Modify: `kernelscope/cli.py` (add `_cmd_dispatch_table` and a `p_disp` parser block only)
- Test: `tests/test_dispatch.py` (new)

**Interfaces:**
- Produces:
  - `dispatch.FAMILIES = {"dense": {"heuristic": "flashdecoding", "members": r"^(fa2|flashdecoding|fd_s\d+|sdpa_flash)$"}, "paged": {"heuristic": "flashdecoding_paged", "members": r"^(fa2_paged|flashdecoding_paged|fd_s\d+_paged)$"}}`
  - `dispatch_table(df, family="dense", cache_state="cold") -> DataFrame` with columns `workload_key, B, L_kv, H_q, H_kv, ragged, lens, n_variants, best_kernel, best_us, heuristic_us, heuristic_regret, fa2_us, fa2_regret`, sorted by `heuristic_regret` descending. Uses median `profile/kernel_time_us` per (workload, kernel). Regret = `t / best - 1`. Rows lacking a `cache_state` column count as `warm`. `fa2` column is `fa2` for dense and `fa2_paged` for paged.
  - `regret_summary(table) -> dict` with keys `cells, median_regret, max_regret, cells_over_10pct, worst_key` (heuristic regret, NaN-safe)
  - CLI: `kernelscope dispatch-table --results DIR [DIR ...] --family dense|paged --cache-state cold|warm [--out table.csv]` prints the summary and the 10 worst rows

- [ ] **Step 1: Write the failing tests** — create `tests/test_dispatch.py`:

```python
import math

import pandas as pd
import pytest

from kernelscope.analysis.dispatch import dispatch_table, regret_summary

U = "decode_B1_Lq1_Lkv8192_Hq32_Hkv8_d128_float16_causal"
R = "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal"


def _kt(key, kernel, us, state="cold"):
    return {"workload_key": key, "kernel": kernel, "backend": "profile", "metric": "kernel_time_us",
            "unit": "us", "value": us, "launch_idx": 0, "note": None, "cache_state": state}


DF = pd.DataFrame([
    _kt(U, "fa2", 466.0), _kt(U, "flashdecoding", 63.4), _kt(U, "fd_s8", 60.1), _kt(U, "fd_s16", 61.1),
    _kt(R, "fa2", 3744.0), _kt(R, "flashdecoding", 3744.0), _kt(R, "fd_s4", 500.0), _kt(R, "fd_s8", 500.0),
    _kt(R, "fd_s8_paged", 400.0),                          # other family: ignored for dense
    _kt(U, "fd_s8", 20.0, state="warm"),                   # other cache state: ignored
])


def test_best_variant_and_heuristic_regret_per_workload():
    t = dispatch_table(DF, family="dense", cache_state="cold").set_index("workload_key")
    assert t.loc[U, "best_kernel"] == "fd_s8" and t.loc[U, "best_us"] == 60.1
    assert t.loc[U, "heuristic_regret"] == pytest.approx(63.4 / 60.1 - 1)
    assert t.loc[R, "heuristic_regret"] == pytest.approx(3744 / 500 - 1)
    assert t.loc[R, "fa2_regret"] == pytest.approx(3744 / 500 - 1)
    assert bool(t.loc[R, "ragged"]) and t.loc[R, "B"] == 32 and t.loc[R, "L_kv"] == 32768
    assert t.loc[R, "lens"] == "32768x2+1024x30"
    assert t.loc[U, "n_variants"] == 4


def test_table_is_sorted_by_regret_and_summary_reports_the_worst_cell():
    t = dispatch_table(DF)
    assert t.iloc[0]["workload_key"] == R
    s = regret_summary(t)
    assert s["cells"] == 2 and s["worst_key"] == R and s["cells_over_10pct"] == 1
    assert s["max_regret"] == pytest.approx(3744 / 500 - 1)


def test_missing_heuristic_gives_nan_regret():
    t = dispatch_table(DF[DF.kernel != "flashdecoding"])
    assert t["heuristic_regret"].isna().all()


def test_rows_without_cache_state_count_as_warm():
    legacy = DF.drop(columns=["cache_state"])
    assert dispatch_table(legacy, cache_state="cold").empty
    assert len(dispatch_table(legacy, cache_state="warm")) == 2
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_dispatch.py -q` → FAIL.

- [ ] **Step 3: Implement `kernelscope/analysis/dispatch.py`:**

```python
"""Fastest interchangeable kernel variant per workload, and what the library heuristic loses.

Variants in one family compute the same attention on the same cache layout, so a dispatcher may
swap them freely: dense (fa2 = num_splits 1, flashdecoding = the library heuristic, fd_s{N},
sdpa_flash) and paged (the same on a block_table cache). Input: real-HW rows with a
cache_state column (rows without one were measured warm).
"""
import math
import re

import pandas as pd

from kernelscope.workload import Workload, format_lens

FAMILIES = {
    "dense": {"heuristic": "flashdecoding", "fa2": "fa2", "members": r"^(fa2|flashdecoding|fd_s\d+|sdpa_flash)$"},
    "paged": {"heuristic": "flashdecoding_paged", "fa2": "fa2_paged",
              "members": r"^(fa2_paged|flashdecoding_paged|fd_s\d+_paged)$"},
}
COLUMNS = ["workload_key", "B", "L_kv", "H_q", "H_kv", "ragged", "lens", "n_variants", "best_kernel", "best_us",
           "heuristic_us", "heuristic_regret", "fa2_us", "fa2_regret"]


def dispatch_table(df: pd.DataFrame, family: str = "dense", cache_state: str = "cold") -> pd.DataFrame:
    fam = FAMILIES[family]
    states = df["cache_state"].fillna("warm") if "cache_state" in df.columns else pd.Series("warm", index=df.index)
    sel = df[(df.backend == "profile") & (df.metric == "kernel_time_us") & (states == cache_state)
             & df.kernel.str.match(fam["members"])]
    if sel.empty:
        return pd.DataFrame(columns=COLUMNS)
    t = sel.groupby(["workload_key", "kernel"])["value"].median().unstack("kernel")
    rows = []
    for key, r in t.iterrows():
        r = r.dropna()
        w = Workload.from_key(key)
        best_kernel, best = r.idxmin(), r.min()
        h = r.get(fam["heuristic"], math.nan)
        f = r.get(fam["fa2"], math.nan)
        rows.append({"workload_key": key, "B": w.B, "L_kv": w.L_kv, "H_q": w.H_q, "H_kv": w.H_kv,
                     "ragged": w.is_ragged, "lens": format_lens(w.lens()), "n_variants": len(r),
                     "best_kernel": best_kernel, "best_us": best, "heuristic_us": h,
                     "heuristic_regret": h / best - 1, "fa2_us": f, "fa2_regret": f / best - 1})
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("heuristic_regret", ascending=False, na_position="last",
                                                            ignore_index=True)


def regret_summary(table: pd.DataFrame) -> dict:
    r = table["heuristic_regret"].dropna()
    if r.empty:
        return {"cells": len(table), "median_regret": math.nan, "max_regret": math.nan,
                "cells_over_10pct": 0, "worst_key": None}
    return {"cells": len(table), "median_regret": float(r.median()), "max_regret": float(r.max()),
            "cells_over_10pct": int((r > 0.10).sum()),
            "worst_key": table.loc[r.idxmax(), "workload_key"]}
```
(`re` is unused if you use `str.match`; do not import it.)

`cli.py`:
```python
def _cmd_dispatch_table(args):
    import pandas as pd
    from kernelscope.analysis.dispatch import dispatch_table, regret_summary
    df = pd.concat([ResultStore(r).load() for r in args.results], ignore_index=True)
    t = dispatch_table(df, family=args.family, cache_state=args.cache_state)
    print(json.dumps(regret_summary(t), indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.float_format", "{:.4g}".format):
        print(t.head(10).to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(args.out, index=False)
```
parser block:
```python
    p_disp = sub.add_parser("dispatch-table", help="best interchangeable variant per workload and the library heuristic's regret")
    p_disp.add_argument("--results", nargs="+", required=True)
    p_disp.add_argument("--family", choices=["dense", "paged"], default="dense")
    p_disp.add_argument("--cache-state", choices=["cold", "warm"], default="cold")
    p_disp.add_argument("--out")
    p_disp.set_defaults(func=_cmd_dispatch_table)
```

- [ ] **Step 4: Run** — `$PY -m pytest tests/test_dispatch.py -q` → PASS; full CPU suite.
- [ ] **Step 5: Commit** — `git commit -m "Add dispatch-table analysis of interchangeable kernel variants" -- kernelscope/analysis/dispatch.py kernelscope/cli.py tests/test_dispatch.py`

---

### Task 10: Measured machine spec (`kernelscope machine`)

**Depends on:** Task 1 (`props_from_torch`), Task 11 (`block_placement`).

The surrogate model's machine parameters must be measured, not taken from a datasheet. Probe data (spec F5, F6) also showed that a naïve Triton streaming kernel lets the compiler hoist loads out of the repetition loop when the per-program range is a single chunk, producing fake 10–47 TB/s numbers; rotating the chunk index by the repetition index prevents that.

**Files:**
- Create: `kernelscope/bench/stream.py`, `kernelscope/bench/machine.py`
- Modify: `kernelscope/cli.py` (add `_cmd_machine` and a `p_mach` block only)
- Test: `tests/test_machine.py` (new; CPU tests + GPU tests)

**Interfaces:**
- Consumes: `props_from_torch` (Task 1); `measure_ceilings` (existing, `kernelscope/bench/ceilings.py`); `placement.block_placement(ctas_per_sm, device)` (Task 11)
- Produces:
  - `stream.num_chunks(n_elems: int, G: int, block: int = 1024) -> int` (raises `ValueError` if < 2 — a single chunk lets the compiler hoist the load)
  - `stream.stream_gbps(nbytes: int, G: int, *, target_bytes: int = 2 << 30, num_warps: int = 4, iters: int = 10, device="cuda") -> float`
  - `machine.hit_fraction(gbps, dram_gbps, l2_gbps) -> float` in [0, 1] from `1/bw = h/l2 + (1-h)/dram`
  - `machine.measure_machine(device="cuda", *, l2_curve_mib=(16, 32, 48, 64, 72, 80, 96, 128, 192, 256, 512), stream=stream_gbps, tc_tflops=None, clocks=None, placement=None) -> dict` with keys `name, cc, n_sm, max_threads_sm, max_ctas_sm, regs_sm, smem_sm, reserved_smem_per_block, l2_bytes, dram_gbps, l2_gbps, cta_dram_gbps, cta_l2_gbps, l2_hit_curve (list of {mib, gbps, hit}), tc_tflops, clock_mhz, mem_clock_mhz, block_placement, provenance`
  - CLI: `kernelscope machine --out machines/rtx4090.json [--no-placement]`

- [ ] **Step 1: Write the failing CPU tests** — create `tests/test_machine.py`:

```python
import pytest

from kernelscope.bench import machine
from kernelscope.bench.stream import num_chunks


def test_num_chunks_refuses_a_single_chunk_per_program():
    assert num_chunks(1 << 20, G=128) == 8                   # 1 Mi elems / (1024 * 128)
    with pytest.raises(ValueError, match="hoist"):
        num_chunks(1024 * 128, G=128)


def test_hit_fraction_inverts_the_two_level_bandwidth_mix():
    assert machine.hit_fraction(1000.0, 1000.0, 5000.0) == 0.0
    assert machine.hit_fraction(5000.0, 1000.0, 5000.0) == 1.0
    bw = 1 / (0.5 / 5000.0 + 0.5 / 1000.0)
    assert machine.hit_fraction(bw, 1000.0, 5000.0) == pytest.approx(0.5)
    assert machine.hit_fraction(9000.0, 1000.0, 5000.0) == 1.0          # clipped


def test_measure_machine_assembles_the_spec_from_injected_measurements(monkeypatch):
    from kernelscope.backends.realhw.kprofile import RTX4090_PROPS
    monkeypatch.setattr(machine, "props_from_torch", lambda device: dict(RTX4090_PROPS))

    def fake_stream(nbytes, G, **kw):
        if G == 1:
            return 50.0 if nbytes <= 8 << 20 else 27.0
        return 4800.0 if nbytes <= 36 << 20 else (1500.0 if nbytes <= 128 << 20 else 958.0)

    spec = machine.measure_machine("cuda", l2_curve_mib=(16, 512), stream=fake_stream,
                                   tc_tflops=lambda: 150.0, clocks=lambda: (2520.0, 10501.0), placement=None)
    assert spec["n_sm"] == 128 and spec["max_ctas_sm"] == 24 and spec["smem_sm"] == 102400
    assert spec["dram_gbps"] == 958.0 and spec["l2_gbps"] == 4800.0
    assert spec["cta_dram_gbps"] == 27.0 and spec["cta_l2_gbps"] == 50.0
    assert [p["mib"] for p in spec["l2_hit_curve"]] == [16, 512]
    assert spec["l2_hit_curve"][0]["hit"] == 1.0 and spec["l2_hit_curve"][1]["hit"] == 0.0
    assert spec["tc_tflops"] == 150.0 and spec["clock_mhz"] == 2520.0
    assert spec["block_placement"] is None
    assert "date" in spec["provenance"]
```

- [ ] **Step 2: Run to verify it fails** — `$PY -m pytest tests/test_machine.py -q` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `kernelscope/bench/stream.py`:**

```python
"""Triton streaming-read kernel with an explicit grid, for bandwidth measurements.

Each of G programs reads BLOCK-element chunks at stride BLOCK*G. Repetition r visits chunk
(c + r) % num_chunks, so no load is invariant across repetitions: with a single chunk per
program the compiler could hoist the load out of the repetition loop and report bandwidth the
memory system never delivered (the 1-32 MiB outliers in the 2026-09-19 probes).
"""
import statistics

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def kernelscope_stream_read(x_ptr, out_ptr, n, chunks, reps, BLOCK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros((BLOCK,), tl.float32)
    for r in range(reps):
        for c in range(chunks):
            cc = (c + r) % chunks
            offs = cc * BLOCK * G + pid * BLOCK + tl.arange(0, BLOCK)
            acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def num_chunks(n_elems: int, G: int, block: int = BLOCK) -> int:
    c = -(-n_elems // (block * G))
    if c < 2:
        raise ValueError(f"{n_elems} elements over G={G} programs is one chunk per program; "
                         "the compiler could hoist the load out of the repetition loop — use a larger buffer or smaller G")
    return c


def stream_gbps(nbytes: int, G: int, *, target_bytes: int = 2 << 30, num_warps: int = 4, iters: int = 10,
                device="cuda") -> float:
    """Achieved read bandwidth (GB/s) of G programs streaming an fp16 buffer of ``nbytes``."""
    n = nbytes // 2
    chunks = num_chunks(n, G)
    reps = max(1, round(target_bytes / nbytes))
    x = torch.randn(n, device=device, dtype=torch.float16)
    out = torch.empty(G, device=device, dtype=torch.float32)

    def launch():
        kernelscope_stream_read[(G,)](x, out, n, chunks, reps, BLOCK=BLOCK, G=G, num_warps=num_warps)

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 1e3)
    return nbytes * reps / statistics.median(times) / 1e9
```

- [ ] **Step 4: Implement `kernelscope/bench/machine.py`:**

```python
"""Measured machine parameters for the surrogate performance model (machines/<gpu>.json).

Everything the model treats as a machine constant is measured here, except the per-compute-
capability occupancy limits that no API reports (kprofile.MAX_BLOCKS_PER_SM).
"""
import subprocess
import time

from kernelscope.backends.realhw.kprofile import props_from_torch
from kernelscope.bench.stream import stream_gbps


def hit_fraction(gbps: float, dram_gbps: float, l2_gbps: float) -> float:
    """Share of bytes served by L2 implied by an achieved bandwidth: 1/bw = h/l2 + (1-h)/dram."""
    h = (1 / gbps - 1 / dram_gbps) / (1 / l2_gbps - 1 / dram_gbps)
    return min(1.0, max(0.0, h))


def _clocks() -> tuple[float, float]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.splitlines()[0]
    sm, mem = (float(x) for x in out.split(","))
    return sm, mem


def _tc_tflops(device) -> float:
    from kernelscope.bench.ceilings import measure_ceilings
    return measure_ceilings(device=device)["fp16_matmul_tflops"]


def measure_machine(device="cuda", *, l2_curve_mib=(16, 32, 48, 64, 72, 80, 96, 128, 192, 256, 512),
                    stream=stream_gbps, tc_tflops=None, clocks=None, placement=None) -> dict:
    p = props_from_torch(device)
    n = p["num_sms"]
    dram = max(stream(1 << 30, G) for G in (2 * n, 4 * n))
    l2 = max(stream(p["l2_bytes"] // 2, G) for G in (n, 2 * n))
    cta_dram = stream(1 << 30, 1, target_bytes=256 << 20)
    cta_l2 = stream(8 << 20, 1, target_bytes=256 << 20)
    curve = []
    for mib in l2_curve_mib:
        g = stream(mib << 20, 2 * n)
        curve.append({"mib": mib, "gbps": g, "hit": hit_fraction(g, dram, l2)})
    sm_clock, mem_clock = (clocks or _clocks)()
    try:
        import torch
        prov = {"torch": torch.__version__, "cuda": torch.version.cuda}
    except ImportError:
        prov = {}
    return {
        "name": p["name"], "cc": p["cc"], "n_sm": n, "max_threads_sm": p["max_threads_per_sm"],
        "max_ctas_sm": p["max_blocks_per_sm"], "regs_sm": p["regs_per_sm"], "smem_sm": p["smem_per_sm"],
        "reserved_smem_per_block": p["reserved_smem_per_block"], "l2_bytes": p["l2_bytes"],
        "dram_gbps": dram, "l2_gbps": l2, "cta_dram_gbps": cta_dram, "cta_l2_gbps": cta_l2,
        "l2_hit_curve": curve,
        "tc_tflops": (tc_tflops or (lambda: _tc_tflops(device)))(),
        "clock_mhz": sm_clock, "mem_clock_mhz": mem_clock,
        "block_placement": placement() if placement else None,
        "provenance": {"date": time.strftime("%Y-%m-%d %H:%M:%S"), **prov},
    }
```

`cli.py`:
```python
def _cmd_machine(args):
    from kernelscope.bench.machine import measure_machine
    placement = None
    if not args.no_placement:
        from kernelscope.bench.placement import block_placement
        placement = lambda: block_placement(ctas_per_sm=2, device=args.device)  # noqa: E731
    spec = measure_machine(args.device, placement=placement)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(spec, indent=2))
    print(json.dumps({k: v for k, v in spec.items() if k not in ("l2_hit_curve", "block_placement")}, indent=2))
```
parser block:
```python
    p_mach = sub.add_parser("machine", help="measure the machine spec used by the performance model")
    p_mach.add_argument("--out", default="machines/rtx4090.json")
    p_mach.add_argument("--device", default="cuda")
    p_mach.add_argument("--no-placement", action="store_true", help="skip the CUDA-extension block-placement probe")
    p_mach.set_defaults(func=_cmd_machine)
```

- [ ] **Step 5: Run CPU tests** — `$PY -m pytest tests/test_machine.py -q -m "not gpu"` → PASS.

- [ ] **Step 6: GPU acceptance** — append to `tests/test_machine.py`:
```python
@pytest.mark.gpu
def test_stream_bandwidths_on_the_4090():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernelscope.bench.stream import stream_gbps
    dram = stream_gbps(1 << 30, 512)
    l2 = stream_gbps(36 << 20, 128)
    assert 800 < dram < 1100, dram            # rated 1008 GB/s; probes measured 958
    assert l2 > 2.5 * dram, (l2, dram)
    assert stream_gbps(512 << 20, 256) < 1.3 * dram
```
Run (hygiene first): `$PY -m pytest -q -m gpu tests/test_machine.py` → PASS. Report the three numbers.
- [ ] **Step 7: Commit** — `git commit -m "Add measured machine spec (kernelscope machine)" -- kernelscope/bench/stream.py kernelscope/bench/machine.py kernelscope/cli.py tests/test_machine.py`

---

### Task 11: CUDA bench extension — SM blocker and block placement

**Depends on:** nothing in this plan (runs in wave 1).

MPS does not start on this GeForce (spec F20). An SM blocker — a kernel whose CTAs request almost all of an SM's shared memory and spin without touching memory — makes real SM-count what-ifs possible. An `%smid` probe answers which SM runs which block (spec F19: two long CTAs appeared to share an SM).

**Files:**
- Create: `kernelscope/bench/cuda_ext.py`, `kernelscope/bench/sm_blocker.py`, `kernelscope/bench/placement.py`
- Test: `tests/test_cuda_ext.py` (new; one CPU test, the rest GPU)

**Interfaces:**
- Produces:
  - `cuda_ext.load_ext()` → cached module with `launch_blocker(n_ctas, smem_bytes, spin_cycles, stream_ptr)` and `smid_probe(n_ctas, threads, smem_bytes, spin_cycles) -> torch.Tensor[int32]`
  - `sm_blocker.SMBlocker(device="cuda")` with `.smem_bytes`, `.cycles_per_us`, `.block(n_sms, duration_us)`, `.time_blocked(fn, n_sms, iters=10, warmup=3) -> float` (median µs of `fn` on the current stream while `n_sms` SMs are blocked)
  - `placement.smem_for_ctas_per_sm(k: int, smem_per_sm: int, reserved: int) -> int` (dynamic smem so that exactly k CTAs fit per SM)
  - `placement.block_placement(ctas_per_sm=2, device="cuda") -> dict` with keys `ctas_per_sm, n_ctas, distinct_sms, max_ctas_on_one_sm, adjacent_pair_same_sm, first_wave_distinct, smid`

- [ ] **Step 1: Write the CPU test** — create `tests/test_cuda_ext.py`:

```python
import pytest

from kernelscope.bench.placement import smem_for_ctas_per_sm


@pytest.mark.parametrize("smem_per_sm", [102400, 167936])
@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_smem_request_admits_exactly_k_ctas_per_sm(k, smem_per_sm):
    s = smem_for_ctas_per_sm(k, smem_per_sm, reserved=1024)
    assert smem_per_sm // (s + 1024) == k
```

- [ ] **Step 2: Run to verify it fails**, implement `kernelscope/bench/placement.py` helper first:

```python
"""Which SM runs which block: record %smid for a grid that exactly fills every SM."""
from collections import Counter


def smem_for_ctas_per_sm(k: int, smem_per_sm: int, reserved: int) -> int:
    """Dynamic shared memory per CTA such that exactly k CTAs fit on one SM (256-byte aligned)."""
    s = (smem_per_sm // k - reserved) // 256 * 256
    while smem_per_sm // (s + reserved) > k:
        s += 256
    return s


def block_placement(ctas_per_sm: int = 2, device: str = "cuda", threads: int = 128, spin_cycles: int = 2_000_000) -> dict:
    """Launch n_sm * ctas_per_sm blocks that each record %smid and then spin long enough to be co-resident."""
    from kernelscope.backends.realhw.kprofile import props_from_torch
    from kernelscope.bench.cuda_ext import load_ext
    p = props_from_torch(device)
    n = p["num_sms"] * ctas_per_sm
    smem = smem_for_ctas_per_sm(ctas_per_sm, p["smem_per_sm"], p["reserved_smem_per_block"])
    smid = load_ext().smid_probe(n, threads, smem, spin_cycles).cpu().tolist()
    counts = Counter(smid)
    pairs = [smid[i] == smid[i + 1] for i in range(0, n - 1, 2)]
    return {"ctas_per_sm": ctas_per_sm, "n_ctas": n, "distinct_sms": len(counts),
            "max_ctas_on_one_sm": max(counts.values()),
            "adjacent_pair_same_sm": sum(pairs) / len(pairs),
            "first_wave_distinct": len(set(smid[: p["num_sms"]])), "smid": smid}
```
Run `$PY -m pytest tests/test_cuda_ext.py -q -m "not gpu"` → PASS.

- [ ] **Step 3: Implement `kernelscope/bench/cuda_ext.py`:**

```python
"""Small CUDA kernels Triton cannot express: an SM blocker and an %smid probe.

Built once with torch.utils.cpp_extension.load_inline and the CUDA 12.9 toolchain of the
accelsim-build conda env: /usr/bin/nvcc is CUDA 10.1 and cannot target sm_89.
"""
import os
from pathlib import Path

CUDA_HOME = os.environ.get("KERNELSCOPE_CUDA_HOME", "/home/skkai/miniforge3/envs/accelsim-build")

CPP = """
void launch_blocker(int64_t n_ctas, int64_t smem_bytes, int64_t spin_cycles, int64_t stream_ptr);
torch::Tensor smid_probe(int64_t n_ctas, int64_t threads, int64_t smem_bytes, int64_t spin_cycles);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void __launch_bounds__(32) ks_sm_blocker(long long spin_cycles) {
    extern __shared__ char smem[];
    if (threadIdx.x == 0) smem[0] = 0;
    __syncthreads();
    long long start = clock64();
    while (clock64() - start < spin_cycles) { }
}

__global__ void ks_smid_probe(int* out, long long spin_cycles) {
    extern __shared__ char smem[];
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        out[blockIdx.x] = (int)smid;
        smem[0] = 0;
    }
    __syncthreads();
    long long start = clock64();
    while (clock64() - start < spin_cycles) { }
}

void launch_blocker(int64_t n_ctas, int64_t smem_bytes, int64_t spin_cycles, int64_t stream_ptr) {
    if (n_ctas <= 0) return;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaFuncSetAttribute(ks_sm_blocker, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    ks_sm_blocker<<<(unsigned)n_ctas, 32, (size_t)smem_bytes, stream>>>((long long)spin_cycles);
}

torch::Tensor smid_probe(int64_t n_ctas, int64_t threads, int64_t smem_bytes, int64_t spin_cycles) {
    auto out = torch::full({n_ctas}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA));
    cudaFuncSetAttribute(ks_smid_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    auto stream = c10::cuda::getCurrentCUDAStream();
    ks_smid_probe<<<(unsigned)n_ctas, (unsigned)threads, (size_t)smem_bytes, stream>>>(
        out.data_ptr<int>(), (long long)spin_cycles);
    return out;
}
"""

_EXT = None


def load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    import torch.utils.cpp_extension as ce
    major, minor = torch.cuda.get_device_capability()
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    ce.CUDA_HOME = CUDA_HOME                      # cpp_extension caches CUDA_HOME at import time
    build = Path.home() / ".cache" / "kernelscope" / f"cuda_ext_sm{major}{minor}"
    build.mkdir(parents=True, exist_ok=True)
    _EXT = ce.load_inline(
        name="kernelscope_cuda_ext", cpp_sources=CPP, cuda_sources=CUDA,
        functions=["launch_blocker", "smid_probe"],
        extra_cuda_cflags=["-ccbin", f"{CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++", f"-arch=sm_{major}{minor}"],
        build_directory=str(build), verbose=False,
    )
    return _EXT
```

- [ ] **Step 4: Implement `kernelscope/bench/sm_blocker.py`:**

```python
"""Real-hardware SM-count what-if: occupy n SMs while a target kernel runs.

Each blocker CTA requests the per-block shared-memory maximum, so exactly one fits on an SM and
no other kernel's CTA needing more than ~1 KiB of shared memory can share that SM. Blocker
CTAs spin on clock64() and touch no global memory, so they take SMs away without taking
bandwidth. Caveat: tiny kernels (e.g. split-KV's combine kernel, <= 640 B of shared memory)
may still land on a blocked SM; verify placement when that matters.
"""
import statistics

import torch

from kernelscope.bench.cuda_ext import load_ext


class SMBlocker:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ext = load_ext()
        p = torch.cuda.get_device_properties(device)
        self.n_sms = p.multi_processor_count
        self.smem_bytes = getattr(p, "shared_memory_per_block_optin", 101376)
        self.stream = torch.cuda.Stream(device)
        self.cycles_per_us = self._calibrate()

    def _calibrate(self, cycles: int = 20_000_000) -> float:
        """clock64() ticks per microsecond at the current clocks (runs the blocker on one SM)."""
        rate = None
        for _ in range(3):                         # the first runs also ramp the clocks up
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self.stream):
                s.record()
                self.ext.launch_blocker(1, self.smem_bytes, cycles, self.stream.cuda_stream)
                e.record()
            e.synchronize()
            rate = cycles / (s.elapsed_time(e) * 1e3)
        return rate

    def block(self, n_sms: int, duration_us: float) -> None:
        """Occupy n_sms SMs for about duration_us, starting now (asynchronous, side stream)."""
        if not 0 <= n_sms < self.n_sms:
            raise ValueError(f"n_sms must be in [0, {self.n_sms}), got {n_sms}")
        self.ext.launch_blocker(n_sms, self.smem_bytes, int(duration_us * self.cycles_per_us), self.stream.cuda_stream)

    def time_blocked(self, fn, n_sms: int, iters: int = 10, warmup: int = 3) -> float:
        """Median CUDA-event time (µs) of fn on the current stream while n_sms SMs are blocked."""
        cur = torch.cuda.current_stream(self.device)
        t0 = []
        for _ in range(2):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(cur); fn(); e.record(cur); e.synchronize()
            t0.append(s.elapsed_time(e) * 1e3)
        spin_us = 4 * min(t0) * self.n_sms / max(1, self.n_sms - n_sms) + 200.0
        times = []
        for i in range(warmup + iters):
            self.block(n_sms, spin_us)
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(cur); fn(); e.record(cur)
            torch.cuda.synchronize(self.device)
            if i >= warmup:
                times.append(s.elapsed_time(e) * 1e3)
        return statistics.median(times)
```

- [ ] **Step 5: Write the GPU tests** — append to `tests/test_cuda_ext.py`:
```python
@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    return torch


@pytest.mark.gpu
def test_block_placement_fills_every_sm_exactly(cuda):
    from kernelscope.bench.placement import block_placement
    r = block_placement(ctas_per_sm=2)
    n = cuda.cuda.get_device_properties(0).multi_processor_count
    assert r["n_ctas"] == 2 * n
    assert r["distinct_sms"] == n and r["max_ctas_on_one_sm"] == 2
    assert all(0 <= s < n for s in r["smid"])


@pytest.mark.gpu
def test_blocking_half_the_sms_slows_a_gemm_but_not_idle_time(cuda):
    from kernelscope.bench.sm_blocker import SMBlocker
    torch = cuda
    b = SMBlocker()
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    fn = lambda: a @ a  # noqa: E731
    free = b.time_blocked(fn, 0)
    half = b.time_blocked(fn, b.n_sms // 2)
    assert 1.6 < half / free < 2.4, (free, half)       # probe: 1.83x at 64 of 128 SMs
```
Run (hygiene first): `$PY -m pytest -q -m gpu tests/test_cuda_ext.py` → PASS (first run builds the extension, ~1 min). Report `adjacent_pair_same_sm`, `first_wave_distinct` and the GEMM ratio in your hand-off — these are new facts for the spec.
- [ ] **Step 6: Full CPU suite, then commit** — `git commit -m "Add CUDA SM blocker and block-placement probe" -- kernelscope/bench/cuda_ext.py kernelscope/bench/sm_blocker.py kernelscope/bench/placement.py tests/test_cuda_ext.py`

---

### Task 12: Measurement campaign

**Depends on:** all previous tasks. GPU-heavy; expect roughly 1 hour of GPU time.

**Files:**
- Create: `grids/dispatch_s1.yaml`, `grids/dispatch_s2.yaml`, `grids/ragged_s1.yaml`, `grids/ragged_s2.yaml`, `machines/rtx4090.json` (generated), `docs/plan/2026-09-19-p0-campaign.md` (results note)
- Modify: `README.md` (usage of `bench`, `machine`, `dispatch-table`, `--cache-state`), `docs/STATUS.md` (append one dated entry)
- Output (not committed): `/home/skkai/AI_Accelerator/kernelscope/results/hw_4090/<run>/`

- [ ] **Step 1: Write the grids.**

`grids/dispatch_s1.yaml`:
```yaml
# Uniform decode, Llama-3.1-8B / DeepSeek-R1-Distill-Llama-8B head geometry (spec shape S1)
phase: decode
B: [1, 2, 4, 8, 16, 32, 64]
L: [512, 1024, 2048, 4096, 8192, 16384, 32768]
H_q: 32
H_kv: 8
d: 128
```
`grids/dispatch_s2.yaml`: same with `H_q: 28`, `H_kv: 4` and the comment `DeepSeek-R1-Distill-Qwen-7B head geometry (spec shape S2)`.
`grids/ragged_s1.yaml`:
```yaml
# Ragged decode: n_long long sequences + (B - n_long) short ones; B=26 is the S1 threshold
# where flash-attn's heuristic stops splitting (0.8 * 2 * 128 SMs / H_kv 8 = 25.6) — spec F15
phase: decode
H_q: 32
H_kv: 8
d: 128
ragged:
  B: [16, 24, 26, 32, 48, 64]
  n_long: [1, 2, 4]
  L_long: [8192, 16384, 32768]
  L_short: [512, 1024, 2048]
```
`grids/ragged_s2.yaml`: same with `H_q: 28`, `H_kv: 4`, `B: [16, 32, 48, 52, 64]` and the comment's threshold for H_kv 4 (51.2).

Verify: `$PY -c "from kernelscope.cli import load_grid; [print(g, len(load_grid('grids/'+g))) for g in ['dispatch_s1.yaml','dispatch_s2.yaml','ragged_s1.yaml','ragged_s2.yaml']]"` → 49, 49, 162, 135.

- [ ] **Step 2: Machine spec** (hygiene first): `$PY -m kernelscope.cli machine --out machines/rtx4090.json`. Sanity: `dram_gbps` 800–1100, `l2_gbps` > 2.5 × dram, hit curve decreasing from ≈1 at 16 MiB to ≈0 at 512 MiB, `block_placement.distinct_sms == 128`.

- [ ] **Step 3: Measurements.** Set `R=/home/skkai/AI_Accelerator/kernelscope/results/hw_4090`, `DENSE=fa2,flashdecoding,fd_s2,fd_s4,fd_s8,fd_s16,fd_s32,fd_s64,fd_s128`, `PAGED=fa2_paged,flashdecoding_paged,fd_s2_paged,fd_s4_paged,fd_s8_paged,fd_s16_paged,fd_s32_paged,fd_s64_paged,fd_s128_paged`. Check GPU hygiene before each command; each command resumes if interrupted.
```bash
$PY -m kernelscope.cli bench --grid grids/dispatch_s1.yaml --plugins $DENSE --results $R/uniform_s1_dense --cache-state cold,warm
$PY -m kernelscope.cli bench --grid grids/dispatch_s1.yaml --plugins $PAGED --results $R/uniform_s1_paged --cache-state cold,warm
$PY -m kernelscope.cli bench --grid grids/dispatch_s2.yaml --plugins $DENSE --results $R/uniform_s2_dense --cache-state cold
$PY -m kernelscope.cli bench --grid grids/dispatch_s2.yaml --plugins $PAGED --results $R/uniform_s2_paged --cache-state cold
$PY -m kernelscope.cli bench --grid grids/ragged_s1.yaml --plugins $DENSE --results $R/ragged_s1_dense --cache-state cold
$PY -m kernelscope.cli bench --grid grids/ragged_s1.yaml --plugins $PAGED --results $R/ragged_s1_paged --cache-state cold
$PY -m kernelscope.cli bench --grid grids/ragged_s2.yaml --plugins $PAGED --results $R/ragged_s2_paged --cache-state cold
# the seven cells of Codex's sim validation, cold and warm (Codex compares against the cold rows)
$PY -m kernelscope.cli bench --grid grids/w1_min_B1.yaml --grid grids/w1_min_B16.yaml --plugins fa2,flashdecoding --results $R/codex_validation --cache-state cold,warm
```
If a cell errors with CUDA OOM, leave it recorded as an error and continue; list such cells in the results note. Before the Codex-validation command, open both `w1_min_*` grids and confirm they contain the six cells in `docs/setup/accelsim_4090_gate_report.md` §5 (B1 L1K, B1 L8K, B16 L1K); if a grid holds more cells that is fine.

- [ ] **Step 4: Acceptance against the probe facts** (spec §1). Run `dispatch-table` for every results directory with its family, e.g. `$PY -m kernelscope.cli dispatch-table --results $R/uniform_s1_dense --family dense --cache-state cold --out $R/tables/uniform_s1_dense_cold.csv`. Required, and to be stated with numbers in the results note:
  - uniform S1 dense cold: heuristic median regret ≤ 5 %, max ≤ 30 % (probe: 1.0 %, 15.7 %);
  - uniform S1 dense warm: max regret ≥ 20 % (probe: 40.4 %);
  - ragged S1 dense cold, lens `32768x2+1024x30`: heuristic regret between 5× and 10× (probe and independent reproduction: 7.2–7.5×);
  - every `summaries.jsonl` has `iterations_dropped` ≤ 2 for ≥ 99 % of ok cells;
  - `check_ok` is true for every cell where the check ran.
  If a criterion fails, do not tune anything: record the observed numbers and stop for review.
- [ ] **Step 5: Results note** — write `docs/plan/2026-09-19-p0-campaign.md` with: date, GPU hygiene log, the `machines/rtx4090.json` headline numbers, one `regret_summary` block per table, the five acceptance checks with numbers, the paged-vs-dense comparison for the ragged worst case (a new fact: spec F14 was dense-only), error cells, and total GPU time.
- [ ] **Step 6: Docs** — README: add `bench`, `machine`, `dispatch-table`, `--cache-state` usage to the quick start and the layout section (keep existing content). STATUS: append `## 2026-09-19 — design-1-3: Phase 0 foundation` with a 10-line summary pointing to the results note.
- [ ] **Step 7: Commit** — `git commit -m "Run the Phase 0 measurement campaign on the RTX 4090" -- grids/dispatch_s1.yaml grids/dispatch_s2.yaml grids/ragged_s1.yaml grids/ragged_s2.yaml machines/rtx4090.json docs/plan/2026-09-19-p0-campaign.md README.md docs/STATUS.md`
