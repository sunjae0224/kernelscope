# Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Streamlit dashboard — the face of the graduation demo — with four views: **Diagnose** (why one kernel variant is slow on one workload), **What-if** (model predictions on a scaled machine, with the simulator and real SM-blocker points overlaid), **Kernel map** (best variant and heuristic regret across the (B, L) grid, measured vs model), and **Serving** (per-policy attention time, step time and TPOT from the dispatcher experiments, plus a live run of a small scenario).

**Architecture:** `kernelscope/dashboard/` holds pure functions that turn result files into tables and plotly figures (`style.py`, `data.py`, `figures.py`) and are unit-tested without a browser; `dashboard/app.py` is a thin Streamlit layer that wires widgets to those functions, smoke-tested with `streamlit.testing.v1.AppTest`. No new measurement code.

**Tech Stack:** streamlit 1.64, plotly 7.1, pandas (all already in the gradkernel env).

**Spec:** `docs/plan/2026-09-19-design-surrogate-dispatcher.md` §4. Inputs are only result files: `results/hw_4090/**` (bench rows), `results/sim_4090/**` (simulator rows, Codex), `results/hw_4090/blocked_s1` (V3), `results/serve_4090/**` (serving logs), `machines/rtx4090.json`, `models/rtx4090.json`.

## Global Constraints

- **Worktree:** `/home/skkai/AI_Accelerator/kernelscope-design`, branch `design-1-3`; never edit `/home/skkai/AI_Accelerator/kernelscope`. Result roots are read from `/home/skkai/AI_Accelerator/kernelscope/results` by default (`KERNELSCOPE_RESULTS` env var overrides).
- **Python:** `PY=/home/skkai/miniforge3/envs/gradkernel/bin/python`; whole suite `$PY -m pytest -q -p no:cacheprovider`. No installs.
- **Color (dataviz method; reference palette):** categorical slots in fixed order — light `#2a78d6 #eb6834 #1baf7a #eda100 #e87ba4 #008300 #4a3aa7 #e34948`, dark `#3987e5 #d95926 #199e70 #c98500 #d55181 #008300 #9085e9 #e66767`. Color follows the entity, never its rank: serving policies map to fixed slots (`heuristic` 1, `model` 2, `table` 3, `fa2` 4, all `fixed*` share slot 7 with a dash pattern per split count). Sequential magnitude uses the blue ramp `#cde2fb #b7d3f6 #9ec5f4 #86b6ef #6da7ec #5598e7 #3987e5 #2a78d6 #256abf #1c5cab #184f95 #104281 #0d366b`. No rainbow, no dual y-axis, no color-only identity (≥ 2 series always have a legend; ≤ 4 series also get direct labels). Text uses text colors, never series colors. Surfaces: light `#fcfcfb`, dark `#1a1a19`; text light `#0b0b0b` / `#52514e`, dark `#ffffff` / `#c3c2b7`.
- **Marks:** bars with gaps (`bargap` ≥ 0.15), lines 2px, markers ≥ 8px, recessive grid, hover tooltips on every mark with units.
- **Theme:** read `st.context.theme.type` (`"light"` / `"dark"`, default light) and pass it to every figure builder; figures set `paper_bgcolor` / `plot_bgcolor` to that mode's surface.
- **Commits:** `git add` new files, then `git commit -m "<msg>" -- <paths>`; end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

## Execution order

Sequential: Task 1 → 2 → 3 → 4 → 5.

---

### Task 1: Style and data loading

**Files:**
- Create: `kernelscope/dashboard/__init__.py`, `kernelscope/dashboard/style.py`, `kernelscope/dashboard/data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Produces:
  - `style.CATEGORICAL = {"light": [...8 hex...], "dark": [...]}`, `style.SEQUENTIAL` (13 hex), `style.SURFACE`, `style.TEXT = {"light": ("#0b0b0b", "#52514e"), "dark": ("#ffffff", "#c3c2b7")}`, `style.GRID`
  - `style.policy_color(policy: str, theme: str) -> str`; `style.policy_dash(policy: str) -> str` (`"solid"` except `fixed{N}` → one of `"dot", "dash", "dashdot", "longdash", "longdashdot"` by `log2(N)`)
  - `style.layout(fig, theme: str, title: str) -> fig`
  - `data.results_root() -> Path`; `data.find_dirs(kind) -> list[Path]` for `"hw"` (dirs under `hw_4090` holding `summaries.jsonl`), `"sim"` (under `sim_4090` holding `*.parquet`), `"serve"` (scenario dirs `serve_4090/<model>/<scenario>/` whose policy subdirs hold `steps.parquet`)
  - `data.load_rows(dirs) -> pd.DataFrame` (concatenated `ResultStore` rows; missing `cache_state` → `"warm"`)
  - `data.kernel_table(rows, cache_state) -> pd.DataFrame` indexed by (kernel, workload_key): `kernel_time_us, latency_us, launches`, launch-0 `grid_blocks, block_threads, regs, smem_bytes, blocks_per_sm_limit, sm_coverage, occupancy_device`, `achieved_gbps, dram_util`
  - `data.load_serving(scenario_dir) -> dict[str, dict[str, pd.DataFrame]]` (`policy -> {"steps", "tokens", "prefill"}`)

- [ ] **Step 1: Failing tests** — `tests/test_dashboard_data.py`:
```python
import pandas as pd

from kernelscope.dashboard import data, style
from kernelscope.results.store import ResultStore


def test_policy_colors_are_fixed_per_entity():
    assert style.policy_color("heuristic", "light") == "#2a78d6"
    assert style.policy_color("model", "dark") == "#d95926"
    assert style.policy_color("fixed8", "light") == style.policy_color("fixed32", "light")
    assert style.policy_dash("fixed8") != style.policy_dash("fixed32")
    assert style.policy_dash("model") == "solid"


def test_find_dirs_and_kernel_table(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    d = tmp_path / "hw_4090" / "u"
    key = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"
    rows = [{"workload_key": key, "kernel": "fa2", "backend": b, "metric": m, "unit": "", "value": v,
             "launch_idx": 0, "note": None}
            for b, m, v in [("profile", "kernel_time_us", 60.0), ("latency", "median_s", 70e-6),
                            ("profile", "grid_blocks", 8.0), ("profile", "blocks_per_sm_limit", 2.0),
                            ("analytic", "achieved_gbps", 70.0)]]
    ResultStore(d).write(rows, tag="x", extra={"cache_state": "cold"})
    (d / "summaries.jsonl").write_text("")
    assert data.find_dirs("hw") == [d]
    t = data.kernel_table(data.load_rows([d]), "cold")
    r = t.loc[("fa2", key)]
    assert r.kernel_time_us == 60.0 and r.latency_us == 70.0 and r.grid_blocks == 8 and r.achieved_gbps == 70.0


def test_load_serving_reads_each_policy_dir(tmp_path):
    for p in ("heuristic", "model"):
        (tmp_path / p).mkdir()
        pd.DataFrame({"step": [0], "policy": [p], "B": [2], "attn_us": [1.0], "step_us": [2.0]}).to_parquet(tmp_path / p / "steps.parquet")
        pd.DataFrame({"rid": [0], "step": [0], "token": [5], "t_us": [0.0]}).to_parquet(tmp_path / p / "tokens.parquet")
        pd.DataFrame({"rid": [0], "prompt_len": [8], "prefill_us": [3.0]}).to_parquet(tmp_path / p / "prefill.parquet")
    s = data.load_serving(tmp_path)
    assert set(s) == {"heuristic", "model"} and len(s["model"]["steps"]) == 1
```
- [ ] **Step 2: Implement.** `kernelscope/dashboard/__init__.py`: `"""Streamlit dashboard helpers: pure data and figure builders (spec §4)."""`. `style.py`:
```python
"""Chart colors and layout (reference data-viz palette; see the plan's Global Constraints)."""
import math

CATEGORICAL = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
SEQUENTIAL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}
TEXT = {"light": ("#0b0b0b", "#52514e"), "dark": ("#ffffff", "#c3c2b7")}
GRID = {"light": "#e8e7e3", "dark": "#2e2e2b"}
_SLOT = {"heuristic": 0, "model": 1, "table": 2, "fa2": 3}
_DASH = ["dot", "dash", "dashdot", "longdash", "longdashdot"]


def policy_color(policy: str, theme: str) -> str:
    return CATEGORICAL[theme][_SLOT.get(policy, 6)]


def policy_dash(policy: str) -> str:
    if policy.startswith("fixed"):
        return _DASH[int(math.log2(int(policy[5:]))) % len(_DASH)]
    return "solid"


def layout(fig, theme: str, title: str):
    ink, muted = TEXT[theme]
    fig.update_layout(title={"text": title, "font": {"color": ink, "size": 16}},
                      paper_bgcolor=SURFACE[theme], plot_bgcolor=SURFACE[theme],
                      font={"color": muted}, legend={"orientation": "h", "y": 1.08, "font": {"color": ink}},
                      margin={"l": 60, "r": 20, "t": 70, "b": 50}, bargap=0.2,
                      hoverlabel={"bgcolor": SURFACE[theme], "font": {"color": ink}})
    fig.update_xaxes(gridcolor=GRID[theme], zeroline=False, linecolor=GRID[theme])
    fig.update_yaxes(gridcolor=GRID[theme], zeroline=False, linecolor=GRID[theme])
    return fig
```
`data.py`:
```python
"""Loading result files for the dashboard. Everything is read-only."""
import os
from pathlib import Path

import pandas as pd

from kernelscope.results.store import ResultStore

DEFAULT_ROOT = "/home/skkai/AI_Accelerator/kernelscope/results"
_LAUNCH0 = ["grid_blocks", "block_threads", "regs", "smem_bytes", "blocks_per_sm_limit", "sm_coverage", "occupancy_device"]


def results_root() -> Path:
    return Path(os.environ.get("KERNELSCOPE_RESULTS", DEFAULT_ROOT))


def find_dirs(kind: str) -> list:
    root = results_root()
    if kind == "hw":
        return sorted(p.parent for p in (root / "hw_4090").glob("*/summaries.jsonl"))
    if kind == "sim":
        return sorted({p.parent for p in (root / "sim_4090").glob("*/*.parquet")})
    if kind == "serve":
        return sorted({p.parent.parent for p in (root / "serve_4090").glob("*/*/*/steps.parquet")})
    raise ValueError(kind)


def load_rows(dirs) -> pd.DataFrame:
    df = pd.concat([ResultStore(d).load() for d in dirs], ignore_index=True)
    if "cache_state" not in df.columns:
        df["cache_state"] = None
    df["cache_state"] = df["cache_state"].astype(object).fillna("warm")
    return df


def kernel_table(rows: pd.DataFrame, cache_state: str) -> pd.DataFrame:
    r = rows[rows.cache_state == cache_state]

    def pick(backend, metric, launch0=False):
        s = r[(r.backend == backend) & (r.metric == metric)]
        if launch0:
            s = s[s.launch_idx == 0]
        return s.groupby(["kernel", "workload_key"])["value"].median()

    cols = {"kernel_time_us": pick("profile", "kernel_time_us"), "latency_us": pick("latency", "median_s") * 1e6,
            "launches": pick("profile", "launches_per_iter"),
            **{m: pick("profile", m, True) for m in _LAUNCH0},
            "achieved_gbps": pick("analytic", "achieved_gbps"), "dram_util": pick("analytic", "dram_util")}
    return pd.DataFrame(cols)


def load_serving(scenario_dir) -> dict:
    out = {}
    for p in sorted(Path(scenario_dir).iterdir()):
        if (p / "steps.parquet").exists():
            out[p.name] = {k: pd.read_parquet(p / f"{k}.parquet") for k in ("steps", "tokens", "prefill")
                           if (p / f"{k}.parquet").exists()}
    return out
```
- [ ] **Step 3: Run → PASS; whole suite. Step 4: Commit** — `git add kernelscope/dashboard tests/test_dashboard_data.py && git commit -m "Add dashboard style and data loading" -- kernelscope/dashboard tests/test_dashboard_data.py`

---

### Task 2: Figure builders

**Files:**
- Create: `kernelscope/dashboard/figures.py`
- Test: `tests/test_dashboard_figures.py`

**Interfaces:**
- Consumes: `style` (Task 1); `whatif.table` (model plan); `dispatch_table` (Phase 0); `report.tpot_us` (dispatcher plan).
- Produces (every builder takes `theme` and returns a `plotly.graph_objects.Figure`, and calls `style.layout(fig, theme, title)` last):
  - `variants_bar(predicted: pd.DataFrame, measured: pd.Series | None, heuristic: str, theme, title)` — horizontal bars, one row per variant in `predicted` order (`predicted` has the columns of a `whatif.table`); series "model" (slot 1) and, when given, "measured" (slot 2, indexed by plugin); the heuristic's row label gets the suffix " ← library heuristic"; x axis "kernel time (µs)"; hover shows µs with one decimal plus `splits`, `slots_per_sm`, `limiter`
  - `whatif_compare(base: pd.DataFrame, scaled: pd.DataFrame, theme, title, overlay: pd.DataFrame | None = None)` — grouped horizontal bars "measured machine" (slot 1) vs "what-if machine" (slot 2) per variant; optional overlay markers (slot 3, size ≥ 8, symbol `diamond`) from a frame with columns `plugin, time_us, source`, source shown in the hover
  - `regret_heatmap(table: pd.DataFrame, theme, title)` — uniform rows only (`ragged == False`): x = L_kv as category labels in ascending order, y = B ascending, z = heuristic regret, colorscale = `style.SEQUENTIAL` spread over [0, 1], `zmin=0`, `zmax=max(0.5, max regret)`, cell text = best variant's split label (`"fa2"`, `"heur"` for flashdecoding*, `"s8"` for fd_s8 / fd_s8_paged), hover with best variant, best µs, heuristic µs and regret as a percentage, colorbar title "heuristic regret"
  - `ragged_bars(table: pd.DataFrame, theme, title, top=12)` — the `top` ragged rows by regret: horizontal bars of slowdown `heuristic_us / best_us` (slot 1) labelled `"B{B} {lens}"`, x axis "heuristic slowdown (×)", a vertical line at 1
  - `serving_steps(serving: dict, theme, title, metric="attn_us")` — one line per policy (policy color and dash), x = step, y = metric / 1000 labelled in ms, legend always; when ≤ 4 policies, an annotation with the policy name at each line's last point
  - `serving_batch(steps: pd.DataFrame, theme, title)` — stacked areas `n_long` (slot 1, "long ≥ 4096") and `B - n_long` (slot 2, "short"), x = step, y axis "sequences"
  - `tpot_box(serving: dict, theme, title)` — one box per policy (policy colors), y = TPOT (ms) from `report.tpot_us(tokens) / 1000`, `boxpoints=False`
- [ ] **Step 1: Failing tests** — `tests/test_dashboard_figures.py`: for each builder construct a small frame and assert concretely: trace count and names; `fig.data[0].marker.color == style.CATEGORICAL["light"][0]` for the first series; axis titles; the heuristic row label ends with "← library heuristic"; `regret_heatmap` puts "s8" in the text of the (B, L) cell whose best kernel is `fd_s8`, and its `zmin == 0`; `serving_steps` gives `heuristic` the slot-1 color and `model` the slot-2 color even when the dict lists `model` first; `fig.layout.paper_bgcolor == style.SURFACE[t]` for `t` in light and dark.
- [ ] **Step 2: Implement `figures.py`** with `plotly.graph_objects` (`go.Bar(orientation="h")`, `go.Heatmap`, `go.Scatter(mode="lines")`, `go.Scatter(stackgroup="one")`, `go.Box`), each builder under ~40 lines, units in every `hovertemplate` (for example `"%{x:.1f} µs<extra>%{fullData.name}</extra>"`).
- [ ] **Step 3: Run → PASS; whole suite. Step 4: Commit** — `git add kernelscope/dashboard/figures.py tests/test_dashboard_figures.py && git commit -m "Add dashboard figure builders" -- kernelscope/dashboard/figures.py tests/test_dashboard_figures.py`

---

### Task 3: The Streamlit app (four views)

**Files:**
- Create: `dashboard/app.py`
- Test: `tests/test_dashboard_app.py` (AppTest)

**Layout:**
- `st.set_page_config(page_title="kernelscope", layout="wide")`; `theme = (st.context.theme.type or "light")`.
- Sidebar: results root (text input, default `data.results_root()`, sets `KERNELSCOPE_RESULTS` for the session), cache state (radio cold/warm, default cold), family (radio dense/paged).
- `st.tabs(["Diagnose", "What-if", "Kernel map", "Serving"])`.
- **Diagnose:** selectors for results dir (`find_dirs("hw")`), workload key (keys present in that dir), variant; a metrics row with `st.metric` for kernel time, latency, CTAs (`grid_blocks`), CTAs/SM (`blocks_per_sm_limit`), achieved GB/s, and for cold rows DRAM utilisation; a launch table (all launches of the variant, from the profile rows); the model prediction for the same cell (`predict` with `machines/rtx4090.json`, `models/rtx4090.json`) with kind, splits, slots/SM, limiter, bandwidth-bound fraction and the measured / predicted ratio; `variants_bar` with the family's model table and the measured times of that workload.
- **What-if:** workload builder (radio uniform / ragged; uniform: B, L; ragged: B, n_long, L_long, L_short); sliders SM × (0.25–2.0, step 0.25), DRAM × (0.5–2.0, step 0.25), L2 × (0.5–2.0, step 0.25), shared memory per SM (select 100 KiB / 164 KiB → scale 1.0 / 1.64); `whatif_compare(table(w, family, m, p, state), table(w, family, m, p, state, scales))`. Overlays: when only the SM slider moved and `results/hw_4090/blocked_s1` has the workload, add the V3 points (the measured blocked time scaled by the model's unblocked time over the measured unblocked time, source "SM blocker (measured)"); when a simulator variant matching a single-resource slider exists (bw_x2 ↔ DRAM 2.0, bw_half ↔ 0.5, l2_x2 / l2_half, sm_x2 / sm_half) for the workload and plugin, add it with source "Accel-Sim (uncalibrated)". Caption: the heuristic's split count and the model's best variant on each machine.
- **Kernel map:** `dispatch_table(rows, family, state)` → `regret_heatmap` (measured) and, in a second column, the same heatmap from model predictions for the same workloads (best = argmin of predicted times among the measured plugins; this map shows what the model believes); `ragged_bars` below; `st.dataframe` of the ragged rows (the table view).
- **Serving:** scenario selector (`find_dirs("serve")`); `report.summarize(dir)` table; `serving_steps` for `attn_us`, then a separate `serving_steps` for `step_us` (never a dual axis); `serving_batch` of the heuristic run; `tpot_box`; `equivalence.csv` table if present in the scenario dir; a "Live run" expander with a button that runs `kernelscope serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/tiny.yaml --policy heuristic --policy model --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 2 --out <tmp dir>` via `subprocess.run` inside `st.spinner`, then renders `serving_steps` and the summary from the new directory. Disable the button (with a caption) when `nvidia-smi --query-compute-apps` lists a process other than `rerun`.
- Every chart: `st.plotly_chart(fig, use_container_width=True, theme=None)`.

- [ ] **Step 1: AppTest smoke tests** — `tests/test_dashboard_app.py`: build a temporary results root with `hw_4090/u` (bench rows for `fa2` and `flashdecoding` on one uniform workload, cold, plus a `summaries.jsonl`) and `serve_4090/qwen3_4b/tiny/{heuristic,model}` (synthetic steps/tokens/prefill parquet); set `KERNELSCOPE_RESULTS`; `at = AppTest.from_file("dashboard/app.py", default_timeout=60).run()`; assert `not at.exception`, `len(at.tabs) == 4`, the Diagnose metrics include the kernel time value, and setting the cache-state radio to "warm" then `.run()` raises no exception. Skip the module when `models/rtx4090.json` is absent.
- [ ] **Step 2: Implement `dashboard/app.py`** as above; under ~250 lines, delegating to `figures` / `data`; cache loaders with `@st.cache_data`.
- [ ] **Step 3: Run → PASS; whole suite. Step 4: Commit** — `git add dashboard/app.py tests/test_dashboard_app.py && git commit -m "Add the Streamlit dashboard" -- dashboard/app.py tests/test_dashboard_app.py`

---

### Task 4: Palette validation and visual check

**Files:**
- Create: `docs/plan/2026-09-19-p3-dashboard-check.md`, `docs/img/dashboard_*.png`

- [ ] **Step 1:** Validate the categorical slots in use (1, 2, 3, 4, 7) for both modes with the dataviz validator `node /tmp/claude-1000/bundled-skills/2.1.276/5aaf2d7b6c256472791bc7fd07d4da3d/dataviz/scripts/validate_palette.js "<hex,...>" --mode light` and `--mode dark` (check the script's usage line for the surface option). Record the output. If `node` is missing, record that and continue.
- [ ] **Step 2:** Start `$PY -m streamlit run dashboard/app.py --server.headless true --server.port 8599` in the background against the real results root; capture each tab in light and dark if a headless browser is already available (check `which chromium chromium-browser google-chrome` and `$PY -c "import playwright"`); otherwise state that screenshots were not possible. Look for label collisions, overflow and unreadable text; fix layout issues in `figures.py` / `app.py`, re-run the tests, re-capture. Stop the server afterwards.
- [ ] **Step 3:** Write the check note (validator output, screenshots or their absence, issues found and fixed) and commit: `git add docs/plan/2026-09-19-p3-dashboard-check.md docs/img && git commit -m "Validate the dashboard palette and layout" -- docs/plan/2026-09-19-p3-dashboard-check.md docs/img kernelscope/dashboard dashboard`

---

### Task 5: README and demo script

**Files:**
- Modify: `README.md`, `docs/STATUS.md`
- Create: `docs/demo.md`

- [ ] **Step 1:** README: how to start the dashboard, what each tab shows, where its data comes from.
- [ ] **Step 2:** `docs/demo.md`, a five-minute demo script: (1) Diagnose fa2 on B=1, L=32K — 8 CTAs on 128 SMs, two CTAs per SM by registers; (2) Kernel map — the heuristic is fine on uniform batches (cold) and fails on ragged ones; (3) What-if — halve the SMs and watch the heuristic's split count change; raise shared memory to 164 KiB and watch the split kernel fit twice per SM; (4) Serving — the ragged scenario's TPOT under heuristic vs model, and the equivalence table. Quote measured numbers from `docs/plan/2026-09-19-p0-campaign.md`, `docs/plan/2026-09-19-p1-model-validation.md` and `docs/plan/2026-09-19-p2-serving-results.md` (only numbers that exist there).
- [ ] **Step 3:** STATUS entry (≤ 10 lines). Commit: `git add docs/demo.md && git commit -m "Document the dashboard and the demo script" -- README.md docs/STATUS.md docs/demo.md`
