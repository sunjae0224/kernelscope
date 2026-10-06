# Demo Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fixed-order presentation page (Demo) to the existing Streamlit app, with an A/B token-race replay of recorded serving runs, while keeping the five exploration tabs as a second page (Lab).

**Architecture:** `dashboard/app.py` becomes a two-page `st.navigation` entrypoint; the old body moves unchanged to `dashboard/lab_page.py`; `dashboard/demo_page.py` renders five scenes. All data shaping lives in `kernelscope/dashboard/demo.py` (pure functions over recorded parquet/csv/json, no Streamlit, no torch) so it is unit-tested without a browser. The one animated element, the token race, is a self-contained HTML+JS template (`kernelscope/dashboard/race.html`) filled with a JSON payload and embedded via `st.components.v1.html`; its play/pause/speed controls live in JS so Streamlit reruns never interrupt playback.

**Tech Stack:** streamlit 1.64 (`st.navigation`, `st.Page`, `streamlit.testing.v1.AppTest.switch_page`), plotly 7.1, pandas, pyyaml. No new dependencies.

**Spec:** `docs/plan/2026-10-06-design-demo-page.md` (read it first; §4 defines the race payload, §6 the module interfaces).

## Global Constraints

- Repo: worktree `/home/skkai/AI_Accelerator/kernelscope-design`, branch `design-1-3`. Python is always `.venv/bin/python` (= `/home/skkai/miniforge3/envs/gradkernel/bin/python`). No installs.
- **Do not `git commit`, `git add` or `git stash`.** Leave all changes in the working tree (project rule: commits only on the user's explicit request).
- Run tests with the GPU hidden so a measurement running on the shared 4090 is not disturbed: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider <files>`. The full suite is `CUDA_VISIBLE_DEVICES= make test`.
- `kernelscope/dashboard/demo.py` must not import `streamlit` or `torch` at module level (the verify/laptop path). Heavy imports (`kernelscope.serve.dispatch`, `kernelscope.model.*`, `kernelscope.serve.hf`) go inside functions.
- Colors follow the dataviz method already encoded in `kernelscope/dashboard/style.py`: policy colors come only from `style.policy_color` (heuristic blue, model orange, table green, hybrid slot 6 purple, fixed* slot 6 with dash patterns); surfaces/text from `style.SURFACE`, `style.TEXT`, `style.GRID`. Never color text with a series color; every multi-series chart has a legend; no dual y-axes; `bargap >= 0.15`.
- UI copy is Korean. Every number shown on the Demo page is computed from recorded files at render time; **no result numbers are hard-coded in page copy** (so `kernelscope verify` needs no new entries). Evidence labels: 실측 (measured), 재생 (replay), 계산 (derived from lengths/geometry), 예측 (model).
- Existing `lab_page.py` content (the former `app.py` body) is moved, not edited, except for the three substitutions in Task 1.
- Do not run GPU commands. The recording `../kernelscope/results/serve_4090/demo_text_20261006/ragged` already exists (heuristic / table / hybrid, 3 repeats, natural text).

## Review Focus

Inputs the spec implies but no happy-path test covers; each has a pinned test in the owning task:

1. A serving directory with only `heuristic/` (no comparison policy) or without `heuristic/` must give the page an info message, never an exception (Task 2 `test_race_payload_requires_pair`, Task 4 app test with a single policy).
2. Arrival-style runs: requests admitted mid-run give token timestamps earlier than a naive "all admitted" clock zero; the first-wave rule must keep every `t_ms >= 0` (Task 2 `test_race_payload_arrivals_clock`).
3. Model/hybrid policy construction can fail (missing params file, simulator backend); the decision card must return a note row for that policy and valid rows for the others (Task 3 `test_decision_card_without_params`).
4. `summary.csv` may lack the requested policy or the `tokens_equivalent` column; `headline` returns `None` fields, and the hero tiles print "—" (Task 2 `test_headline_and_missing_policy`).
5. Recordings whose policy folder holds `steps.parquet` directly (no `repeat_*`), as `data.load_serving` already accepts (Task 2 `test_race_payload_flat_policy_dir`).

---

### Task 1: Two-page navigation (Demo default, Lab = former app)

**Files:**
- Modify: `kernelscope/dashboard/style.py` (add `current_theme`, `PAGE_CSS`)
- Create: `dashboard/lab_page.py` (copy of current `dashboard/app.py` with three substitutions)
- Create: `dashboard/demo_page.py` (placeholder; Task 4 replaces it)
- Rewrite: `dashboard/app.py` (entrypoint only)
- Modify: `tests/test_dashboard_app.py`

**Interfaces:**
- Produces: `style.current_theme(default="light") -> str`; `style.PAGE_CSS: str` (the `<style>…</style>` block the pages share); entrypoint `dashboard/app.py` registering pages `demo_page.py` (default) and `lab_page.py`.

- [ ] **Step 1: Read the AppTest page-switch contract**

Run:
```bash
.venv/bin/python -c "from streamlit.testing.v1 import AppTest; print(AppTest.switch_page.__doc__)"
```
Expected: the docstring says `switch_page("pages/my_page.py")` is relative to the main script and that `run()` must be called once before switching. Our pages sit next to the entrypoint, so the path is `"lab_page.py"`.

- [ ] **Step 2: Write the failing tests**

Replace the whole of `tests/test_dashboard_app.py` with:

```python
import json
from pathlib import Path

import pytest
import pandas as pd

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
LAB = "lab_page.py"
WORKLOAD = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"


def _hw_rows(tmp_path):
    directory = tmp_path / "hw_4090" / "tiny"
    directory.mkdir(parents=True)
    rows = [dict(status="ok", plugin=p, workload_key=WORKLOAD, cache_state="cold", kernel_time_us=t)
            for p, t in [("fa2", 20.), ("flashdecoding", 30.)]]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def _lab(app):
    app.switch_page(LAB)
    return app.run()


def test_entrypoint_opens_demo_then_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    _hw_rows(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "같은 답, 더 빠른 토큰."
    assert not app.tabs                       # the Demo page is one flow, no tabs
    app = _lab(app)
    assert not app.exception
    assert len(app.tabs) == 5
    assert any(metric.value == "30.0 µs" for metric in app.metric)
    assert any("아직 없습니다" in item.value for item in app.info)
    cache = next(radio for radio in app.radio if "Cache" in radio.label)
    cache.set_value("warm")
    app.run()
    assert not app.exception
    assert any("GPU 측정 기록이 없습니다" in item.value for item in app.info)


def test_lab_empty_result_root_is_browsable(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    app = _lab(AppTest.from_file(str(APP), default_timeout=60).run())
    assert not app.exception
    assert len(app.tabs) == 5
    assert app.metric[0].value == "0"
```

Then keep every other existing test function from the old file (those after `test_app_empty_result_root_is_browsable`) **unchanged except** wrapping their first `AppTest.from_file(...).run()` with `_lab(...)`, e.g. `app = _lab(AppTest.from_file(str(APP), default_timeout=60).run())`. Do not change their assertions.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_app.py -x`
Expected: FAIL (`switch_page` raises ValueError because `lab_page.py` does not exist / `st.navigation` is not active).

- [ ] **Step 4: Add the shared theme helper and CSS to style.py**

Append to `kernelscope/dashboard/style.py`:

```python
PAGE_CSS = """<style>
  .block-container { max-width: 1440px; padding-top: 2.4rem; padding-bottom: 3rem; }
  h1 { letter-spacing: -0.065em; font-weight: 750 !important; }
  h2,h3 { letter-spacing: -0.025em; }
  [data-testid="stMetric"] { border: 1px solid #b8bfcc55; border-radius: 14px; padding: 17px 20px; }
  [data-testid="stMetricLabel"] { font-size: .84rem; }
  [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
  [data-testid="stTabs"] button { font-weight: 650; padding-left: 18px; padding-right: 18px; }
  .eyebrow { font-size: .73rem; letter-spacing: .18em; font-weight: 750; color: #56729b; margin-bottom: 10px; }
  .hero-note { max-width: 850px; font-size: 1.07rem; line-height: 1.7; opacity: .78; margin-bottom: 22px; }
  .evidence { display: inline-block; border: 1px solid #8b97aa55; border-radius: 999px;
     padding: 5px 11px; font-size: .75rem; margin: 0 6px 8px 0; }
  .footnote { font-size: .78rem; opacity: .66; line-height: 1.6; }
</style>"""


def current_theme(default: str = "light") -> str:
    """The theme the browser reports, else the configured base; ``default`` outside Streamlit.

    Streamlit may report the browser's default light scheme on the first render, before it
    receives the configured app colors, so the first call in a session returns the configured base."""
    try:
        import streamlit as st
    except ImportError:
        return default
    configured = st.get_option("theme.base") or default
    try:
        theme = st.context.theme.type or configured
    except (AttributeError, KeyError):
        theme = configured
    theme = theme if theme in ("light", "dark") else configured
    if not st.session_state.get("_dashboard_theme_initialized"):
        theme = configured
        st.session_state["_dashboard_theme_initialized"] = True
    return theme
```

- [ ] **Step 5: Create lab_page.py from the current app.py**

```bash
cp dashboard/app.py dashboard/lab_page.py
```
Then in `dashboard/lab_page.py` make exactly these edits:
1. Change the module docstring's first line to `"""Lab page: the five exploration tabs (Diagnose / Kernel map / What-if / Serving / Policy lab)."""`.
2. Delete the line `st.set_page_config(page_title="KernelScope · GPU Attention Observatory", page_icon="◈", layout="wide")`.
3. Replace the theme block (from `configured_theme = st.get_option("theme.base") or "light"` through `st.session_state["_dashboard_theme_initialized"] = True`, 11 lines) with the single line `theme = style.current_theme()`.
4. Replace the inline `st.markdown("""<style> … </style>""", unsafe_allow_html=True)` block (13 lines) with `st.markdown(style.PAGE_CSS, unsafe_allow_html=True)`.
5. Change `from kernelscope.dashboard import data, figures, research` to `from kernelscope.dashboard import data, figures, research, style`.
Nothing else changes.

- [ ] **Step 6: Write the placeholder Demo page and the new entrypoint**

`dashboard/demo_page.py` (placeholder, Task 4 replaces it):
```python
"""Demo page: fixed-order scenes for the presentation (placeholder until Task 4)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from kernelscope.dashboard import style

st.markdown(style.PAGE_CSS, unsafe_allow_html=True)
st.title("같은 답, 더 빠른 토큰.")
st.caption("Demo 페이지 준비 중 · Lab 페이지에서 다섯 탭을 볼 수 있습니다.")
```

`dashboard/app.py` (whole file):
```python
"""Run: python -m streamlit run dashboard/app.py --server.address 127.0.0.1

Two pages: Demo (fixed-order scenes for the presentation, the default) and Lab (the five
exploration tabs for Q&A drill-down). Streamlit executes this file by path; a source checkout
need not be installed."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="KernelScope · GPU Attention Observatory", page_icon="◈", layout="wide")
pages = [st.Page("demo_page.py", title="Demo", default=True), st.Page("lab_page.py", title="Lab")]
st.navigation(pages).run()
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_app.py tests/test_dashboard_figures.py tests/test_dashboard_data.py`
Expected: all PASS. If `switch_page` complains the page is not registered, confirm the entrypoint uses `st.Page("lab_page.py")` with that exact relative path.

- [ ] **Step 8: Smoke-run the app headless for five seconds**

Run:
```bash
KERNELSCOPE_RESULTS=demo_data timeout 8 .venv/bin/python -m streamlit run dashboard/app.py --server.headless true --server.port 8599 --browser.gatherUsageStats false; echo "exit $?"
```
Expected: the server banner prints and exit code is 124 (killed by timeout), no traceback.

---

### Task 2: demo.py — featured runs, headline, campaign table, race payload

**Files:**
- Create: `kernelscope/dashboard/demo.py`
- Create: `tests/demo_fixtures.py`
- Create: `tests/test_dashboard_demo.py`

**Interfaces:**
- Produces (all in `kernelscope.dashboard.demo`):
  - `WORST_KEY: str`, `FAMILY_LABELS: dict[str, str]`, `POLICY_LABELS: dict[str, str]`, `SYNTHETIC: str`, `DEFAULT_QUESTION: str`
  - `featured_runs(root) -> dict[str, Path]` (keys ⊆ {"ragged", "arrivals", "uniform"})
  - `headline(run_dir, policy: str) -> dict` with keys `heuristic_tpot_ms, policy_tpot_ms, speedup, tokens_equivalent, repeats, model, scenario_name, created_at, policy`
  - `campaign_table(campaign_dir) -> pd.DataFrame[scenario, family, policy, tpot_ms, speedup, repeats, tokens_equivalent]`
  - `policy_dirs(run_dir) -> list[str]` (policy folder names that hold recorded steps)
  - `race_payload(run_dir, policies, reference="heuristic", decode=None) -> dict` (spec §4 schema)
  - `save_race(run_dir, payload) -> Path`, `load_race(run_dir) -> dict | None`
- Consumes: `kernelscope.dashboard.data.find_dirs("serve", root)`, `data.load_manifest`, `style.policy_color`.

- [ ] **Step 1: Write the shared fixture builder**

`tests/demo_fixtures.py`:
```python
"""Synthetic serving recordings for dashboard tests (same files `serve run` writes, tiny numbers)."""
import json
from pathlib import Path

import pandas as pd


def make_run(root, campaign="camp", scenario="ragged", policies=("heuristic", "table"), repeats=2, B=3,
             new_tokens=4, diverge=None, natural=False, created="2026-10-01T00:00:00+00:00",
             scenario_family=None, flat=False, arrivals=False):
    """Write <root>/serve_4090/<campaign>/<scenario>/ with per-policy repeat folders and a summary.

    ``diverge=(policy, rid, position)`` makes that policy emit different tokens from ``position`` on.
    ``flat`` puts steps/tokens/prefill directly under the policy folder (no repeat_* level).
    ``arrivals`` admits requests 1.. one decode step apart instead of all at step 0."""
    run = Path(root) / "serve_4090" / campaign / scenario
    lens = [4096] + [128] * (B - 1)
    summary = []
    for policy in policies:
        speed = 1.0 if policy == "heuristic" else 0.5
        for r in range(repeats if not flat else 1):
            d = run / policy if flat else run / policy / f"repeat_{r:03d}"
            d.mkdir(parents=True)
            first = [500.0 + 50.0 * i for i in range(B)]
            arrival = [0.0] * B
            if arrivals:  # request i arrives i steps late and is admitted right away
                arrival = [0.0] + [first[-1] + i * 1e6 * speed for i in range(1, B)]
                first = [first[0]] + [a + 50.0 for a in arrival[1:]]
            prefill = pd.DataFrame({"rid": range(B), "prompt_len": lens, "prefill_us": [500.0] + [50.0] * (B - 1),
                                    "arrival_us": arrival, "admitted_us": arrival, "first_token_us": first,
                                    "prompt_sha256": "x"})
            tokens = [dict(rid=i, step=0 if not arrivals else i, position=0, token=100 + i, t_us=first[i], phase="prefill")
                      for i in range(B)]
            steps = []
            t0 = first[0] if arrivals else first[-1]
            for s in range(new_tokens - 1):
                t = t0 + (s + 1) * 1e6 * speed * (1 + 0.1 * r)
                for i in range(B):
                    if arrivals and s < i:
                        continue
                    token = 200 + 10 * s + i
                    if diverge and policy == diverge[0] and i == diverge[1] and s + 1 >= diverge[2]:
                        token += 1000
                    tokens.append(dict(rid=i, step=s, position=s + 1 - (i if arrivals else 0), token=token, t_us=t, phase="decode"))
                steps.append(dict(step=s, policy=policy, B=B, len_max=lens[0] + s, len_sum=sum(lens) + B * s, n_long=1,
                                  num_splits=0 if policy == "heuristic" else 16, attn_us=800.0 * speed,
                                  step_us=950.0 * speed, policy_us=3.0, policy_cache_hit=True, decode_wall_us=960.0 * speed,
                                  seq_ids=json.dumps(list(range(B))), lens=json.dumps([n + s + 1 for n in lens])))
            prefill.to_parquet(d / "prefill.parquet", index=False)
            pd.DataFrame(tokens).to_parquet(d / "tokens.parquet", index=False)
            pd.DataFrame(steps).to_parquet(d / "steps.parquet", index=False)
        summary.append(dict(policy=policy, repeats=repeats, tpot_ms_mean=1.0 * speed, speedup_vs_heuristic=1.0 / speed,
                            tokens_equivalent=not (diverge and diverge[0] == policy)))
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    manifest = {"model": "test/model", "scenario": f"/x/scenarios/{scenario}.yaml", "created_at": created, "repeats": repeats,
                "prompt_kind": "local_text" if natural else "seeded_synthetic_token_ids",
                "requests": [{"prompt_len": n, "max_new_tokens": new_tokens} for n in lens]}
    if scenario_family:
        manifest["scenario_family"] = scenario_family
    (run / "manifest.json").write_text(json.dumps(manifest))
    return run
```

- [ ] **Step 2: Write the failing tests**

`tests/test_dashboard_demo.py`:
```python
import json
from pathlib import Path

import pandas as pd
import pytest

from demo_fixtures import make_run
from kernelscope.dashboard import demo


def test_race_payload_structure_and_clock(tmp_path):
    run = make_run(tmp_path)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert [p["name"] for p in payload["policies"]] == ["heuristic", "table"]
    heuristic, table = payload["policies"]
    assert payload["scenario"]["B"] == 3 and payload["scenario"]["lens"] == [4096, 128, 128]
    assert payload["scenario"]["max_new_tokens"] == 4 and payload["scenario"]["natural_text"] is False
    # clock zero = the first wave's last first-token time: prefill tokens sit at 0, the first decode step at +1000 ms
    assert heuristic["requests"][0]["tokens"][0]["t_ms"] == 0.0
    assert heuristic["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(1000.0)
    assert table["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(500.0)
    assert heuristic["end_ms"] == pytest.approx(3000.0) and table["end_ms"] == pytest.approx(1500.0)
    assert [r["featured"] for r in heuristic["requests"]] == [True, True, False]
    assert heuristic["steps"][0] == {"step": 0, "t_ms": pytest.approx(1000.0), "num_splits": 0, "attn_ms": pytest.approx(0.8)}
    assert table["steps"][0]["num_splits"] == 16
    assert heuristic["tpot_ms"] == pytest.approx(1.0) and heuristic["color"].startswith("#")
    assert payload["reference"] == "heuristic"
    assert payload["agreement"]["table"] == {"identical": True, "requests_equal": 3, "requests_total": 3,
                                             "first_divergence": [], "summary_tokens_equivalent": True}
    json.dumps(payload)  # serializable


def test_race_payload_picks_median_repeat(tmp_path):
    run = make_run(tmp_path, repeats=3)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert [p["repeat_index"] for p in payload["policies"]] == [1, 1]   # end times grow with the repeat; the middle one


def test_race_payload_even_repeats_take_the_smaller_median(tmp_path):
    run = make_run(tmp_path, repeats=2)
    assert demo.race_payload(run, ["heuristic", "table"])["policies"][0]["repeat_index"] == 0


def test_race_payload_detects_divergence(tmp_path):
    run = make_run(tmp_path, diverge=("table", 1, 2))
    agreement = demo.race_payload(run, ["heuristic", "table"])["agreement"]["table"]
    assert agreement["identical"] is False and agreement["first_divergence"] == [{"rid": 1, "position": 2}]
    assert agreement["requests_equal"] == 2 and agreement["summary_tokens_equivalent"] is False


def test_race_payload_decodes_incrementally(tmp_path):
    run = make_run(tmp_path, natural=True)
    payload = demo.race_payload(run, ["heuristic", "table"], decode=lambda ids: " ".join(f"w{t}" for t in ids))
    assert [t["text"] for t in payload["policies"][0]["requests"][0]["tokens"]] == ["w100", " w200", " w210", " w220"]
    assert payload["scenario"]["natural_text"] is True


def test_race_payload_without_decode_has_no_text(tmp_path):
    run = make_run(tmp_path, natural=True)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert payload["scenario"]["natural_text"] is False
    assert all(t["text"] == "" for t in payload["policies"][1]["requests"][2]["tokens"])


def test_race_payload_requires_pair(tmp_path):
    run = make_run(tmp_path, policies=("heuristic",))
    with pytest.raises(ValueError):
        demo.race_payload(run, ["heuristic"])
    with pytest.raises(FileNotFoundError):
        demo.race_payload(run, ["heuristic", "table"])
    with pytest.raises(FileNotFoundError):
        demo.race_payload(make_run(tmp_path, campaign="noref", policies=("table",)), ["table"])


def test_race_payload_arrivals_clock_is_nonnegative(tmp_path):
    run = make_run(tmp_path, arrivals=True, B=3)
    payload = demo.race_payload(run, ["heuristic", "table"])
    times = [t["t_ms"] for p in payload["policies"] for r in p["requests"] for t in r["tokens"]]
    assert min(times) == 0.0 and payload["t0_rule"] == "first_wave_admitted"
    assert payload["policies"][0]["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(1000.0)


def test_race_payload_flat_policy_dir(tmp_path):
    run = make_run(tmp_path, flat=True)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert payload["policies"][0]["repeat_index"] == 0 and payload["policies"][0]["end_ms"] == pytest.approx(3000.0)
    assert demo.policy_dirs(run) == ["heuristic", "table"]


def test_save_and_load_race(tmp_path):
    run = make_run(tmp_path)
    payload = demo.race_payload(run, ["heuristic", "table"])
    path = demo.save_race(run, payload)
    assert path == run / "race.json" and demo.load_race(run) == json.loads(json.dumps(payload))
    assert demo.load_race(tmp_path) is None
    path.write_text("{not json")
    assert demo.load_race(run) is None


def test_featured_runs_prefers_natural_then_newest(tmp_path):
    make_run(tmp_path, campaign="old", scenario="ragged", created="2026-09-01T00:00:00+00:00")
    make_run(tmp_path, campaign="new", scenario="ragged", created="2026-10-01T00:00:00+00:00")
    make_run(tmp_path, campaign="text", scenario="heldout_ragged", natural=True, created="2026-09-15T00:00:00+00:00",
             scenario_family="ragged")
    make_run(tmp_path, campaign="new", scenario="control_uniform_fixed8", policies=("heuristic", "fixed8"),
             created="2026-10-01T00:00:00+00:00")
    runs = demo.featured_runs(tmp_path)
    assert runs["ragged"].parent.name == "text" and runs["uniform"].name == "control_uniform_fixed8"
    assert "arrivals" not in runs and list(runs) == ["ragged", "uniform"]


def test_featured_runs_empty_root(tmp_path):
    (tmp_path / "serve_4090").mkdir()
    assert demo.featured_runs(tmp_path) == {}


def test_headline_and_missing_policy(tmp_path):
    run = make_run(tmp_path)
    h = demo.headline(run, "table")
    assert h["heuristic_tpot_ms"] == pytest.approx(1.0) and h["policy_tpot_ms"] == pytest.approx(0.5)
    assert h["speedup"] == pytest.approx(2.0) and h["tokens_equivalent"] is True and h["repeats"] == 2
    assert h["model"] == "test/model" and h["scenario_name"] == "ragged" and h["policy"] == "table"
    missing = demo.headline(run, "model")
    assert missing["policy_tpot_ms"] is None and missing["speedup"] is None and missing["tokens_equivalent"] is None
    assert missing["heuristic_tpot_ms"] == pytest.approx(1.0)


def test_campaign_table(tmp_path):
    run = make_run(tmp_path)
    make_run(tmp_path, scenario="uniform")
    table = demo.campaign_table(run.parent)
    assert set(table.scenario) == {"ragged", "uniform"} and len(table) == 4
    assert set(table.family) == {"ragged", "uniform"}
    assert table[table.policy == "table"].speedup.tolist() == [2.0, 2.0]
    assert demo.campaign_table(tmp_path / "nowhere").empty
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py`
Expected: FAIL at import (`kernelscope.dashboard.demo` does not exist).

- [ ] **Step 4: Implement demo.py**

`kernelscope/dashboard/demo.py`:
```python
"""Demo page data: featured recordings, headline numbers, the token-race payload and the
per-batch decision card. Pure functions over recorded files; no Streamlit, no torch at import
(see docs/plan/2026-10-06-design-demo-page.md §4, §6)."""
import json
import math
from pathlib import Path

import pandas as pd

from kernelscope.dashboard import data, style

WORST_KEY = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
FAMILY_LABELS = {"ragged": "혼합 길이", "arrivals": "요청 도착", "uniform": "균일"}
POLICY_LABELS = {"heuristic": "FlashAttention 휴리스틱 (기본)", "table": "KernelScope · 실측표 선택",
                 "hybrid": "KernelScope · 혼합 정책", "model": "KernelScope · 성능 모델 선택",
                 "fixed8": "고정 분할 8", "fa2": "분할 없음 (fa2)"}
SYNTHETIC = "seeded_synthetic_token_ids"
DEFAULT_QUESTION = "Read the notes above. Summarize one monitoring practice in a short sentence.\nAnswer:"


# ---------------------------------------------------------------- recordings

def _family(manifest: dict, directory: Path) -> str | None:
    family = manifest.get("scenario_family")
    if family in FAMILY_LABELS:
        return family
    name = f"{Path(str(manifest.get('scenario', directory.name))).stem} {directory.name}".lower()
    for key in ("arrivals", "uniform", "ragged"):
        if key in name:
            return key
    return None


def _summary(run_dir) -> pd.DataFrame:
    path = Path(run_dir) / "summary.csv"
    return pd.read_csv(path).set_index("policy") if path.exists() else pd.DataFrame()


def _cell(summary: pd.DataFrame, policy: str, column: str):
    if policy in summary.index and column in summary and pd.notna(summary.at[policy, column]):
        return summary.at[policy, column]
    return None


def featured_runs(root) -> dict[str, Path]:
    """One scenario directory per family: natural-text prompts first, then the newest manifest."""
    candidates = {}
    for directory in data.find_dirs("serve", root):
        if not (directory / "summary.csv").exists():
            continue
        manifest = data.load_manifest(directory)
        family = _family(manifest, directory)
        if family is None:
            continue
        rank = (manifest.get("prompt_kind", SYNTHETIC) != SYNTHETIC, str(manifest.get("created_at", "")))
        if family not in candidates or rank > candidates[family][0]:
            candidates[family] = (rank, directory)
    return {family: candidates[family][1] for family in FAMILY_LABELS if family in candidates}


def headline(run_dir, policy: str) -> dict:
    """Hero numbers straight from summary.csv; None where the recording lacks a value."""
    run_dir = Path(run_dir)
    summary, manifest = _summary(run_dir), data.load_manifest(run_dir)
    heuristic, chosen = _cell(summary, "heuristic", "tpot_ms_mean"), _cell(summary, policy, "tpot_ms_mean")
    speedup = _cell(summary, policy, "speedup_vs_heuristic")
    if speedup is None and heuristic and chosen:
        speedup = heuristic / chosen
    equivalent, repeats = _cell(summary, policy, "tokens_equivalent"), _cell(summary, policy, "repeats")
    return {"heuristic_tpot_ms": None if heuristic is None else float(heuristic),
            "policy_tpot_ms": None if chosen is None else float(chosen),
            "speedup": None if speedup is None else float(speedup),
            "tokens_equivalent": None if equivalent is None else bool(equivalent),
            "repeats": None if repeats is None else int(repeats),
            "model": manifest.get("model"), "scenario_name": Path(str(manifest.get("scenario", run_dir.name))).stem,
            "created_at": manifest.get("created_at"), "policy": policy}


def campaign_table(campaign_dir) -> pd.DataFrame:
    """Every scenario × policy of one campaign: speedup, repeats, token agreement (controls included)."""
    columns = ["scenario", "family", "policy", "tpot_ms", "speedup", "repeats", "tokens_equivalent"]
    campaign_dir = Path(campaign_dir)
    rows = []
    for directory in sorted(campaign_dir.iterdir()) if campaign_dir.is_dir() else []:
        if not (directory / "summary.csv").exists():
            continue
        family = _family(data.load_manifest(directory), directory) or directory.name
        for policy, r in _summary(directory).iterrows():
            rows.append({"scenario": directory.name, "family": family, "policy": policy,
                         "tpot_ms": r.get("tpot_ms_mean"), "speedup": r.get("speedup_vs_heuristic"),
                         "repeats": r.get("repeats"), "tokens_equivalent": r.get("tokens_equivalent")})
    return pd.DataFrame(rows, columns=columns)


# ---------------------------------------------------------------- race payload

def _runs(policy_dir: Path) -> list[Path]:
    runs = sorted(p.parent for p in policy_dir.glob("repeat_*/steps.parquet"))
    return runs or ([policy_dir] if (policy_dir / "steps.parquet").exists() else [])


def policy_dirs(run_dir) -> list[str]:
    run_dir = Path(run_dir)
    return sorted(p.name for p in run_dir.iterdir() if p.is_dir() and _runs(p)) if run_dir.is_dir() else []


def _frames(run: Path) -> dict:
    out = {}
    for kind in ("steps", "tokens", "prefill"):
        path = run / f"{kind}.parquet"
        out[kind] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return out


def _t0_us(prefill: pd.DataFrame, tokens: pd.DataFrame) -> float:
    """Clock zero: when the first wave of requests (same arrival time) had all produced a first token."""
    if not prefill.empty and {"arrival_us", "first_token_us"} <= set(prefill):
        wave = prefill[prefill.arrival_us <= prefill.arrival_us.min() + 1.0]
        return float(wave.first_token_us.max())
    first = tokens[tokens.phase == "prefill"] if "phase" in tokens else tokens[tokens.position == 0]
    return float(first.t_us.max()) if not first.empty else float(tokens.t_us.min())


def _median_run(runs: list[Path]) -> tuple[int, dict]:
    loaded = [(i, _frames(r)) for i, r in enumerate(runs)]
    loaded = [(i, f) for i, f in loaded if not f["tokens"].empty]
    if not loaded:
        raise ValueError("the recording has no tokens.parquet")
    by_end = sorted(loaded, key=lambda item: item[1]["tokens"].t_us.max() - _t0_us(item[1]["prefill"], item[1]["tokens"]))
    return by_end[(len(by_end) - 1) // 2]


def _incremental(decode, ids: list[int]) -> list[str]:
    pieces, previous = [], ""
    for i in range(len(ids)):
        text = decode(ids[: i + 1])
        pieces.append(text[len(previous):] if len(text) >= len(previous) else "")
        previous = text
    return pieces


def race_payload(run_dir, policies, reference="heuristic", decode=None) -> dict:
    """The reference policy and one or more others of one recording as a replayable token race.

    Sequential measurements replayed on one clock (zero = the first wave fully admitted). ``decode``
    maps a token-id prefix to text; without it, or for synthetic-token prompts, texts are empty."""
    run_dir = Path(run_dir)
    policies = [reference] + [p for p in policies if p != reference]
    if len(policies) < 2:
        raise ValueError("a race needs the reference policy and at least one other policy")
    manifest, summary = data.load_manifest(run_dir), _summary(run_dir)
    natural = manifest.get("prompt_kind", SYNTHETIC) != SYNTHETIC and decode is not None
    panels, sequences = [], {}
    for policy in policies:
        runs = _runs(run_dir / policy)
        if not runs:
            raise FileNotFoundError(f"{run_dir / policy} has no recorded steps")
        index, frames = _median_run(runs)
        tokens, steps, prefill = frames["tokens"], frames["steps"], frames["prefill"]
        t0 = _t0_us(prefill, tokens)
        ms = lambda us: max(0.0, (float(us) - t0) / 1000.0)
        lengths = prefill.set_index("rid").prompt_len.to_dict() if not prefill.empty else {}
        requests, seq = [], {}
        for rid, group in tokens.sort_values(["rid", "position"]).groupby("rid", sort=True):
            ids = [int(t) for t in group.token]
            seq[int(rid)] = ids
            pieces = _incremental(decode, ids) if natural else [""] * len(ids)
            requests.append({"rid": int(rid), "prompt_len": int(lengths.get(rid, 0)), "featured": False,
                             "tokens": [{"t_ms": ms(t), "text": piece} for t, piece in zip(group.t_us, pieces)]})
        if requests:
            longest = min(requests, key=lambda r: (-r["prompt_len"], r["rid"]))
            shortest = min(requests, key=lambda r: (r["prompt_len"], r["rid"]))
            longest["featured"] = shortest["featured"] = True
        decode_tokens = tokens[tokens.phase == "decode"] if "phase" in tokens else tokens[tokens.position > 0]
        step_time = decode_tokens.groupby("step").t_us.max()
        step_rows = [{"step": int(r.step), "t_ms": ms(step_time.get(r.step, t0)), "num_splits": int(r.num_splits),
                      "attn_ms": float(r.attn_us) / 1000.0} for r in steps.itertuples()] if not steps.empty else []
        tpot = _cell(summary, policy, "tpot_ms_mean")
        panels.append({"name": policy, "label": POLICY_LABELS.get(policy, policy), "color": style.policy_color(policy),
                       "repeat_index": index, "tpot_ms": None if tpot is None else float(tpot),
                       "end_ms": ms(tokens.t_us.max()), "steps": step_rows, "requests": requests})
        sequences[policy] = seq
    agreement = {}
    for policy in policies[1:]:
        a, b = sequences[reference], sequences[policy]
        divergences = []
        for rid in sorted(set(a) | set(b)):
            x, y = a.get(rid, []), b.get(rid, [])
            position = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
            if position is None and len(x) != len(y):
                position = min(len(x), len(y))
            if position is not None:
                divergences.append({"rid": rid, "position": position})
        equivalent = _cell(summary, policy, "tokens_equivalent")
        total = len(set(a) | set(b))
        agreement[policy] = {"identical": not divergences, "requests_equal": total - len(divergences),
                             "requests_total": total, "first_divergence": divergences,
                             "summary_tokens_equivalent": None if equivalent is None else bool(equivalent)}
    reference_panel = panels[0]
    return {"scenario": {"name": Path(str(manifest.get("scenario", run_dir.name))).stem, "family": _family(manifest, run_dir),
                         "B": len(reference_panel["requests"]),
                         "lens": [r["prompt_len"] for r in reference_panel["requests"]],
                         "max_new_tokens": max((len(r["tokens"]) for r in reference_panel["requests"]), default=0),
                         "natural_text": natural, "model": manifest.get("model"),
                         "repeats": int(manifest.get("repeats", len(_runs(run_dir / reference))))},
            "t0_rule": "first_wave_admitted", "reference": reference, "policies": panels, "agreement": agreement}


def save_race(run_dir, payload) -> Path:
    path = Path(run_dir) / "race.json"
    path.write_text(json.dumps(payload, ensure_ascii=False))
    return path


def load_race(run_dir) -> dict | None:
    path = Path(run_dir) / "race.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and "policies" in value else None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py -v`
Expected: all PASS. If `test_race_payload_arrivals_clock_is_nonnegative` fails on the `1000.0` assertion, check that the fixture's first-wave is request 0 alone (arrival 0) so `t0 = first[0]`.

- [ ] **Step 6: Check the real recordings**

Run:
```bash
.venv/bin/python - <<'EOF'
from kernelscope.dashboard import demo
p = demo.race_payload("demo_data/serve_4090/hybrid_20261002/ragged", ["heuristic", "table", "hybrid"])
print(p["scenario"], [(x["name"], x["repeat_index"], round(x["end_ms"]), x["tpot_ms"]) for x in p["policies"]], p["agreement"]["table"]["identical"])
q = demo.race_payload("../kernelscope/results/serve_4090/demo_text_20261006/ragged", ["heuristic", "table", "hybrid"])
print(q["scenario"]["B"], q["scenario"]["natural_text"], [round(x["end_ms"]) for x in q["policies"]], q["agreement"]["hybrid"])
print(demo.featured_runs("demo_data"))
EOF
```
Expected: B=32, synthetic → `natural_text` False, heuristic `end_ms` roughly twice table's, `identical` True; `featured_runs` returns ragged/arrivals/uniform directories.

---

### Task 3: demo.py analysis helpers, compose_command extension, two figures

**Files:**
- Modify: `kernelscope/dashboard/demo.py` (append)
- Modify: `kernelscope/dashboard/lab.py` (`compose_command`)
- Modify: `kernelscope/dashboard/figures.py` (append two builders)
- Modify: `tests/test_dashboard_demo.py`, `tests/test_dashboard_lab.py`, `tests/test_dashboard_figures.py`

**Interfaces:**
- Produces:
  - `demo.cta_work(workload_key: str, variant: str, n_sm: int) -> dict` with keys `variant, kind, splits, ctas, keys (list[int]), longest_over_mean, ctas_per_sm`
  - `demo.decision_card(lens, n_heads, n_kv_heads, table_csv, machine_path, params_path, measured: pd.Series | None, cache_state="cold") -> list[dict]` rows `{policy, label, num_splits, kernel, kernel_us, speedup_vs_heuristic, note}` in the order heuristic, table, hybrid, model
  - `demo.load_demo_corpus(path) -> tuple[list[str], dict]`, `demo.compose_scenario(long_len, n_short, short_len, max_new_tokens, question, corpus, dataset) -> dict`, `demo.write_scenario(path, spec) -> Path`
  - `demo.verify_tail(output: str) -> dict`
  - `lab.compose_command(kind="bench", ..., workload=None, warmup=None, iters=None)`
  - `figures.cta_work_curves(launches: list[dict], theme="light", title=...)`, `figures.backend_bar(row: dict, theme="light", title=...)`
- Consumes: `kernelscope.model.geometry.build_launch/parse_variant/resolve_splits`, `kernelscope.model.hybrid.variant_for_splits`, `kernelscope.serve.dispatch.make_policy`, `kernelscope.model.machine.MachineSpec.from_json`, `kernelscope.model.params.ModelParams.from_json`, `kernelscope.workload.Workload`, `kernelscope.serve.scenarios.load_scenario`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_demo.py`:
```python
import yaml

from kernelscope.serve import scenarios
from kernelscope.workload import Workload

ROOT = Path(__file__).resolve().parents[1]
TABLE_CSV = ROOT / "demo_data" / "dispatch_paged_cold.csv"
MACHINE, PARAMS = ROOT / "machines" / "rtx4090.json", ROOT / "models" / "rtx4090.json"
MEASURED = pd.Series({"flashdecoding_paged": 1071.8, "fa2_paged": 1071.2, "fd_s8_paged": 283.5, "fd_s16_paged": 279.4})


def test_cta_work_heuristic_vs_split():
    base = demo.cta_work(demo.WORST_KEY, "flashdecoding_paged", 128)
    split = demo.cta_work(demo.WORST_KEY, "fd_s16_paged", 128)
    assert base["splits"] == 1 and base["ctas"] == 32 * 8 and base["ctas_per_sm"] == 2.0
    assert split["splits"] == 16 and split["ctas"] == 32 * 8 * 16
    assert max(base["keys"]) == 32768 and max(split["keys"]) == 32768 // 16
    assert base["longest_over_mean"] > 10 > split["longest_over_mean"] > 1
    assert all(isinstance(k, int) for k in split["keys"])


def test_decision_card_rows():
    lens = list(Workload.from_key(demo.WORST_KEY).lens())
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, PARAMS, MEASURED)
    assert [r["policy"] for r in rows] == ["heuristic", "table", "hybrid", "model"]
    heuristic, table, hybrid, model = rows
    assert heuristic["num_splits"] == 1 and heuristic["kernel"] == "flashdecoding_paged"
    assert heuristic["kernel_us"] == pytest.approx(1071.8) and heuristic["speedup_vs_heuristic"] == pytest.approx(1.0)
    assert "0.8" in heuristic["note"]
    assert table["num_splits"] == 16 and table["kernel"] == "fd_s16_paged"
    assert table["speedup_vs_heuristic"] == pytest.approx(1071.8 / 279.4) and "거리" in table["note"]
    for row in (hybrid, model):
        assert row["note"]                       # source / ranking text, or the reason it could not run
        assert row["num_splits"] is None or isinstance(row["num_splits"], int)


def test_decision_card_without_params(tmp_path):
    lens = list(Workload.from_key(demo.WORST_KEY).lens())
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, tmp_path / "missing.json", MEASURED)
    assert rows[0]["num_splits"] == 1 and rows[1]["num_splits"] == 16
    assert rows[2]["num_splits"] is None and rows[2]["note"].startswith("선택 불가")
    assert rows[3]["num_splits"] is None and rows[3]["note"].startswith("선택 불가")
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, PARAMS, None)
    assert rows[1]["kernel_us"] is None and rows[1]["speedup_vs_heuristic"] is None


def test_compose_scenario_roundtrip(tmp_path):
    corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
    assert len(corpus) >= 3 and dataset["id"]
    spec = demo.compose_scenario(16384, 7, 384, 32, "What is measured?\nAnswer:", corpus, dataset)
    path = demo.write_scenario(tmp_path / "live" / "scenario.yaml", spec)
    requests, loaded_dataset = scenarios.load_scenario(path)
    assert len(requests) == 8 and requests[0].prompt_len == 16384 and requests[7].prompt_len == 384
    assert requests[0].max_new_tokens == 32 and requests[0].prompt_suffix == "What is measured?\nAnswer:"
    assert requests[0].prompt_recipe == "corpus_repeat" and loaded_dataset["scenario_family"] == "ragged"
    assert yaml.safe_load(path.read_text())["requests"][1]["count"] == 7


@pytest.mark.parametrize("bad", [dict(long_len=0), dict(n_short=0), dict(short_len=-1), dict(max_new_tokens=0), dict(question="  ")])
def test_compose_scenario_rejects(bad):
    corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
    args = dict(long_len=8192, n_short=3, short_len=256, max_new_tokens=8, question="Q:", corpus=corpus, dataset=dataset)
    args.update(bad)
    with pytest.raises(ValueError):
        demo.compose_scenario(**args)


def test_verify_tail():
    assert demo.verify_tail("header\n  row PASS\n{\"PASS\": 111}\n") == {"PASS": 111}
    assert demo.verify_tail("{\"PASS\": 3, \"FAIL\": 1}") == {"PASS": 3, "FAIL": 1}
    assert demo.verify_tail("no json here") == {}
```

Append to `tests/test_dashboard_lab.py`:
```python
def test_compose_bench_single_workload():
    from kernelscope.dashboard import lab
    argv = lab.compose_command("bench", python="py", workload="decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal",
                               plugins=["fd_s16_paged", "flashinfer_paged_cudacore"], results="out", cache_state="cold",
                               warmup=5, iters=20)
    assert argv[:4] == ["py", "-m", "kernelscope.cli", "bench"]
    assert "--grid" not in argv and argv[argv.index("--workload") + 1].startswith("decode_B1")
    assert argv[argv.index("--warmup") + 1] == "5" and argv[argv.index("--iters") + 1] == "20"
    grid = lab.compose_command("bench", grid="grids/g.yaml", plugins=["fa2_paged"], results="out")
    assert "--grid" in grid and "--warmup" not in grid
```

Append to `tests/test_dashboard_figures.py`:
```python
def test_cta_work_curves_and_backend_bar():
    from kernelscope.dashboard import figures
    fig = figures.cta_work_curves([{"label": "a", "color": "#2a78d6", "keys": [5, 0, 3, 1]},
                                   {"label": "b", "color": "#1baf7a", "keys": [2, 2, 2, 2]}], "light")
    assert [trace.name for trace in fig.data] == ["a", "b"]
    assert list(fig.data[0].y) == [5, 3, 1]                      # zero-work CTAs are dropped, sorted busiest first
    row = {"heuristic_us": 1071.8, "best_fa2_kernel": "fd_s16_paged", "best_fa2_us": 279.4,
           "flashinfer_tensorcore_us": 670.9, "flashinfer_cudacore_us": float("nan")}
    bar = figures.backend_bar(row, "dark")
    assert len(bar.data) == 1 and len(bar.data[0].x) == 3        # the NaN backend is left out
    assert bar.layout.paper_bgcolor == "#1a1a19"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py tests/test_dashboard_lab.py tests/test_dashboard_figures.py 2>&1 | tail -15`
Expected: the new tests FAIL with AttributeError / TypeError (missing functions, unexpected keyword `workload`).

- [ ] **Step 3: Append the analysis helpers to demo.py**

```python
# ---------------------------------------------------------------- why / how

def cta_work(workload_key: str, variant: str, n_sm: int) -> dict:
    """Keys per CTA of one launch (hardware linear order) and the imbalance a presenter quotes.
    Derived from lengths and the launch geometry, not measured."""
    from kernelscope.model.geometry import build_launch, parse_variant
    from kernelscope.workload import Workload
    launch = build_launch(Workload.from_key(workload_key), parse_variant(variant), n_sm)
    keys = [int(k) for k in launch.keys]
    busy = [k for k in keys if k > 0]
    return {"variant": variant, "kind": launch.kind, "splits": int(launch.splits), "ctas": len(keys), "keys": keys,
            "longest_over_mean": (max(busy) / (sum(busy) / len(busy))) if busy else 0.0,
            "ctas_per_sm": len(keys) / n_sm}


def decision_card(lens, n_heads, n_kv_heads, table_csv, machine_path, params_path, measured, cache_state="cold") -> list[dict]:
    """What each policy picks for one batch and what that variant measured.
    A policy that cannot run (missing files, backend) becomes a row with a note, never an exception."""
    from kernelscope.model.geometry import parse_variant, resolve_splits
    from kernelscope.model.hybrid import variant_for_splits
    from kernelscope.serve.dispatch import make_policy
    from kernelscope.workload import Workload
    machine = params = None
    setup_error = None
    try:
        from kernelscope.model.machine import MachineSpec
        from kernelscope.model.params import ModelParams
        machine, params = MachineSpec.from_json(machine_path), ModelParams.from_json(params_path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        setup_error = f"{type(exc).__name__}: {exc}"
    lens = [int(x) for x in lens]
    workload = Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=n_heads, H_kv=n_kv_heads, d=128,
                        dtype="float16", kv_lens=tuple(lens))
    n_sm = machine.n_sm if machine is not None else 128

    def kernel_us(name):
        value = measured.get(name) if measured is not None else None
        return None if value is None or pd.isna(value) else float(value)

    heuristic_us = kernel_us("flashdecoding_paged")
    specs = [("heuristic", "heuristic"), ("table", f"table:{table_csv}"), ("hybrid", f"hybrid:{table_csv}:0.2"), ("model", "model")]
    rows = []
    for name, spec in specs:
        row = {"policy": name, "label": POLICY_LABELS.get(name, name), "num_splits": None, "kernel": None,
               "kernel_us": None, "speedup_vs_heuristic": None, "note": ""}
        try:
            if name in ("hybrid", "model") and setup_error:
                raise ValueError(setup_error)
            policy = make_policy(spec, machine, params, cache_state=cache_state)
            splits = int(policy.choose(lens, n_heads, n_kv_heads))
            if name == "heuristic":
                resolved = resolve_splits(parse_variant("flashdecoding_paged"), workload, n_sm)
                row.update(num_splits=resolved, kernel="flashdecoding_paged", kernel_us=heuristic_us,
                           note=f"라이브러리 규칙: CTA {len(lens) * n_kv_heads}개 ≥ 0.8 × {2 * n_sm} → 분할 {resolved}")
            else:
                kernel = variant_for_splits(splits)
                row.update(num_splits=splits, kernel=kernel, kernel_us=kernel_us(kernel))
                last = getattr(policy, "last", None)
                if name == "table" and isinstance(last, dict):
                    row["note"] = f"가장 가까운 측정 셀 {last.get('workload_key')} · 거리 {float(last.get('distance', 0)):.2f}"
                elif name == "hybrid" and isinstance(last, dict):
                    ranked = last.get("model") or {}
                    best = min(ranked, key=ranked.get) if ranked else "-"
                    row["note"] = f"출처 {last.get('source')} · 모델 1위 {best} · 테이블 {last.get('pick')}"
                elif name == "model" and isinstance(last, list) and last:
                    row["note"] = "예측 순위 " + " < ".join(f"{k} {t:,.0f}µs" for k, t in last[:3])
            if row["kernel_us"] and heuristic_us:
                row["speedup_vs_heuristic"] = heuristic_us / row["kernel_us"]
        except (ValueError, KeyError, OSError, RuntimeError, TypeError) as exc:
            row["note"] = f"선택 불가: {exc}"
        rows.append(row)
    return rows


# ---------------------------------------------------------------- live measurement

def load_demo_corpus(path) -> tuple[list[str], dict]:
    import yaml
    spec = yaml.safe_load(Path(path).read_text())
    return list(spec.get("corpus", [])), dict(spec.get("dataset", {}))


def compose_scenario(long_len, n_short, short_len, max_new_tokens, question, corpus, dataset) -> dict:
    """A ragged batch the live panel describes, in the shape scenarios.load_scenario accepts."""
    for name, value in (("long_len", long_len), ("n_short", n_short), ("short_len", short_len), ("max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonempty text")
    if not corpus:
        raise ValueError("corpus must be a nonempty list of paragraphs")
    group = {"max_new_tokens": max_new_tokens, "arrival_step": 0, "prompt_recipe": "corpus_repeat", "prompt_suffix": question}
    return {"dataset": {**dataset, "scenario_family": "ragged", "construction_note": "composed in the dashboard live panel"},
            "corpus": list(corpus),
            "requests": [{"count": 1, "prompt_len": long_len, **group}, {"count": n_short, "prompt_len": short_len, **group}]}


def write_scenario(path, spec) -> Path:
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False))
    return path


def verify_tail(output: str) -> dict:
    """The {"PASS": n[, "FAIL": m]} line kernelscope verify prints last; {} when absent."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return {}
```

- [ ] **Step 4: Extend lab.compose_command**

In `kernelscope/dashboard/lab.py` change the signature to add `workload=None, warmup=None, iters=None` after `cache_state="cold"`, and replace the `else:` branch with:
```python
    else:
        argv += ["bench"]
        argv += ["--workload", str(workload)] if workload is not None else ["--grid", str(grid)]
        argv += ["--plugins", ",".join(plugins), "--results", str(results), "--cache-state", cache_state]
        if warmup is not None:
            argv += ["--warmup", str(warmup)]
        if iters is not None:
            argv += ["--iters", str(iters)]
```
Update the module docstring sentence about `compose_command` to mention the single-workload form.

- [ ] **Step 5: Append the two figure builders**

Append to `kernelscope/dashboard/figures.py` (it already imports `go`, `pd`, `np`, `style`; add `import math` at the top):
```python
def cta_work_curves(launches, theme="light", title="CTA 작업량 · 블록별 KV keys (바쁜 순)"):
    """launches: dicts with label, color, keys (demo.cta_work). Zero-work CTAs are dropped."""
    fig = go.Figure()
    for item in launches:
        keys = sorted((int(k) for k in item["keys"] if k > 0), reverse=True)
        fig.add_trace(go.Scatter(x=list(range(1, len(keys) + 1)), y=keys, mode="lines", name=item["label"],
                                 line={"color": item["color"], "width": 2},
                                 hovertemplate="CTA #%{x}<br>%{y:,} keys<extra>" + str(item["label"]) + "</extra>"))
    style.layout(fig, theme, title)
    fig.update_xaxes(title_text="CTA 순위 (로그)", type="log")
    fig.update_yaxes(title_text="KV keys / CTA (로그)", type="log")
    return fig


def backend_bar(row, theme="light", title="같은 셀 · 휴리스틱, 최선 FA2 분할, FlashInfer (µs)"):
    """One kernel cell across backends (lab.backend_comparison row). Missing backends are left out."""
    items = [("라이브러리 휴리스틱", row.get("heuristic_us")),
             (f"최선 FA2 분할 · {row.get('best_fa2_kernel')}", row.get("best_fa2_us")),
             ("FlashInfer tensor-core", row.get("flashinfer_tensorcore_us")),
             ("FlashInfer CUDA-core", row.get("flashinfer_cudacore_us"))]
    items = [(label, float(v)) for label, v in items if v is not None and not (isinstance(v, float) and math.isnan(v))]
    labels, values = [i[0] for i in items], [i[1] for i in items]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=style.CATEGORICAL[theme][0],
                           text=[f"{v:,.0f}" for v in values], textposition="auto",
                           hovertemplate="%{y}<br>%{x:,.1f} µs<extra></extra>"))
    style.layout(fig, theme, title)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Kernel time (µs)")
    return fig
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py tests/test_dashboard_lab.py tests/test_dashboard_figures.py -v 2>&1 | tail -30`
Expected: all PASS. `test_decision_card_rows` may take a few seconds (the surrogate model ranks candidates); that is fine. If the model row's note says `선택 불가`, print the exception and check `models/rtx4090.json` covers `cold`.

---

### Task 4: Race component and Demo scenes 0–1 (hero, race, live panel)

**Files:**
- Create: `kernelscope/dashboard/race.html`
- Modify: `kernelscope/dashboard/demo.py` (append `render_race_html`)
- Rewrite: `dashboard/demo_page.py`
- Modify: `tests/test_dashboard_demo.py`, `tests/test_dashboard_app.py`

**Interfaces:**
- Produces: `demo.render_race_html(payload: dict, policy_a: str, policy_b: str, theme="light") -> str`; `demo.RACE_TEMPLATE: Path`.
- Consumes: Task 2 payload schema; `lab.compose_command("serve_run", ...)`, `lab.launch`, `lab.tail`, `lab.shell_line`; `style.PAGE_CSS`, `style.current_theme`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_dashboard_demo.py`:
```python
def test_render_race_html_embeds_two_policies(tmp_path):
    run = make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    payload = demo.race_payload(run, ["heuristic", "table", "hybrid"])
    html = demo.render_race_html(payload, "heuristic", "hybrid", "dark")
    assert "__PAYLOAD__" not in html and html.count("<script>") == 1
    start, end = html.index("const P = ") + len("const P = "), html.index(";\n", html.index("const P = "))
    embedded = json.loads(html[start:end])
    assert [p["name"] for p in embedded["policies"]] == ["heuristic", "hybrid"]
    assert embedded["agreement"]["identical"] is True and embedded["theme"] == "dark"
    assert embedded["colors"]["bg"] == "#1a1a19" and embedded["policies"][1]["color"] == "#9085e9"
    assert payload["policies"][2]["color"] == "#4a3aa7"          # the cached payload is not mutated
    with pytest.raises(KeyError):
        demo.render_race_html(payload, "heuristic", "model")


def test_race_template_has_one_payload_slot():
    text = demo.RACE_TEMPLATE.read_text(encoding="utf-8")
    assert text.count("__PAYLOAD__") == 1 and "</script>" in text and 'id="play"' in text
```

Append to `tests/test_dashboard_app.py`:
```python
from demo_fixtures import make_run


def test_demo_page_renders_hero_and_race(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "같은 답, 더 빠른 토큰."
    assert any("2.00×" in metric.value for metric in app.metric)
    assert any(metric.value == "일치" for metric in app.metric)
    assert any("생성 토큰 일치" in item.value for item in app.success)
    policy = next(radio for radio in app.radio if radio.label == "KernelScope 정책")
    policy.set_value("hybrid")
    app.run()
    assert not app.exception


def test_demo_page_without_pair_is_informative(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic",))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert any("비교할 정책 쌍이 없습니다" in item.value for item in app.info)


def test_demo_page_empty_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("서빙 기록이 없습니다" in item.value for item in app.info)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py -k "render or template" tests/test_dashboard_app.py -k demo 2>&1 | tail -8`
Expected: FAIL (`render_race_html` missing; Demo page lacks metrics/radio).

- [ ] **Step 3: Write the race template**

`kernelscope/dashboard/race.html`:
```html
<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<style>
  :root { --bg: #fcfcfb; --ink: #0b0b0b; --muted: #52514e; --grid: #e8e7e3; }
  body { margin: 0; background: var(--bg); color: var(--ink); font: 13px/1.5 Inter, "Noto Sans KR", sans-serif; }
  .bar { display: flex; gap: 8px; align-items: center; margin: 2px 0 12px; }
  .bar .label { color: var(--muted); margin-left: auto; font-variant-numeric: tabular-nums; }
  button { font: inherit; padding: 4px 12px; border-radius: 999px; border: 1px solid var(--grid); background: transparent; color: var(--ink); cursor: pointer; }
  button.active { border-color: var(--ink); font-weight: 650; }
  .wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .pane { border: 1px solid var(--grid); border-radius: 14px; padding: 14px 16px; min-width: 0; }
  .pane h3 { margin: 0 0 8px; font-size: 15px; display: flex; align-items: center; gap: 8px; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
  .stats { display: flex; gap: 22px; margin-bottom: 8px; font-variant-numeric: tabular-nums; }
  .stats div span { display: block; font-size: 11px; letter-spacing: .08em; color: var(--muted); }
  .stats div b { font-size: 22px; font-weight: 700; }
  .grid { display: grid; grid-template-columns: repeat(16, 1fr); gap: 3px; margin: 6px 0 10px; }
  .cell { height: 7px; background: var(--grid); border-radius: 2px; overflow: hidden; }
  .cell i { display: block; height: 100%; width: 0; }
  .text { border-left: 3px solid var(--grid); padding-left: 10px; margin: 6px 0; min-height: 60px; white-space: pre-wrap; word-break: break-word; }
  .text small { display: block; color: var(--muted); font-size: 11px; margin-bottom: 2px; }
  .strip { display: flex; gap: 1px; height: 30px; align-items: flex-end; margin-top: 8px; }
  .strip i { flex: 1; background: var(--grid); min-height: 2px; }
  .strip i.on { background: var(--series); }
  .meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 11px; margin-top: 3px; font-variant-numeric: tabular-nums; }
  .flag { min-height: 22px; margin-top: 6px; font-weight: 700; }
  .foot { margin-top: 12px; font-size: 13px; }
</style></head>
<body>
<div class="bar">
  <button id="play">▶ 재생</button><button id="reset">↺ 처음부터</button>
  <span style="color:var(--muted)">속도</span>
  <button data-speed="0.25">0.25×</button><button data-speed="0.5" class="active">0.5×</button><button data-speed="1">1×</button>
  <span class="label" id="label"></span>
</div>
<div class="wrap" id="wrap"></div>
<div class="foot" id="foot"></div>
<script>
const P = __PAYLOAD__;
const root = document.documentElement;
for (const [k, v] of Object.entries(P.colors || {})) root.style.setProperty("--" + k, v);
const fmtMs = (ms) => ms.toLocaleString("en-US", {maximumFractionDigits: 0}) + " ms";
const maxAttn = Math.max(1e-6, ...P.policies.flatMap(p => p.steps.map(s => s.attn_ms)));
const totalEnd = Math.max(...P.policies.map(p => p.end_ms));
const maxTokens = Math.max(1, P.scenario.max_new_tokens);

const panes = P.policies.map((policy, idx) => {
  const el = document.createElement("div"); el.className = "pane"; el.style.setProperty("--series", policy.color);
  const featured = policy.requests.filter(r => r.featured);
  el.innerHTML = `
    <h3><span class="dot" style="background:${policy.color}"></span>${policy.label}</h3>
    <div class="stats"><div><span>경과</span><b class="clock">0 ms</b></div><div><span>TOKENS / S</span><b class="rate">0</b></div>
      <div><span>분할 수</span><b class="splits">–</b></div></div>
    <div class="grid">${policy.requests.map(() => '<div class="cell"><i style="background:' + policy.color + '"></i></div>').join("")}</div>
    ${featured.map(r => `<div class="text" data-rid="${r.rid}"><small>요청 ${r.rid} · 프롬프트 ${r.prompt_len.toLocaleString()} tokens</small><span></span></div>`).join("")}
    <div class="strip">${policy.steps.map(() => '<i></i>').join("")}</div>
    <div class="meta"><span>step별 분할 수 · attention 시간(실측)</span><span>${policy.steps.length} steps</span></div>
    <div class="flag"></div>`;
  document.getElementById("wrap").appendChild(el);
  const events = [];
  policy.requests.forEach((r, i) => r.tokens.forEach(t => events.push({t: t.t_ms, i, text: t.text, rid: r.rid})));
  events.sort((a, b) => a.t - b.t);
  const strip = [...el.querySelectorAll(".strip i")];
  policy.steps.forEach((s, k) => { strip[k].style.height = Math.max(6, 100 * s.attn_ms / maxAttn) + "%"; strip[k].title = `step ${s.step} · 분할 ${s.num_splits} · attention ${s.attn_ms.toFixed(2)} ms`; });
  return {policy, el, events, cursor: 0, counts: new Array(policy.requests.length).fill(0), shown: 0,
          cells: [...el.querySelectorAll(".cell i")], texts: Object.fromEntries([...el.querySelectorAll(".text")].map(d => [d.dataset.rid, d.querySelector("span")])),
          clock: el.querySelector(".clock"), rate: el.querySelector(".rate"), splits: el.querySelector(".splits"), strip, flag: el.querySelector(".flag"), finished: false};
});

let elapsed = 0, speed = 0.5, playing = false, last = null;
function reset() {
  elapsed = 0; last = null;
  for (const pane of panes) {
    pane.cursor = 0; pane.shown = 0; pane.counts.fill(0); pane.finished = false;
    pane.cells.forEach(c => c.style.width = "0");
    Object.values(pane.texts).forEach(s => s.textContent = "");
    pane.strip.forEach(i => i.classList.remove("on"));
    pane.clock.textContent = "0 ms"; pane.rate.textContent = "0"; pane.splits.textContent = "–"; pane.flag.textContent = "";
  }
  draw();
}
function draw() {
  for (const pane of panes) {
    const ev = pane.events;
    while (pane.cursor < ev.length && ev[pane.cursor].t <= elapsed) {
      const e = ev[pane.cursor++];
      pane.counts[e.i] += 1; if (e.t > 0) pane.shown += 1;
      pane.cells[e.i].style.width = Math.min(100, 100 * pane.counts[e.i] / maxTokens) + "%";
      if (pane.texts[e.rid] !== undefined) pane.texts[e.rid].textContent += e.text;
    }
    const shownMs = Math.min(elapsed, pane.policy.end_ms);
    pane.clock.textContent = fmtMs(shownMs);
    pane.rate.textContent = shownMs > 0 ? (1000 * pane.shown / shownMs).toFixed(0) : "0";
    let current = null;
    pane.policy.steps.forEach((s, k) => { if (s.t_ms <= elapsed) { pane.strip[k].classList.add("on"); current = s; } });
    if (current) pane.splits.textContent = current.num_splits === 0 ? "라이브러리" : String(current.num_splits);
    if (!pane.finished && elapsed >= pane.policy.end_ms) {
      pane.finished = true;
      const tpot = pane.policy.tpot_ms != null ? ` · TPOT ${pane.policy.tpot_ms.toFixed(1)} ms` : "";
      pane.flag.textContent = `완료 · ${fmtMs(pane.policy.end_ms)}${tpot}`;
    }
  }
  document.getElementById("label").textContent = `${fmtMs(Math.min(elapsed, totalEnd))} / ${fmtMs(totalEnd)} · 재생 ${speed}× · 순차 측정, 동시 재생`;
}
function frame(now) {
  if (!playing) return;
  if (last !== null) elapsed = Math.min(totalEnd + 400, elapsed + (now - last) * speed);
  last = now; draw();
  if (elapsed >= totalEnd + 400) { playing = false; document.getElementById("play").textContent = "▶ 다시 재생"; return; }
  requestAnimationFrame(frame);
}
document.getElementById("play").addEventListener("click", () => {
  if (elapsed >= totalEnd + 400) reset();
  playing = !playing; last = null;
  document.getElementById("play").textContent = playing ? "⏸ 일시정지" : "▶ 재생";
  if (playing) requestAnimationFrame(frame);
});
document.getElementById("reset").addEventListener("click", () => { playing = false; document.getElementById("play").textContent = "▶ 재생"; reset(); });
document.querySelectorAll("[data-speed]").forEach(b => b.addEventListener("click", () => {
  speed = parseFloat(b.dataset.speed); document.querySelectorAll("[data-speed]").forEach(x => x.classList.toggle("active", x === b)); draw();
}));
const A = P.agreement || {};
const foot = document.getElementById("foot");
if (A.requests_total) {
  foot.textContent = A.identical ? `생성 토큰 일치 · ${A.requests_equal}/${A.requests_total} 요청 — 두 정책의 출력이 같습니다.`
    : `생성 토큰 불일치 · ${A.first_divergence.length}건 (요청 ${A.first_divergence.slice(0, 4).map(d => d.rid + "@" + d.position).join(", ")})`;
}
if (!P.scenario.natural_text) foot.textContent += "  합성 토큰 기록이라 글 대신 토큰 수만 채웁니다.";
reset();
</script>
</body></html>
```

- [ ] **Step 4: Append render_race_html to demo.py**

```python
# ---------------------------------------------------------------- race component

RACE_TEMPLATE = Path(__file__).with_name("race.html")


def render_race_html(payload: dict, policy_a: str, policy_b: str, theme: str = "light") -> str:
    """The race template with exactly two policies of ``payload`` embedded as JSON (payload untouched)."""
    panels = {p["name"]: p for p in payload["policies"]}
    if policy_a not in panels or policy_b not in panels:
        raise KeyError(f"race payload lacks {policy_a!r} or {policy_b!r}; it has {sorted(panels)}")
    agreement = payload.get("agreement", {})
    pair = [dict(panels[name], color=style.policy_color(name, theme)) for name in (policy_a, policy_b)]
    subset = {**payload, "policies": pair, "theme": theme,
              "agreement": agreement.get(policy_b) or agreement.get(policy_a) or {},
              "colors": {"bg": style.SURFACE[theme], "ink": style.TEXT[theme][0], "muted": style.TEXT[theme][1],
                         "grid": style.GRID[theme]}}
    html = RACE_TEMPLATE.read_text(encoding="utf-8")
    if html.count("__PAYLOAD__") != 1:
        raise ValueError("race.html must contain exactly one __PAYLOAD__ slot")
    return html.replace("__PAYLOAD__", json.dumps(subset, ensure_ascii=False).replace("</", "<\\/"))
```

- [ ] **Step 5: Write the Demo page (scenes 0–1 and the live panel; scenes 2–4 come in Task 5)**

`dashboard/demo_page.py`:
```python
"""Demo page: fixed-order scenes for the presentation. Lab (lab_page.py) keeps the exploration tabs.

Scenes: 0 headline tiles · 1 token race (sequential measurements replayed on one clock) + live
measurement panel · 2 why (batch shape, CTA work, measured kernels, op breakdown) · 3 how (kernel map,
decision card, other backends, plugin bench) · 4 verification (campaign table, divergence, verify)."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from kernelscope.dashboard import data, demo, figures, lab, style

theme = style.current_theme()
root = data.results_root()
PY = sys.executable
GPU_HOST = Path("/usr/bin/nvidia-smi").exists() or Path("/usr/local/bin/nvidia-smi").exists()
TABLE_CSV = ROOT / "demo_data" / "dispatch_paged_cold.csv"
MACHINE, PARAMS = ROOT / "machines" / "rtx4090.json", ROOT / "models" / "rtx4090.json"
STAMP = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")

st.markdown(style.PAGE_CSS, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def featured(root_str: str) -> dict:
    return {family: str(path) for family, path in demo.featured_runs(Path(root_str)).items()}


@st.cache_data(ttl=300, show_spinner="기록을 읽는 중…")
def race(run_dir: str, policies: tuple) -> dict | None:
    cached = demo.load_race(run_dir)
    if cached and {p["name"] for p in cached["policies"]} >= set(policies):
        return cached
    decode = None
    try:  # without torch or the local model snapshot the race shows token counts instead of text
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        tokenizer = load_tokenizer(snapshot_dir(data.load_manifest(run_dir).get("model")))
        decode = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True)
    except Exception:
        decode = None
    try:
        return demo.race_payload(run_dir, list(policies), decode=decode)
    except (FileNotFoundError, ValueError, OSError):
        return None


def chart(fig, key):
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def fmt(value, suffix="", digits=1):
    return "—" if value is None or pd.isna(value) else f"{value:,.{digits}f}{suffix}"


def job_panel(prefix: str, argv, start_label: str, before_launch, done_check):
    """Shared launcher: command preview, doctor gate, background job, log tail. Returns True once done."""
    st.code(lab.shell_line(argv), language="bash")
    job, log_path = st.session_state.get(f"{prefix}_job"), st.session_state.get(f"{prefix}_log")
    running = job is not None and job.poll() is None
    if st.button(start_label, disabled=not GPU_HOST or running, key=f"{prefix}_start"):
        doctor = subprocess.run([PY, "-m", "kernelscope.cli", "serve", "doctor"], cwd=ROOT, capture_output=True, text=True)
        if doctor.returncode != 0:
            st.error("serve doctor 실패 · 다른 GPU 프로세스가 있거나 모델·의존성이 준비되지 않았습니다.")
            st.code((doctor.stdout or doctor.stderr)[-1500:], language="text")
        else:
            log_path = before_launch()
            st.session_state[f"{prefix}_job"] = lab.launch(argv, log_path, cwd=ROOT)
            st.session_state[f"{prefix}_log"] = str(log_path)
            st.rerun()
    if log_path:
        st.caption(("실행 중 · " if running else f"종료 (코드 {job.returncode}) · " if job is not None else "기록 · ") + str(log_path))
        st.code(lab.tail(log_path, 30) or "(아직 출력이 없습니다)", language="text")
        if st.button("로그 새로고침", key=f"{prefix}_refresh"):
            st.rerun()
        if not running and done_check():
            return True
    if not GPU_HOST:
        st.caption("이 장비에는 nvidia-smi가 없어 실행 버튼이 비활성입니다. 명령을 복사해 연구실 호스트에서 돌리세요.")
    return False


# ---------------------------------------------------------------- scene 0 · headline
runs = featured(str(root))
st.markdown('<div class="eyebrow">KERNELSCOPE · DEMO</div>', unsafe_allow_html=True)
st.title("같은 답, 더 빠른 토큰.")
st.markdown('<div class="hero-note">길이가 다른 요청이 한 배치에 섞이면 attention 커널이 GPU를 다 쓰지 못합니다. '
            'KernelScope는 매 decode step의 KV 분할 수를 골라, 출력은 그대로 두고 토큰 간격을 줄입니다.</div>',
            unsafe_allow_html=True)
if not runs:
    st.info("서빙 기록이 없습니다. 결과 루트(KERNELSCOPE_RESULTS) 또는 demo_data 아래 serve_4090/<campaign>/<scenario>/summary.csv가 필요합니다.")
    st.stop()
hero_family = "ragged" if "ragged" in runs else next(iter(runs))
hero_run = runs[hero_family]
hero_candidates = [p for p in ("table", "hybrid", "model", "fixed8") if p in demo.policy_dirs(hero_run)]
hero = demo.headline(hero_run, hero_candidates[0] if hero_candidates else "heuristic")
tiles = st.columns(3)
tiles[0].metric("토큰 간격 TPOT · 기본 → KernelScope", f"{fmt(hero['heuristic_tpot_ms'])} → {fmt(hero['policy_tpot_ms'])} ms",
                help="요청이 토큰을 받는 평균 간격. 왼쪽은 FlashAttention 휴리스틱, 오른쪽은 KernelScope 정책.")
tiles[1].metric("배율", fmt(hero["speedup"], "×", 2), help="heuristic TPOT ÷ 정책 TPOT, 같은 실험의 반복 평균.")
tiles[2].metric("생성 토큰 일치", {True: "일치", False: "불일치", None: "—"}[hero["tokens_equivalent"]],
                help="모든 반복에서 정책의 생성 토큰이 heuristic과 같았는지.")
st.caption(f"실측 · {hero['model']} · {demo.FAMILY_LABELS.get(hero_family, hero_family)} ({hero['scenario_name']}) · 정책 {hero['policy']} · "
           f"반복 {hero['repeats']}회 · 측정 {str(hero['created_at'] or '')[:10]} · {hero_run}")

# ---------------------------------------------------------------- scene 1 · race
st.divider()
st.subheader("1 · 경주 — 같은 배치, 같은 모델, 다른 커널 선택")
st.write("왼쪽은 FlashAttention의 기본 분할 수 규칙, 오른쪽은 KernelScope의 선택입니다. 같은 GPU에서 **순차로 측정한 기록**을 "
         "타임스탬프대로 **동시에 재생**합니다. 두 정책을 동시에 돌리면 측정이 서로를 오염시키므로 그렇게 하지 않습니다.")
pick = st.columns([1.2, 1.6, 2])
family = pick[0].radio("배치 종류", list(runs), format_func=lambda f: demo.FAMILY_LABELS.get(f, f), horizontal=True, key="race_family")
run_dir = runs[family]
with st.expander("다른 기록 고르기"):
    all_runs = sorted({str(d) for d in data.find_dirs("serve", root) if (d / "summary.csv").exists()}
                      | set(st.session_state.get("demo_live_runs", [])))
    run_dir = st.selectbox("기록 폴더", all_runs, index=all_runs.index(run_dir) if run_dir in all_runs else 0, key="race_run")
available = demo.policy_dirs(run_dir)
candidates = [p for p in ("table", "hybrid", "model", "fixed8", "fa2") if p in available]
if "heuristic" not in available or not candidates:
    st.info("이 기록에는 heuristic과 비교할 정책 쌍이 없습니다. 다른 기록을 고르세요.")
else:
    policy = pick[1].radio("KernelScope 정책", candidates, horizontal=True, key="race_policy")
    payload = race(str(run_dir), tuple(["heuristic", *candidates]))
    if payload is None:
        st.warning("이 기록의 토큰 parquet를 읽을 수 없습니다.")
    else:
        components.html(demo.render_race_html(payload, "heuristic", policy, theme), height=560, scrolling=False)
        agreement = payload["agreement"].get(policy, {})
        if agreement.get("identical"):
            st.success(f"생성 토큰 일치 · {agreement['requests_equal']}/{agreement['requests_total']} 요청 — 재생한 반복에서 두 정책이 같은 토큰을 만들었습니다. "
                       f"전체 반복 기준: {'일치' if agreement.get('summary_tokens_equivalent') else '불일치 또는 미기록'}.")
        else:
            first = agreement.get("first_divergence", [])
            st.warning(f"생성 토큰 불일치 {len(first)}건 · 요청 {', '.join(str(d['rid']) for d in first[:5])} — 분기 사건의 tie 분류는 Lab의 05 Policy lab에서 봅니다.")
        pick[2].caption(f"{payload['scenario']['B']}개 요청 · 프롬프트 {min(payload['scenario']['lens']):,}~{max(payload['scenario']['lens']):,} tokens · "
                        f"{payload['scenario']['max_new_tokens']}토큰 생성 · 반복 {payload['scenario']['repeats']}회 중 중앙값 실행을 재생")
        if not payload["scenario"]["natural_text"]:
            st.caption("이 기록은 합성 토큰 프롬프트이거나 토크나이저가 없어 글 대신 토큰 수로 재생합니다. 자연어 기록(demo_text_ragged)은 글이 보입니다.")
        st.caption("재생 · 시계 0 = 첫 요청 묶음의 prefill이 모두 끝난 시점. 띠는 step별 분할 수와 attention 시간(실측).")

with st.expander("라이브 측정 · 배치를 구성해 GPU 호스트에서 재고 바로 재생 (모델 로드 포함 1~2분 · 발표 중에는 기록 재생을 권함)"):
    c = st.columns(5)
    long_len = c[0].selectbox("긴 요청 길이", [8192, 16384, 32768], index=2, key="live_long")
    n_short = c[1].selectbox("짧은 요청 수", [7, 15, 31], index=2, key="live_n")
    short_len = c[2].selectbox("짧은 요청 길이", [384, 512], index=1, key="live_short")
    new_tokens = c[3].selectbox("생성 토큰", [32, 64], index=1, key="live_tokens")
    live_policy = c[4].selectbox("비교 정책", ["table", "hybrid", "model"], key="live_policy")
    question = st.text_input("긴 문서에 던질 질문 (프롬프트 끝에 붙습니다)", value=demo.DEFAULT_QUESTION, key="live_question")
    live_dir = root / "serve_4090" / f"demo_live_{STAMP}"
    spec_map = {"table": f"table:{TABLE_CSV}", "hybrid": f"hybrid:{TABLE_CSV}:0.2", "model": "model"}
    argv = lab.compose_command("serve_run", python=PY, scenario=live_dir / "scenario.yaml", policies=["heuristic", spec_map[live_policy]],
                               out=live_dir / "ragged", kv_gib=10, warmup_runs=1, warmup_steps=2, repeats=1, seed=0,
                               policy_cache="keep", machine=str(MACHINE), params=str(PARAMS))

    def _prepare_live():
        corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
        spec = demo.compose_scenario(int(long_len), int(n_short), int(short_len), int(new_tokens), question, corpus, dataset)
        demo.write_scenario(live_dir / "scenario.yaml", spec)
        st.session_state["demo_live_dir"] = str(live_dir / "ragged")
        return live_dir / "serve.log"

    def _live_done():
        done = st.session_state.get("demo_live_dir")
        return bool(done and (Path(done) / "summary.csv").exists())

    if job_panel("demo_live", argv, "GPU 호스트에서 측정", _prepare_live, _live_done):
        done = st.session_state["demo_live_dir"]
        if done not in st.session_state.get("demo_live_runs", []):
            st.session_state.setdefault("demo_live_runs", []).append(done)
        st.success("측정이 끝났습니다. 위 '다른 기록 고르기'에서 이 폴더를 고르면 재생됩니다: " + done)

# scenes 2–4 are appended in Task 5
st.divider()
st.markdown('<div class="footnote">KERNELSCOPE · Graduation research demo<br>'
            '실측은 재현 가능한 기록으로, 예측은 검증할 가설로, 서비스 성능은 전체 실행으로 평가합니다. 자세한 탐색은 Lab 페이지.</div>',
            unsafe_allow_html=True)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py tests/test_dashboard_app.py -v 2>&1 | tail -25`
Expected: all PASS. If AppTest cannot find the radio by label, check `pick[1].radio(...)` is reached (requires `heuristic` + a candidate in the fixture).

- [ ] **Step 7: Visual smoke on the real data**

Run (leave it running in another shell or background for the reviewer):
```bash
KERNELSCOPE_RESULTS=../kernelscope/results PORT=8502 ./run.sh dashboard
```
Open http://localhost:8502. Expected: Demo page opens with three tiles, the race replays with readable English answers for the ragged preset (natural text), the two panes finish at different times, the success line reads "생성 토큰 일치 · 32/32 요청". Switch to Lab in the sidebar: the five tabs are intact. Stop the server.

---

### Task 5: Demo scenes 2–4 (why, how, verify) and the plugin bench

**Files:**
- Modify: `dashboard/demo_page.py` (insert scenes before the footer)
- Modify: `tests/test_dashboard_app.py`

**Interfaces:**
- Consumes: `demo.cta_work`, `demo.decision_card`, `demo.campaign_table`, `demo.verify_tail`, `demo.WORST_KEY`; `figures.workload_lengths`, `figures.variants_bar`, `figures.regret_heatmap`, `figures.cta_work_curves`, `figures.backend_bar`; `lab.backend_comparison`, `lab.divergence_overview`, `lab.compose_command("bench", workload=...)`; `kernelscope.analysis.dispatch.FAMILIES/dispatch_table`; `kernelscope.run_kernel.load_registry/DEFAULT_REGISTRY`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard_app.py`:
```python
def test_demo_page_scenes_two_to_four(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic", "table"))
    make_run(tmp_path, scenario="uniform", policies=("heuristic", "table"))
    directory = tmp_path / "hw_4090" / "worst"
    directory.mkdir(parents=True)
    worst = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
    rows = [dict(status="ok", plugin=p, workload_key=worst, cache_state="cold", kernel_time_us=t)
            for p, t in [("flashdecoding_paged", 1071.8), ("fd_s16_paged", 279.4), ("fa2_paged", 1071.2), ("flashinfer_paged_cudacore", 269.9)]]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    headers = [s.value for s in app.subheader]
    assert any(h.startswith("2 ·") for h in headers) and any(h.startswith("3 ·") for h in headers) and any(h.startswith("4 ·") for h in headers)
    assert any("가장 긴 CTA" in metric.label for metric in app.metric)
    assert any("3.84×" in item.value for item in app.caption)            # 1071.8 / 279.4 from the fixture rows
    assert any("대조군" in item.value for item in app.caption)
    assert len(app.dataframe) >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_app.py -k scenes 2>&1 | tail -5`
Expected: FAIL (no subheader starts with "2 ·").

- [ ] **Step 3: Insert scenes 2–4 into demo_page.py**

Insert the following **between** the live-measurement expander and the `# scenes 2–4 are appended in Task 5` comment (then delete that comment). Add `from kernelscope.analysis.dispatch import FAMILIES, dispatch_table` and `from kernelscope.workload import Workload` to the imports at the top.

```python
# ---------------------------------------------------------------- scene 2 · why
@st.cache_data(ttl=300, show_spinner=False)
def hw_index(dirs: tuple) -> pd.DataFrame:
    return data.load_index([Path(d) for d in dirs])


@st.cache_data(show_spinner="정책을 계산하는 중…")
def decision(lens: tuple, n_heads: int, n_kv_heads: int, measured_items: tuple) -> list:
    return demo.decision_card(list(lens), n_heads, n_kv_heads, TABLE_CSV, MACHINE, PARAMS, pd.Series(dict(measured_items)))


@st.cache_data(show_spinner=False)
def plugin_names(workload_key: str) -> list:
    try:
        from kernelscope.run_kernel import DEFAULT_REGISTRY, load_registry
        return sorted(p.name for p in load_registry(DEFAULT_REGISTRY).supporting(Workload.from_key(workload_key)))
    except Exception:  # plugins import torch / flash-attn; a laptop without them still shows the page
        return []


st.divider()
st.subheader("2 · 왜 느린가 — 긴 요청 하나가 블록 하나에 갇힌다")
w = Workload.from_key(demo.WORST_KEY)
n_sm = int(json.loads(MACHINE.read_text()).get("n_sm", 128)) if MACHINE.exists() else 128
index = hw_index(tuple(str(d) for d in data.find_dirs("hw", root)))
family = FAMILIES["paged"]
if not index.empty:
    states = index["cache_state"].fillna("warm") if "cache_state" in index else pd.Series("warm", index=index.index)
    cold = index[states == "cold"]
else:
    cold = index
cell = cold[(cold.workload_key == demo.WORST_KEY) & cold.kernel.str.match(family["members"])] if not cold.empty else cold
measured = cell.groupby("kernel").kernel_time_us.median().sort_values() if not cell.empty else pd.Series(dtype=float)
left, right = st.columns([1, 1.5])
with left:
    chart(figures.workload_lengths(w, theme), "demo_lengths")
    st.write(f"{w.B}개 요청 · 긴 요청 {w.L_kv:,} tokens 1개 + 짧은 요청 {min(w.lens()):,} tokens {w.B - 1}개 · 총 {sum(w.lens()):,} KV tokens")
with right:
    base = demo.cta_work(demo.WORST_KEY, "flashdecoding_paged", n_sm)
    best_kernel = str(measured.idxmin()) if not measured.empty else "fd_s16_paged"
    chosen = demo.cta_work(demo.WORST_KEY, best_kernel, n_sm)
    chart(figures.cta_work_curves([
        {"label": f"휴리스틱 · 분할 {base['splits']} · CTA {base['ctas']:,}개", "color": style.policy_color("heuristic", theme), "keys": base["keys"]},
        {"label": f"{best_kernel} · 분할 {chosen['splits']} · CTA {chosen['ctas']:,}개", "color": style.policy_color("table", theme), "keys": chosen["keys"]},
    ], theme), "demo_cta")
    m = st.columns(2)
    m[0].metric("가장 긴 CTA / 평균 CTA · 휴리스틱", f"{base['longest_over_mean']:.1f}×")
    m[1].metric("가장 긴 CTA / 평균 CTA · 선택", f"{chosen['longest_over_mean']:.1f}×")
    st.caption(f"계산 · 길이와 분할 수에서 구한 CTA 작업량(실측 아님). 휴리스틱은 CTA {w.B * w.H_kv}개 ≥ 0.8 × {2 * n_sm}이면 분할 {base['splits']}로 "
               f"조기 결정해, 긴 요청의 {w.L_kv:,} keys를 블록 하나가 처리하는 동안 {n_sm}개 SM 대부분이 빕니다.")
if measured.empty:
    st.info("최악 셀의 paged 커널 기록이 없습니다 (hw_4090/*/summaries.jsonl).")
else:
    chart(figures.variants_bar(pd.DataFrame(), measured, family["heuristic"], theme, "같은 입력, 다른 분할 수 · 커널 시간 실측 (paged KV, cold)"), "demo_variants")
    h = measured.get(family["heuristic"])
    if h is not None and pd.notna(h):
        st.caption(f"실측 · 라이브러리 휴리스틱 {h:,.0f} µs vs 가장 빠른 분할 {measured.idxmin()} {measured.min():,.0f} µs = {h / measured.min():.2f}× "
                   f"(attention 커널만). dense 레이아웃의 같은 셀은 Lab 01 Diagnose에서.")
diagnoses = sorted((root / "serve_4090").rglob("diagnosis.json")) if (root / "serve_4090").exists() else []
diagnoses = [p for p in diagnoses if p.parent.name == "ragged"] or diagnoses
if diagnoses:
    try:
        d = json.loads(diagnoses[-1].read_text())
        shares = {policy: next((r["share"] for r in p["ops"] if r["op_class"] == "attention"), None) for policy, p in d["policies"].items()}
    except (OSError, ValueError, KeyError, TypeError):
        shares = {}
    if shares:
        png = diagnoses[-1].with_name("op_breakdown.png")
        png = png if png.exists() else ROOT / "docs" / "img" / "op_breakdown_ragged.png"
        cols = st.columns([1.5, 1])
        if png.exists():
            cols[0].image(str(png), caption="실측 · decode step을 8개 연산 클래스로 분해 (CUDA event)")
        with cols[1]:
            for policy, share in shares.items():
                st.metric(f"attention 비중 · {policy}", "—" if share is None else f"{100 * share:.1f}%")
            st.caption("attention 외 연산은 이미 DRAM 상한 근처라 커널 선택의 여지가 작습니다. 분할 수를 바꾸면 attention 비중만 줄어듭니다.")

# ---------------------------------------------------------------- scene 3 · how
st.divider()
st.subheader("3 · 어떻게 고르나 — 실측표와 성능 모델이 같은 배치에 내리는 결정")
paged_cold = cold[cold.kernel.str.match(family["members"]) & cold.workload_key.str.contains(f"_Hq{w.H_q}_Hkv{w.H_kv}_d{w.d}_{w.dtype}_")] if not cold.empty else cold
table = dispatch_table(data.index_rows(paged_cold), "paged", "cold") if not paged_cold.empty else pd.DataFrame()
left, right = st.columns([1.3, 1])
with left:
    if table.empty:
        st.info("커널 지도를 그릴 paged cold 기록이 없습니다.")
    else:
        chart(figures.regret_heatmap(table, theme, "실측 · 라이브러리 휴리스틱의 선택 손실 (paged, cold)"), "demo_heatmap")
        st.caption("손실 = heuristic 시간 / 최적 실측 시간 − 1. 균일 길이 행은 거의 0, 혼합 길이 행에서 커집니다.")
with right:
    st.markdown("**이 배치에 대한 결정 카드**")
    for row in decision(tuple(w.lens()), w.H_q, w.H_kv, tuple(measured.items())):
        split = "—" if row["num_splits"] is None else f"분할 {row['num_splits']}"
        us = "" if row["kernel_us"] is None else f" · {row['kernel_us']:,.0f} µs"
        gain = "" if row["speedup_vs_heuristic"] is None else f" · 휴리스틱 대비 {row['speedup_vs_heuristic']:.2f}×"
        st.markdown(f"**{row['label']}** — {split}{' · ' + row['kernel'] if row['kernel'] else ''}{us}{gain}  \n"
                    f"<span class='footnote'>{row['note']}</span>", unsafe_allow_html=True)
    st.caption("분할 수와 출처는 정책 코드가 지금 계산한 값(계산), 커널 시간은 그 변형의 실측 중앙값(실측).")
backend = lab.backend_comparison(index, "cold") if not index.empty else pd.DataFrame()
backend_row = backend[backend.workload_key == demo.WORST_KEY] if not backend.empty else backend
if not backend_row.empty:
    r = backend_row.iloc[0].to_dict()
    chart(figures.backend_bar(r, theme), "demo_backends")
    fi = r.get("flashinfer_us")
    if fi is not None and pd.notna(fi) and pd.notna(r.get("best_fa2_us")):
        st.caption(f"실측 · FlashInfer(더 빠른 변형) / 최선 FA2 분할 = {fi / r['best_fa2_us']:.3f}배. 분할 수를 밖에서 고르는 것으로 커널 교체 이득의 대부분을 얻습니다.")
with st.expander("플러그인 벤치 · 등록된 커널을 이 셀에서 직접 재기 (약 30초)"):
    names = plugin_names(demo.WORST_KEY)
    if not names:
        st.caption("이 장비에서는 플러그인 레지스트리를 불러올 수 없습니다(torch/flash-attn 필요).")
    default = [n for n in ("flashdecoding_paged", "fd_s16_paged", "flashinfer_paged_cudacore") if n in names]
    chosen_plugins = st.multiselect("플러그인", names, default=default, key="bench_plugins")
    bench_dir = root / "hw_4090" / f"demo_bench_{STAMP}"
    bench_argv = lab.compose_command("bench", python=PY, workload=demo.WORST_KEY, plugins=chosen_plugins, results=bench_dir,
                                     cache_state="cold", warmup=5, iters=20)

    def _prepare_bench():
        st.session_state["demo_bench_dir"] = str(bench_dir)
        bench_dir.mkdir(parents=True, exist_ok=True)
        return bench_dir / "bench.log"

    def _bench_done():
        done = st.session_state.get("demo_bench_dir")
        return bool(done and (Path(done) / "summaries.jsonl").exists())

    if chosen_plugins and job_panel("demo_bench", bench_argv, "GPU 호스트에서 벤치", _prepare_bench, _bench_done):
        rows = [json.loads(line) for line in (Path(st.session_state["demo_bench_dir"]) / "summaries.jsonl").read_text().splitlines() if line.strip()]
        ok = pd.Series({row["plugin"]: float(row["kernel_time_us"]) for row in rows if row.get("status") == "ok"}).sort_values()
        if not ok.empty:
            chart(figures.variants_bar(pd.DataFrame(), ok, family["heuristic"], theme, "방금 측정 · 플러그인 벤치 (µs)"), "demo_bench_chart")
        for row in rows:
            if row.get("status") != "ok":
                st.caption(f"실패 · {row.get('plugin')}: {str(row.get('error', ''))[:160]}")

# ---------------------------------------------------------------- scene 4 · verify
st.divider()
st.subheader("4 · 검증 — 반복, 토큰 일치, 대조군, 문서 수치 재계산")
campaign = Path(run_dir).parent
table4 = demo.campaign_table(campaign)
if table4.empty:
    st.info("이 캠페인의 조건별 summary.csv가 없습니다.")
else:
    show = table4.assign(family=table4.family.map(lambda f: demo.FAMILY_LABELS.get(f, f)))
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"scenario": "조건", "family": "배치 종류", "policy": "정책",
                                "tpot_ms": st.column_config.NumberColumn("TPOT (ms)", format="%.2f"),
                                "speedup": st.column_config.NumberColumn("배율", format="%.3f×"),
                                "repeats": "반복", "tokens_equivalent": st.column_config.CheckboxColumn("토큰 일치")})
    uniform = table4[(table4.family == "uniform") & (table4.policy != "heuristic")].dropna(subset=["speedup"])
    if not uniform.empty:
        st.caption(f"대조군 · 균일 배치의 배율 {uniform.speedup.min():.3f}~{uniform.speedup.max():.3f}×. 개선 대상은 길이가 섞인 배치이며, 모든 입력에서 빨라진다고 주장하지 않습니다.")
    else:
        st.caption("대조군 · 이 캠페인에는 균일 배치 조건이 없습니다. Lab 04 Serving에서 graduation_20260922/uniform을 보세요.")
divergence = lab.divergence_overview([campaign])
if not divergence.empty:
    keep = [c for c in ("scenario", "policy", "repeats", "distinct_positions", "tie_1ulp", "tie_2ulp", "clear", "unclassified") if c in divergence]
    st.dataframe(divergence[keep], hide_index=True, width="stretch",
                 column_config={"scenario": "조건", "policy": "정책", "repeats": "반복", "distinct_positions": "분기 사건", "unclassified": "미분류"})
    st.caption("분기 사건 = 요청별 첫 불일치 위치의 수. tie_1ulp/2ulp는 기준 1위 로짓과 후보가 고른 토큰의 차이가 bf16 간격 1~2개인 동점, clear는 조사 대상.")
if st.button("문서 수치 재계산 · kernelscope verify (GPU 불필요)", key="demo_verify"):
    with st.spinner("커밋된 기록에서 다시 계산하는 중…"):
        result = subprocess.run([PY, "-m", "kernelscope.cli", "verify"], cwd=ROOT, capture_output=True, text=True)
    counts = demo.verify_tail(result.stdout)
    if counts.get("FAIL"):
        st.error(f"FAIL {counts['FAIL']}개 · PASS {counts.get('PASS', 0)}개")
    elif counts.get("PASS"):
        st.success(f"PASS {counts['PASS']}개 · 문서의 수치는 기록에서 다시 계산됩니다.")
    else:
        st.warning("verify 출력을 해석할 수 없습니다.")
    st.code((result.stdout or result.stderr)[-2500:], language="text")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_app.py tests/test_dashboard_demo.py -v 2>&1 | tail -25`
Expected: all PASS. The `3.84×` caption comes from `1071.8 / 279.4 = 3.836`; if the `.2f` formatting yields `3.84×` it matches.

- [ ] **Step 5: Full dashboard test group and visual check**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_app.py tests/test_dashboard_demo.py tests/test_dashboard_lab.py tests/test_dashboard_figures.py tests/test_dashboard_data.py tests/test_dashboard_research.py`
Expected: all PASS. Then `KERNELSCOPE_RESULTS=../kernelscope/results PORT=8502 ./run.sh dashboard` and scroll the Demo page: scene 2 shows the length bars, the two CTA curves, the kernel bar and the op-breakdown image; scene 3 the heatmap, four decision rows and the backend bar; scene 4 the campaign table (uniform row if the campaign has one), the divergence table and the verify button (click it: PASS count appears within ~30 s).

---

### Task 6: Export script, run.sh demo-record, documentation

**Files:**
- Create: `scripts/export_race.py`
- Modify: `run.sh`
- Modify: `docs/demo.md`, `README.md` (졸업 프로젝트 데모 절), `TODO.md` (§1 item 6), `PLAN.md` (§2 item 5), `docs/STATUS.md` (new top entry)
- Test: `tests/test_dashboard_demo.py` (one test for the script's policy discovery)

**Interfaces:**
- Consumes: `demo.race_payload`, `demo.save_race`, `demo.policy_dirs`, `kernelscope.serve.hf.load_tokenizer/snapshot_dir`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_dashboard_demo.py`:
```python
def test_export_race_script_without_text(tmp_path):
    import subprocess, sys
    run = make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_race.py"), "--run", str(run), "--no-text"],
                            capture_output=True, text=True, cwd=ROOT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stderr
    payload = demo.load_race(run)
    assert [p["name"] for p in payload["policies"]] == ["heuristic", "hybrid", "table"]
    assert "race.json" in result.stdout and "natural_text False" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py -k export 2>&1 | tail -4`
Expected: FAIL (script missing, returncode 2).

- [ ] **Step 3: Write the export script**

`scripts/export_race.py`:
```python
"""Write race.json next to a serving recording so the Demo page replays it without the tokenizer.

    python scripts/export_race.py --run ../kernelscope/results/serve_4090/demo_text_20261006/ragged
    python scripts/export_race.py --run <dir> --policies heuristic,table --no-text

Policies default to every policy folder with recorded steps (the reference `heuristic` first).
Text decoding needs the local model snapshot (HF_HUB_OFFLINE) and torch; `--no-text` skips it."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernelscope.dashboard import data, demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="scenario directory holding <policy>/repeat_*/tokens.parquet")
    parser.add_argument("--policies", help="comma list; default: every recorded policy")
    parser.add_argument("--reference", default="heuristic")
    parser.add_argument("--no-text", action="store_true", help="skip token decoding (no torch / model needed)")
    args = parser.parse_args()
    run = Path(args.run)
    policies = args.policies.split(",") if args.policies else demo.policy_dirs(run)
    if args.reference not in policies:
        raise SystemExit(f"{run} has no {args.reference!r} recording; policies: {policies}")
    decode = None
    if not args.no_text:
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        tokenizer = load_tokenizer(snapshot_dir(data.load_manifest(run).get("model")))
        decode = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True)
    payload = demo.race_payload(run, policies, reference=args.reference, decode=decode)
    path = demo.save_race(run, payload)
    print(f"{path} · policies {[p['name'] for p in payload['policies']]} · natural_text {payload['scenario']['natural_text']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_dashboard_demo.py -k export 2>&1 | tail -3`
Expected: PASS.

- [ ] **Step 5: Add demo-record to run.sh**

In `run.sh`:
1. In `usage()`, under "GPU 필요" after the `generate` line add:
```
    demo-record  대시보드 레이스용 자연어 혼합 길이 기록(demo_text_ragged: 32K×1 + 512×31, 64토큰): heuristic · table · hybrid, REPEATS=3  ~2분
```
   and change the "시연 순서 제안" line 2 to `2) ./run.sh dashboard (첫 화면 Demo: 결론 → 경주 → 왜 → 어떻게 → 검증; Q&A는 Lab 페이지의 01~05 탭)`.
2. Add after `cmd_generate()`:
```bash
cmd_demo_record() {
  mkdir -p "$OUT"; doctor
  local repeats="${REPEATS:-3}"
  say "데모 레이스 기록 — demo_text_ragged (자연어, 32K×1 + 512×31, 64토큰), heuristic · table · hybrid, ${repeats}회, 정책 캐시 유지"
  "$PY" -m kernelscope.cli serve run --model "$MODEL" --scenario scenarios/demo_text_ragged.yaml \
      --policy heuristic --policy "table:$TABLE" --policy "$HYBRID" "${MODEL_INPUTS[@]}" \
      --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --repeats "$repeats" --seed 0 --policy-cache keep \
      --out "$OUT/demo_text/ragged" > "$OUT/demo_record.log" 2>&1 || { tail -n 20 "$OUT/demo_record.log"; exit 1; }
  "$PY" scripts/export_race.py --run "$OUT/demo_text/ragged" --policies heuristic,table,hybrid
  note "대시보드 Demo의 '다른 기록 고르기'에서 $OUT/demo_text/ragged 를 고르면 글이 보이는 레이스가 재생됩니다."
  note "번들에 넣으려면: $PY scripts/package_demo.py --results $KERNELSCOPE_RESULTS"
}
```
3. Add `demo-record) cmd_demo_record ;;` to the `case`.

Run `bash -n run.sh && ./run.sh help | grep -n demo-record` — expected: syntax OK and the usage line prints.

- [ ] **Step 6: Rewrite docs/demo.md**

Replace the file's content above the "## 질문에 답할 때" section with:

```markdown
# 5분 발표 시나리오 — Demo 페이지 순서

실행: 프로젝트 루트에서 `./run.sh dashboard` → <http://localhost:8501>. 첫 화면이 **Demo**(고정 순서 다섯 장면), 사이드바의 **Lab**이 기존 다섯 탭(01 Diagnose · 02 Kernel map · 03 What-if · 04 Serving · 05 Policy lab)이다. 발표는 Demo를 위에서 아래로 넘기고, 질문이 나오면 Lab으로 내려간다.
발표 노트북에서는 연구실 4090 호스트의 Streamlit을 SSH 포트 포워딩으로 띄운다(`ssh -L 8501:127.0.0.1:8501 <host>` 뒤 호스트에서 `./run.sh dashboard`). 그러면 라이브 측정·플러그인 벤치 버튼이 동작하고, 연결이 끊겨도 기록 재생은 그대로 된다. 발표 전 점검 `./run.sh check`, 레이스용 자연어 기록이 없으면 `./run.sh demo-record`(약 2분), 마무리는 Demo 4장면의 verify 버튼 또는 `./run.sh verify`.

모든 수치는 화면이 기록에서 계산한다. 라벨 규칙: **실측**(GPU 기록), **재생**(실측 기록을 타임스탬프대로 재생), **계산**(길이·분할 수에서 구한 값), **예측**(성능 모델).

## 0:00–0:20 · 결론 (장면 0)

타일 세 개를 읽는다: 토큰 간격 TPOT 기본 → KernelScope, 배율, 생성 토큰 일치. "출력은 같고 토큰은 더 빨리 나온다"가 오늘의 주장이다.

## 0:20–1:50 · 경주 (장면 1)

배치 종류 **혼합 길이**, 정책 **table**(또는 hybrid)에서 ▶ 재생. 왼쪽 FlashAttention 휴리스틱, 오른쪽 KernelScope가 같은 32개 요청의 답을 쓰고, 오른쪽이 먼저 끝난다. 아래 띠에서 왼쪽은 분할 수가 "라이브러리"(실제로는 1), 오른쪽은 16으로 고정되어 attention 시간이 1/4로 짧은 것을 가리킨다. 하단 "생성 토큰 일치 32/32 요청"을 읽는다.
반드시 말할 것: **같은 GPU에서 순차로 측정한 기록을 동시에 재생**한 것이다(동시 실행은 측정을 오염시킨다). 시계 0은 모든 요청의 prefill이 끝난 시점이다. 배치 종류를 **균일**로 바꾸면 두 패널이 거의 같이 끝난다 — 이득은 혼합 길이에서만 난다.
시간이 있으면 "라이브 측정" 패널을 열어 명령을 보여 준다(모델 로드 포함 1~2분이라 발표 중 실행은 권하지 않는다; Q&A나 발표 직전에 쓴다).

## 1:50–2:50 · 왜 (장면 2)

배치 구성 막대(32K 하나 + 512 서른한 개) → CTA 작업량 곡선: 휴리스틱은 CTA 256개 ≥ 0.8×256이라 분할 1로 조기 결정하고, 그러면 블록 하나가 32K keys를 혼자 처리하는 동안 나머지 SM이 빈다(가장 긴 CTA / 평균 CTA 지표). 분할 16이면 CTA 4,096개로 고르게 퍼진다. 아래 실측 커널 막대에서 두 변형의 배율을 읽고, 연산 분해에서 attention 비중이 휴리스틱 → 테이블로 어떻게 줄어드는지, 나머지 연산은 이미 DRAM 상한 근처라 선택의 여지가 작다는 것을 말한다.

## 2:50–3:50 · 어떻게 (장면 3)

커널 지도에서 균일 길이 행은 손실이 거의 0이고 혼합 길이 행에서 커진다는 것을 보여 준다. 결정 카드에서 같은 배치에 heuristic / table / hybrid / model이 고른 분할 수와 그 변형의 실측 시간을 읽는다. "다른 커널은?" 막대로 FlashInfer와 비교해, 분할 수를 밖에서 고르는 것으로 커널 교체 이득의 대부분을 얻는다고 말한다. 질문이 나오면 플러그인 벤치로 최악 셀 하나를 30초 안에 다시 잰다.

## 3:50–4:30 · 검증 (장면 4)

조건별 표에서 반복 수·토큰 일치·균일 대조군(배율 ≈ 1)을 읽고, 분기 사건 표에서 불일치가 어떻게 분류되는지 말한다. verify 버튼으로 문서의 수치가 기록에서 다시 계산되는 것을 보여 준다.

## 4:30–5:00 · 기여와 한계

"커널 하나의 벤치마크에서 끝나지 않고, 입력 길이 분포에 따른 원인을 찾고, 실제 모델의 생성에 적용해 이득과 비용을 검증했습니다." 단일 RTX 4090, 통제된 자연어 구성 입력이며 운영 트래픽·여러 GPU·HTTP 부하는 평가하지 않았다.
```
Keep the existing "## 질문에 답할 때" section verbatim below it.

- [ ] **Step 7: README, TODO, PLAN, STATUS one-liners**

- `README.md`, 절 "## 졸업 프로젝트 데모": after the code block and its paragraph ("네 개의 화면에서 …"), replace that paragraph's first sentence with: `첫 화면 **Demo**는 결론 → 경주(두 정책의 기록을 타임스탬프대로 동시 재생) → 왜 → 어떻게 → 검증의 다섯 장면이고, 사이드바 **Lab**이 병목 진단, 조건별 최적 커널, 가상 하드웨어 예측, 실제 생성 실험, Policy lab의 다섯 탭입니다.` and add after it: `레이스용 자연어 기록은 `./run.sh demo-record`(GPU, 약 2분)로 만듭니다. 발표 순서는 [docs/demo.md](docs/demo.md).`
- `TODO.md` §1 item 6 becomes: `6. ~~Policy lab 탭 다듬기~~ → 2026-10-06 **Demo 페이지**(dashboard/demo_page.py, kernelscope/dashboard/{demo.py,race.html}; 설계 docs/plan/2026-10-06-design-demo-page.md) 완료: 결론·경주·왜·어떻게·검증 다섯 장면, 라이브 측정·플러그인 벤치 패널, `run.sh demo-record`, 자연어 기록 `demo_text_20261006`. 남은 것: 실행 버튼의 백그라운드 작업 세션 밖 생존, 결과 폴더 자동 새로고침.`
- `PLAN.md` §2 item 5: append ` 2026-10-06 Demo 페이지 추가(기존 5탭은 Lab 페이지): 기록 재생 토큰 레이스 + 라이브 측정 패널, [설계](docs/plan/2026-10-06-design-demo-page.md).`
- `docs/STATUS.md`: add a new top entry (same style as the existing ones) titled `## 2026-10-06 · Demo 페이지와 토큰 레이스` with three bullets: what was added (files), the recording `demo_text_20261006/ragged` (heuristic / table / hybrid, 3 repeats, natural text, tokens equivalent — numbers from its summary.csv, quoted to 2 decimals), and the rule that race replays are sequential measurements on one clock.

- [ ] **Step 8: Full suite and verify**

Run:
```bash
CUDA_VISIBLE_DEVICES= make test 2>&1 | tail -3
make verify 2>&1 | tail -1
bash -n run.sh && ./run.sh help >/dev/null && echo run.sh ok
```
Expected: all tests pass (count ≥ 563 + the new ones), `{"PASS": 111}`, `run.sh ok`.
