"""Demo page data: featured recordings, headline numbers, the token-race payload and the
per-batch decision card. Pure functions over recorded files; no Streamlit, no torch at import
(see docs/plan/2026-10-06-design-demo-page.md §4, §6)."""
import json
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
    """One scenario directory per family: natural-text prompts first, then the ragged pick's model, then newest."""
    entries = {}
    for directory in data.find_dirs("serve", root):
        if not (directory / "summary.csv").exists():
            continue
        manifest = data.load_manifest(directory)
        family = _family(manifest, directory)
        if family is None:
            continue
        entries.setdefault(family, []).append((manifest.get("prompt_kind", SYNTHETIC) != SYNTHETIC,
                                               str(manifest.get("created_at", "")), manifest.get("model"), directory))

    def pick(family, model=None):
        return max(entries[family], key=lambda e: (e[0], model is not None and e[2] == model, e[1], str(e[3])))

    out, model = {}, None
    if "ragged" in entries:
        chosen = pick("ragged")
        out["ragged"], model = chosen[3], chosen[2]
    for family in FAMILY_LABELS:
        if family in entries and family not in out:
            out[family] = pick(family, model)[3]
    return out


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


def featured_campaign(root) -> Path | None:
    """The campaign directory (parent of scenario dirs) covering the most families; ties → newest created_at."""
    scores = {}
    for directory in data.find_dirs("serve", root):
        if not (directory / "summary.csv").exists():
            continue
        manifest = data.load_manifest(directory)
        family = _family(manifest, directory)
        if family is None:
            continue
        families, created = scores.get(directory.parent, (set(), ""))
        scores[directory.parent] = (families | {family}, max(created, str(manifest.get("created_at", ""))))
    if not scores:
        return None
    return max(scores, key=lambda c: (len(scores[c][0]), scores[c][1], str(c)))


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
                      "attn_ms": float(r.attn_us) / 1000.0 if pd.notna(r.attn_us) else 0.0} for r in steps.itertuples()] if not steps.empty else []
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
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def load_race(run_dir) -> dict | None:
    path = Path(run_dir) / "race.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and "policies" in value else None


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
                ctas = len(lens) * n_kv_heads
                threshold = 0.8 * 2 * n_sm
                row.update(num_splits=resolved, kernel="flashdecoding_paged", kernel_us=heuristic_us,
                           note=f"라이브러리 규칙: CTA {ctas}개 {'≥' if ctas >= threshold else '<'} 0.8 × {2 * n_sm} → 분할 {resolved}")
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
