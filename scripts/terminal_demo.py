#!/usr/bin/env python3
"""KernelScope 터미널 시연 — 명령 하나로 분할 수 선택과 생성 속도 비교를 터미널에 보여 준다.

    ./demo.sh                  # 64K 혼합 길이(포스터의 2.45배), 기록 재생 — GPU 불필요
    ./demo.sh --scenario 32k   # 32K 혼합 길이(1.81배)
    ./demo.sh --full           # 시뮬레이터의 시간별 활성 SM 수도 출력
    ./demo.sh --step           # 장면마다 Enter 로 넘기기(발표용)
    ./demo.sh --live           # 연구실 RTX 4090: 지금 측정한 뒤 그 기록을 재생 (모델 로드 포함 1~3분)
    ./demo.sh --run <기록 폴더>  # `serve run` 결과 폴더를 그대로 재생

기본은 두 장면(약 40줄, 15초 안팎): 분할 수 선택 → 생성 속도 비교, 끝에 TPOT 한 줄.
분할 수 선택은 이 장비의 CPU에서 지금 실행해 계산값을 그대로 찍는다(휴리스틱 이식본, CTA 시뮬레이터, 조회 테이블, 혼합 정책).
속도 비교는 같은 GPU에서 정책별로 순차 측정한 토큰 시각을 한 시계로 재생한 것이다(동시 실행은 측정을 오염시킨다).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

BUNDLE = ROOT / "demo_data"
SERVE = BUNDLE / "serve_4090"
TABLE = "demo_data/dispatch_paged_cold.csv"
MACHINE, PARAMS = "machines/rtx4090.json", "models/rtx4090.json"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
# 재생할 기록(자연어 문맥이라 생성 글이 보인다)과, --live 가 같은 조건을 다시 잴 때 쓰는 시나리오·KV 예산.
SCENARIOS = {
    "64k": {"title": "64K 혼합 길이", "run": SERVE / "longctx_text_20261024/demo_text_ragged_64k",
            "scenario": "scenarios/demo_text_ragged_64k.yaml", "kv_gib": "13"},
    "32k": {"title": "32K 혼합 길이", "run": SERVE / "demo_text_20261006/ragged",
            "scenario": "scenarios/demo_text_ragged.yaml", "kv_gib": "10"},
}
POLICY_SPECS = {"table": f"table:{TABLE}", "hybrid": f"hybrid:{TABLE}:0.2", "model": "model"}
POLICY_NAMES = {"table": "테이블 정책", "hybrid": "혼합 정책", "model": "시뮬레이터 정책"}

# ---------------------------------------------------------------- 터미널 출력

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
# 색은 쓰지 않는다(터미널 기본 글자색). 굵게만 남긴다 — 이름은 호출부가 무엇을 강조하는지 보이도록 둔다.
ORANGE = TEAL = GREEN = GRAY = ""
BOLD = "1"


def paint(text, *codes):
    codes = [c for c in codes if c]
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m" if COLOR and codes else str(text)


def width(text) -> int:
    """화면에서 차지하는 칸 수(한글·전각은 2칸, 색 코드는 0칸)."""
    return sum(0 if unicodedata.combining(c) else 2 if unicodedata.east_asian_width(c) in "WF" else 1
               for c in ANSI.sub("", str(text)))


def pad(text, n, align="<"):
    gap = max(0, n - width(text))
    return " " * gap + str(text) if align == ">" else str(text) + " " * gap


def wrap(text, n) -> list[str]:
    """표시 폭 기준 줄바꿈(단어 경계 우선). 생성 글이 영어·한글 어느 쪽이어도 칸이 맞는다."""
    lines, line = [], ""
    for word in re.findall(r"\S+\s*", text):
        while width(word.rstrip()) > n:                       # 한 줄보다 긴 덩어리는 글자 단위로 자른다
            room = n - width(line)
            cut, used = 0, 0
            for ch in word:
                step = width(ch)
                if used + step > room:
                    break
                cut, used = cut + 1, used + step
            lines.append(line + word[:cut])
            line, word = "", word[cut:]
        if width(line) + width(word.rstrip()) > n:
            lines.append(line.rstrip())
            line = ""
        line += word
    if line.strip():
        lines.append(line.rstrip())
    return lines


def clip(text, n) -> str:
    """표시 폭 n 칸을 넘는 부분을 자른다(색 코드는 보존). 애니메이션 줄이 터미널 폭을 넘지 않게 한다."""
    if width(text) <= n:
        return str(text)
    result, used = "", 0
    for piece in re.split(r"(\x1b\[[0-9;?]*[A-Za-z])", str(text)):
        if ANSI.fullmatch(piece):
            result += piece
            continue
        for ch in piece:
            if used + width(ch) > n - 1:
                return result + "…" + ("\x1b[0m" if COLOR else "")
            result, used = result + ch, used + width(ch)
    return result


def note(text, indent=3):
    """회색 설명 줄. 터미널 폭에 맞춰 접는다."""
    cols = shutil.get_terminal_size((100, 40)).columns
    for line in wrap(text, max(40, cols - indent - 1)):
        out(" " * indent + paint(line, GRAY))


def bar(fraction, n, full="█", empty="░") -> str:
    k = max(0, min(n, round(fraction * n)))
    return full * k + empty * (n - k)


def out(line=""):
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


class Stage:
    """장면 제목. --step 이면 다음 장면 전에 Enter 를 기다린다."""

    def __init__(self, total, step):
        self.total, self.step, self.index = total, step and sys.stdin.isatty(), 0

    def __call__(self, title, note=""):
        if self.index and self.step:
            sys.stdout.write(paint("   Enter ▸ 다음 장면 ", GRAY))
            sys.stdout.flush()
            sys.stdin.readline()
            sys.stdout.write("\x1b[1A\x1b[2K" if COLOR else "")
        self.index += 1
        out()
        head = f"[{self.index}/{self.total}] {title}"
        fits = width(head) + width(note) + 2 < shutil.get_terminal_size((100, 40)).columns
        out(paint(head, BOLD) + (paint(f"  {note}", GRAY) if note and fits else ""))
        if note and not fits:
            out(paint(f"   {note}", GRAY))


# ---------------------------------------------------------------- 기록 읽기

def local_decoder(model):
    """이 장비의 HF 캐시에 토크나이저가 있으면 토큰 → 글 디코더. 없으면 None(글 없이 토큰 수만 보여 준다)."""
    try:
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        tokenizer = load_tokenizer(snapshot_dir(model))
        return lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True)
    except Exception:
        return None


def load_payload(run: Path, policy: str) -> dict:
    """기록 폴더 → 레이스 payload. 사전 디코드된 race.json 을 먼저 쓰고, 없으면 parquet 에서 만든다."""
    from kernelscope.dashboard import data, demo
    payload = demo.load_race(run)
    names = [p["name"] for p in payload["policies"]] if payload else []
    if not payload or policy not in names:
        recorded = demo.policy_dirs(run)
        if "heuristic" not in recorded or policy not in recorded:
            raise SystemExit(f"{run}: heuristic 과 {policy} 기록이 모두 있어야 합니다 (있는 정책: {recorded})")
        payload = demo.race_payload(run, ["heuristic", policy], decode=local_decoder(data.load_manifest(run).get("model")))
    return payload


def first_step(run: Path, policy: str, repeat_index: int) -> dict:
    """재생하는 반복의 첫 디코드 단계: 요청별 KV 길이와 그때 엔진이 실제로 넘긴 분할 수."""
    import pandas as pd
    from kernelscope.dashboard import demo
    runs = demo._runs(run / policy)
    steps = pd.read_parquet(runs[min(repeat_index, len(runs) - 1)] / "steps.parquet")
    row = steps.sort_values("step").iloc[0]
    lens = json.loads(row.lens) if isinstance(row.lens, str) else list(row.lens)     # 기록은 JSON 문자열로 저장한다
    return {"lens": [int(x) for x in lens], "num_splits": int(row.num_splits)}


def head_shape(run: Path) -> tuple[int, int]:
    config = json.loads((run / "manifest.json").read_text()).get("model_config") or {}
    return int(config.get("n_heads", 32)), int(config.get("n_kv_heads", 8))


# ---------------------------------------------------------------- 장면 1: 분할 수 결정

def workload(lens, hq, hkv, dtype="bfloat16"):
    from kernelscope.workload import Workload
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=hq, H_kv=hkv, d=128, dtype=dtype,
                    kv_lens=tuple(lens))


def show_decision(lens, hq, hkv, policy):
    """분할 수 결정을 계산한 값 그대로 찍는다(색 없음): 라이브러리 휴리스틱, CTA 시뮬레이터의 후보별 예측, 선택 정책.

    전부 이 장비의 CPU에서 지금 실행한다. 후보 9개를 모두 평가하고 표에는 대표 분할 수만 보인다.
    반환: (휴리스틱 분할 수, 선택한 분할 수, machine, params)."""
    from kernelscope.model.geometry import parse_variant, resolve_splits
    from kernelscope.model.hybrid import hybrid_pick, variant_for_splits
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.model.predict import PAGED_VARIANTS, predict
    from kernelscope.model.simulate import prepare_simulator
    from kernelscope.serve.dispatch import TablePolicy
    from kernelscope.workload import format_lens
    machine, params = MachineSpec.from_json(MACHINE), ModelParams.from_json(PARAMS)
    w = workload(lens, hq, hkv)
    out(f"batch     B={len(lens)} kv_len={format_lens(lens)} Hq={hq} Hkv={hkv}")
    ctas, slots = len(lens) * hkv * math.ceil((hq // hkv) / 64), 2 * machine.n_sm
    base = resolve_splits(parse_variant("flashdecoding_paged"), w, machine.n_sm)
    out(f"fa2       num_splits_heuristic: CTA {ctas} {'>=' if ctas >= 0.8 * slots else '<'} {0.8 * slots:g} "
        f"-> num_splits={base}")

    backend = prepare_simulator()["backend"]
    predict(PAGED_VARIANTS[0], workload([512] * 4, hq, hkv), machine, params, "cold")      # 코드 경로만 데운다
    samples = []
    for _ in range(3):                                        # 후보 9개 평가를 세 번 재고 중앙값을 적는다
        start = time.perf_counter()
        rows = [predict(c, w, machine, params, "cold") for c in PAGED_VARIANTS]
        samples.append((time.perf_counter() - start) * 1e3)
    best = min(rows, key=lambda r: r.time_us)

    predicted, picks, reason = {r.plugin: r.time_us for r in rows}, {"model": best.plugin}, {"model": "sim best"}
    try:
        table = TablePolicy(TABLE)
        variant = variant_for_splits(int(table.choose(lens, hq, hkv)))
        near = re.search(r"_Lkv([0-9x+]+)_", str(table.last["workload_key"]))
        hybrid = hybrid_pick(predicted, variant, 0.2)
        picks.update(table=variant, hybrid=hybrid)
        reason.update(table=f"table: 실측 이웃 Lkv{near.group(1) if near else '?'}",
                      hybrid="hybrid: table" if hybrid == variant else "hybrid: sim best")
    except Exception as exc:                                  # 표가 이 헤드 구성을 모르는 등: 시뮬레이터 선택으로 계속한다
        out(f"table     unavailable: {type(exc).__name__}: {exc}")
    chosen = policy if policy in picks else "model"
    ours = resolve_splits(parse_variant(picks[chosen]), w, machine.n_sm)

    out()
    out(f"cta_sim   machine={Path(MACHINE).stem} n_sm={machine.n_sm} backend={backend}")
    out(f"  {'splits':>6}{'main_us':>10}{'comb_us':>10}{'pred_us':>10}{'bw_bd':>7}")
    by_splits = {r.splits: r for r in rows if r.plugin != "flashdecoding_paged"}
    for splits in sorted({1, 4, 8, 16, 128, best.splits, base, ours} & set(by_splits)):
        r = by_splits[splits]
        tags = [t for t, hit in (("fa2 기본", splits == base), ("sim best", r is best and chosen != "model"),
                                 ("선택", splits == ours)) if hit]
        out(f"  {r.splits:>6}{r.main_us:>10.1f}{r.combine_us:>10.1f}{r.time_us:>10.1f}{r.bw_bound_fraction:>7.2f}"
            + (f"   <- {', '.join(tags)}" if tags else ""))
    out(f"  후보 {len(rows)}개 평가 {sorted(samples)[1]:.1f} ms")
    out("  " + pad("main_us  attention 커널 시간", 32) + "comb_us  분할 결과를 합치는 시간")
    out("  " + pad("pred_us  main_us + comb_us", 32) + "bw_bd    메모리 대역폭에 막힌 시간 비율")
    out()
    out(f"dispatch  num_splits={ours} ({reason[chosen]})")
    return base, ours, machine, params


def cta_timeline(lens, hq, hkv, variant, machine, params, cache="cold"):
    """kernelscope/model/simulate.py 의 기준 이벤트 루프에 CTA 별 SM·시작·종료 시각(µs) 기록만 더한 것."""
    import numpy as np
    from kernelscope.analytic import DTYPE_BYTES
    from kernelscope.backends.realhw.kprofile import blocks_per_sm_limit
    from kernelscope.model.geometry import KIND_RESOURCES, build_launch, parse_variant
    from kernelscope.model.predict import bandwidth_bytes_per_us
    w = workload(lens, hq, hkv)
    launch = build_launch(w, parse_variant(variant), machine.n_sm)
    kp = params.states[cache].kinds[launch.kind]
    slots, _ = blocks_per_sm_limit(*KIND_RESOURCES[launch.kind], machine.props())
    bpk = 2 * w.d * DTYPE_BYTES[w.dtype]
    bw = bandwidth_bytes_per_us(machine, cache, float(launch.keys.sum()) * bpk)
    keys = launch.keys.astype(float)
    n, n_slots = len(keys), machine.n_sm * slots
    slot_sm = np.asarray(machine.sm_order())[np.arange(n_slots) % machine.n_sm]
    occupied, fixed_left, keys_left = np.zeros(n_slots, bool), np.zeros(n_slots), np.zeros(n_slots)
    cta = np.full(n_slots, -1)
    sm, start, end = np.zeros(n, int), np.zeros(n), np.zeros(n)
    nxt, t = 0, 0.0

    def admit(free):
        nonlocal nxt
        m = min(len(free), n - nxt)
        if m <= 0:
            return
        free, ids = free[:m], np.arange(nxt, nxt + m)
        nxt += m
        occupied[free], keys_left[free], cta[free] = True, keys[ids], ids
        fixed_left[free] = np.where(keys[ids] > 0, kp.t0_us, kp.t_empty_us)
        sm[ids], start[ids] = slot_sm[free], t

    admit(np.arange(n_slots))
    while occupied.any():
        per_sm = np.bincount(slot_sm[occupied], minlength=machine.n_sm)
        gamma = np.where(per_sm[slot_sm] >= 2, kp.gamma, 1.0)
        fixed = occupied & (fixed_left > 1e-9)
        streaming = occupied & ~fixed & (keys_left > 1e-6)
        rate = np.where(streaming, gamma / kp.cost_us_per_key, 0.0)
        demand = rate.sum() * bpk
        rate *= 1.0 if demand <= bw else bw / demand
        if (occupied & ~fixed & ~streaming).any():
            dt = 0.0
        else:
            dt = min(np.min(fixed_left[fixed]) if fixed.any() else np.inf,
                     np.min(keys_left[streaming] / rate[streaming]) if streaming.any() else np.inf)
        t += dt
        fixed_left[fixed] -= dt
        keys_left[streaming] -= rate[streaming] * dt
        finished = occupied & (fixed_left <= 1e-9) & (keys_left <= 1e-6)
        end[cta[finished]] = t
        occupied[finished] = False
        admit(np.flatnonzero(finished))
    return sm, start, end


def show_sm_activity(lens, hq, hkv, base, ours, machine, params, cols):
    """--full: 같은 시간 축에 분할 수 두 개의 활성 SM 수(포스터의 '시간에 따라 일하는 SM 수')."""
    import numpy as np
    from kernelscope.model.hybrid import variant_for_splits
    lines = {s: cta_timeline(lens, hq, hkv, variant_for_splits(s), machine, params) for s in dict.fromkeys((base, ours))}
    span = max(float(end.max()) for _, _, end in lines.values())
    room = max(20, min(64, cols - 45))
    grid = (np.arange(room) + 0.5) * span / room
    out()
    out(f"cta_sim   active SMs over time (of {machine.n_sm}), 0..{span:.0f} us")
    for splits, (sm, start, end) in lines.items():
        active = [len(set(sm[(start <= x) & (end > x)])) for x in grid]
        cells = "".join(" " if a == 0 else " ▁▂▃▄▅▆▇█"[max(1, math.ceil(8 * a / machine.n_sm))] for a in active)
        out(f"  splits={splits:<4}|{cells}| end_us={end.max():.0f} min_active={min(a for a in active if a)}")


# ---------------------------------------------------------------- 장면 2: 생성 속도 비교

class Panel:
    TEXT_ROWS = (4, 3)        # 긴 요청, 짧은 요청의 글 줄 수

    def __init__(self, panel, title, tone, splits_text):
        import numpy as np
        self.title, self.tone, self.splits_text = title, tone, splits_text
        self.requests = panel["requests"]
        self.times = np.sort(np.fromiter((t["t_ms"] for r in self.requests for t in r["tokens"]), float))
        self.end_ms = float(panel["end_ms"])
        self.tpot = panel.get("tpot_ms")
        attn = [s["attn_ms"] for s in panel.get("steps", []) if s.get("attn_ms")]
        self.attn_ms = sum(attn) / len(attn) if attn else None
        featured = sorted((r for r in self.requests if r.get("featured")), key=lambda r: -r["prompt_len"])
        self.featured = featured[:2] or self.requests[:1]
        self.has_text = any(t["text"] for r in self.featured for t in r["tokens"])

    def render(self, t_ms, w) -> list[str]:
        import numpy as np
        inner = w - 4
        made, total = int(np.searchsorted(self.times, t_ms, "right")), len(self.times)
        done = made >= total
        clock = min(t_ms, self.end_ms)
        rows = [paint(self.splits_text, self.tone, BOLD),
                paint(f"attention {self.attn_ms:.1f} ms/단계" if self.attn_ms else "", GRAY), ""]
        for request, n_rows in zip(self.featured, self.TEXT_ROWS):
            tokens = [t for t in request["tokens"] if t["t_ms"] <= t_ms]
            rows.append(paint(f"요청 {request['rid']} · 문맥 {request['prompt_len']:,} 토큰 · "
                              f"{len(tokens)}/{len(request['tokens'])}", GRAY))
            if self.has_text:
                text = re.sub(r"\s*\n+\s*", " ↵ ", "".join(t["text"] for t in tokens)).strip()
                lines = wrap(text, inner - 1)[-n_rows:]                      # 커서 한 칸을 남긴다
                if lines and not done:
                    lines[-1] += paint("▌", self.tone)
            else:
                lines = [paint(bar(len(tokens) / max(1, len(request["tokens"])), inner, "▮", "▯"), self.tone)]
            rows += lines + [""] * (n_rows - len(lines))
            rows.append("")
        rows.append(f"{paint(bar(made / total, max(8, inner - 18)), self.tone)} {made:>5,}/{total:,} 토큰")
        state = paint(f"완료 {self.end_ms / 1e3:.2f} s", GREEN, BOLD) if done else paint("생성 중…", self.tone)
        rate = f" · {made / clock * 1e3:,.0f} 토큰/s" if clock >= 100 else ""        # 시계 0 직후의 값은 의미가 없다
        rows.append(f"경과 {clock / 1e3:5.2f} s{rate}")
        rows.append(state)
        title = clip(self.title, w - 6)
        top = f"┌─ {title} " + "─" * max(0, w - 5 - width(title)) + "┐"
        body = [paint("│ ", self.tone) + pad(clip(row, inner), inner) + paint(" │", self.tone) for row in rows]
        return [paint(top, self.tone, BOLD)] + body + [paint("└" + "─" * (w - 2) + "┘", self.tone)]


def run_race(payload, policy, base_splits, speed, animate, cols):
    by_name = {p["name"]: p for p in payload["policies"]}
    ref, new = by_name["heuristic"], by_name[policy]
    picks = sorted({s["num_splits"] for s in new.get("steps", [])})
    pick_text = f"분할 {picks[0]}" if len(picks) == 1 else f"분할 {picks[0]}~{picks[-1]} (단계마다 선택)" if picks else "분할 수 선택"
    panels = [Panel(ref, "FlashAttention-2 기본 휴리스틱", ORANGE,
                    f"분할 {base_splits} ({'나누지 않음' if base_splits == 1 else '라이브러리 결정'})"),
              Panel(new, f"KernelScope ({POLICY_NAMES.get(policy, policy)})", TEAL, pick_text)]
    side = cols >= 80
    w = (min(cols, 124) - 3) // 2 if side else min(cols, 78) - 2
    end = max(p.end_ms for p in panels)
    rate = "실제 속도" if speed >= 0.999 else f"{speed:.2f}배속"
    note(f"정책별로 순차 측정한 토큰 시각을 한 시계로 재생 ({rate})")

    def frame(t):
        a, b = (p.render(t, w) for p in panels)
        return [" " + x + " " + y for x, y in zip(a, b)] if side else [" " + x for x in a + b]

    height = len(frame(0.0))
    animate = animate and sys.stdout.isatty() and shutil.get_terminal_size().lines > height + 2
    if animate:
        sys.stdout.write("\x1b[?25l")
        try:
            start, first = time.perf_counter(), True
            while True:
                t = (time.perf_counter() - start) * 1000.0 * speed
                lines = frame(t)
                sys.stdout.write(("" if first else f"\x1b[{height}A") + "".join(line + "\x1b[K\n" for line in lines))
                sys.stdout.flush()
                first = False
                if t >= end:
                    break
                time.sleep(1 / 30)
        finally:
            sys.stdout.write("\x1b[?25h")
            sys.stdout.flush()
    else:
        for line in frame(math.inf):
            out(line)

    agreement = payload.get("agreement", {}).get(policy, {})
    equal, total = agreement.get("requests_equal"), agreement.get("requests_total")
    line = ""
    if ref.get("tpot_ms") and new.get("tpot_ms"):
        line = (paint(f"TPOT  {ref['tpot_ms']:.2f} ms → {new['tpot_ms']:.2f} ms", BOLD) + "  "
                + paint(f"({ref['tpot_ms'] / new['tpot_ms']:.2f}배)", TEAL, BOLD))
    if total:
        line += ("   " if line else "") + f"생성 토큰 동일 {equal}/{total} 요청"
    if line:
        out()
        out("   " + line)


# ---------------------------------------------------------------- --live: 지금 측정

def results_root() -> Path:
    if os.environ.get("KERNELSCOPE_RESULTS"):
        return Path(os.environ["KERNELSCOPE_RESULTS"])
    shared = ROOT.parent / "kernelscope" / "results"
    return shared if shared.is_dir() else ROOT / "results"


def spin(process, label):
    start, i = time.perf_counter(), 0
    while process.poll() is None:
        if sys.stdout.isatty():
            sys.stdout.write(f"\r   {'⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[i % 10]} {label} … {time.perf_counter() - start:4.0f} s\x1b[K")
            sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    if sys.stdout.isatty():
        sys.stdout.write("\r\x1b[K")
    return process.returncode, time.perf_counter() - start


def live_record(scenario, policy) -> Path:
    """연구실 GPU에서 기준·선택 정책을 한 번씩 측정한다(run.sh demo-record 와 같은 규약, 반복 1회)."""
    out_dir = results_root() / "demo_runs" / time.strftime("%Y%m%d_%H%M%S") / "terminal_demo"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    python = [sys.executable, "-m", "kernelscope.cli", "serve"]
    doctor = subprocess.run(python + ["doctor", "--model", MODEL], capture_output=True, text=True, env=env)
    if doctor.returncode != 0:
        try:
            errors = json.loads(doctor.stdout).get("errors", [])
        except ValueError:
            errors = [(doctor.stdout + doctor.stderr).strip()[-600:]]
        out(paint("   라이브 측정을 시작할 수 없습니다:", ORANGE, BOLD))
        for error in errors:
            out(f"    - {error}")
        raise SystemExit("   GPU·모델 상태를 확인한 뒤 다시 실행하거나, --live 없이 기록 재생으로 시연하세요.")
    out_dir.mkdir(parents=True, exist_ok=True)
    command = python + ["run", "--model", MODEL, "--scenario", scenario["scenario"], "--policy", "heuristic",
                        "--policy", POLICY_SPECS[policy], "--machine", MACHINE, "--params", PARAMS,
                        "--kv-gib", scenario["kv_gib"], "--warmup-runs", "1", "--warmup-steps", "2", "--repeats", "1",
                        "--seed", "0", "--policy-cache", "keep", "--out", str(out_dir / "run")]
    with open(out_dir / "serve.log", "w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        code, seconds = spin(process, f"RTX 4090에서 측정 중 (모델 로드 → 프리필 → 정책별 생성, {scenario['title']})")
    if not (out_dir / "run" / "summary.csv").exists():
        out((out_dir / "serve.log").read_text()[-1500:])
        raise SystemExit(f"   측정 실패 (종료 코드 {code}). 로그: {out_dir / 'serve.log'}")
    out(paint(f"   측정 완료 {seconds:.0f} s → {out_dir / 'run'}", GREEN)
        + (paint("  (토큰 불일치가 기록됨 — 아래 비교의 일치 표시 참고)", ORANGE) if code else ""))
    return out_dir / "run"


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="64k", help="재생·측정할 혼합 길이 조건 (기본 64k)")
    parser.add_argument("--policy", choices=sorted(POLICY_SPECS), default="table", help="기본 휴리스틱과 겨룰 정책 (기본 table)")
    parser.add_argument("--speed", type=float, default=0.55, help="재생 속도 배수 (1 = 실제 속도; 기본 0.55)")
    parser.add_argument("--full", action="store_true", help="시뮬레이터의 시간별 활성 SM 수도 출력")
    parser.add_argument("--step", action="store_true", help="장면마다 Enter 로 넘긴다")
    parser.add_argument("--no-anim", action="store_true", help="애니메이션 없이 마지막 화면만 출력")
    parser.add_argument("--live", action="store_true", help="GPU에서 지금 측정한 뒤 그 기록을 재생 (연구실 RTX 4090)")
    parser.add_argument("--run", help="재생할 `serve run` 결과 폴더 (heuristic 과 --policy 의 기록이 있어야 한다)")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed 는 양수여야 합니다")
    cols = shutil.get_terminal_size((100, 40)).columns
    scenario = SCENARIOS[args.scenario]
    stage = Stage(2, args.step)

    if args.live:
        out("kernelscope demo  source=live (이 장비의 GPU에서 지금 측정)")
        run = live_record(scenario, args.policy)
    else:
        run = Path(args.run) if args.run else scenario["run"]
        if not run.is_dir():
            raise SystemExit(f"기록 폴더가 없습니다: {run}")
    payload = load_payload(run, args.policy)
    manifest = json.loads((run / "manifest.json").read_text()) if (run / "manifest.json").exists() else {}
    by_name = {p["name"]: p for p in payload["policies"]}
    step0 = first_step(run, args.policy, by_name[args.policy].get("repeat_index", 0))
    hq, hkv = head_shape(run)
    lens, prompt_lens = step0["lens"], payload["scenario"]["lens"]
    out(f"kernelscope demo  source={'live' if args.live else 'replay'} recorded={str(manifest.get('created_at', ''))[:10]}")
    out(f"model={str(manifest.get('model', MODEL)).split('/')[-1]} requests={len(prompt_lens)}")
    out(f"run={run.relative_to(ROOT) if run.is_relative_to(ROOT) else run}")

    stage("분할 수 선택")
    base, ours, machine, params = show_decision(lens, hq, hkv, args.policy)
    if args.full:
        show_sm_activity(lens, hq, hkv, base, ours, machine, params, cols)

    stage("실제 생성 속도 비교", f"{scenario['title'] if not args.run else run.name} · {manifest.get('model', MODEL).split('/')[-1]}")
    run_race(payload, args.policy, base, args.speed, not args.no_anim, cols)

    out()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stdout.write("\x1b[?25h\n")
        sys.exit(130)
