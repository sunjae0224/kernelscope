# `serve diagnose` (연산 클래스 분해 + 상한 판정) 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** decode step을 8개 연산 클래스로 CUDA event 실측 분해하고, 클래스별 최소 바이트·FLOPs를 측정된 상한에 대어 판정하는 `serve diagnose` 명령과 GPU 없이 재계산되는 리포트를 만든다.

**Architecture:** `serve/model.py`의 `AttentionTimer`를 `OpTimer`로 일반화하고 `Engine.run(ops_mode="event")`가 클래스별 행을 `RunResult.ops`로 반환한다. 새 패키지 `kernelscope/diagnose/`(opmodel·report·figures·run)가 그 행을 바이트·FLOPs 모델과 `MachineSpec` 상한에 결합해 `diagnosis.json`을 만든다. 성능 캠페인 경로(`serve run`)는 `AttentionTimer`가 attention만 기록하므로 값과 경로가 바뀌지 않는다.

**Tech Stack:** Python 3.11, torch 2.8 (CUDA event), pandas/pyarrow, numpy, matplotlib(Agg), pytest. 실행 파이썬은 `.venv/bin/python`(= `/home/skkai/miniforge3/envs/gradkernel/bin/python`과 동일 환경).

**Spec:** [docs/plan/2026-09-27-design-op-breakdown-diagnose.md](2026-09-27-design-op-breakdown-diagnose.md)

## Global Constraints

- **커밋·푸시 금지**: CLAUDE.md 규칙. 각 Task는 테스트 통과에서 끝난다. 커밋은 사용자가 요청할 때 한 번에 한다.
- Codex 영역(`kernelscope/backends/accelsim`, simsweep) 수정 금지.
- `serve run`의 `STEP_COLUMNS`, `attn_us` 정의, `AttentionTimer.start(layer)/stop(layer)/total_us()` 시그니처 불변.
- `kernelscope/diagnose/report.py`, `opmodel.py`는 **torch를 import하지 않는다**(GPU 없는 노트북의 `verify` 경로).
- 새 실행 결과의 `evidence_kind = "diagnostic_op_breakdown"`, `performance_claim = false`.
- 문서에 프로파일러 우월성 표현 금지. 커널 내부(stall, bank conflict)는 ncu의 영역이라고 쓴다.
- 문서에 쓰는 수치는 `kernelscope verify` 항목을 함께 추가한다.
- 테스트 실행: `.venv/bin/python -m pytest -q -p no:cacheprovider <file>::<test>`; 전체는 `make test`.
- GPU가 필요한 Task 0·8은 이 4090 호스트에서 메인 세션이 직접 실행한다.

## Review Focus

1. arrivals처럼 step마다 B·lens가 달라지는 실행 → attention 요약은 step별 행을 남기고 최빈값·평균으로 요약해야 한다 (Task 5 테스트 `test_attention_steps_keep_one_row_per_step`).
2. `control_000/`이 없는 정책 폴더 → `timer_overhead_pct`가 NaN이고 진단은 계속되어야 한다 (Task 5 `test_missing_control_run_gives_nan_overhead`).
3. 어떤 클래스의 행이 하나도 없는 ops(예: rope 없음) → gpu_us 0, 판정 `launch_bound`, achieved NaN, 예외 없음 (Task 5 `test_missing_class_rows_are_zero_not_error`).
4. 청크보다 긴 prompt의 prefill → 비용은 청크 합, 시간은 rid 합 (Task 5 `test_prefill_costs_sum_over_chunks`).
5. `source="unavailable"`·NaN이 섞인 요약을 JSON으로 쓸 때 `allow_nan=False`로 실패하면 안 된다 (Task 5 `test_write_serializes_nan_as_null`).

---

### Task 0: A8 기준선 — 변경 전 `serve run` step 시간 기록 (GPU, 메인 세션)

**Files:**
- Create (스크래치): `$SCR/a8/before/` (`$SCR` = 세션 스크래치 디렉터리)

- [ ] **Step 1: 유휴 확인**

Run: `cd /home/skkai/AI_Accelerator/kernelscope-design && .venv/bin/python -m kernelscope.cli serve doctor | tail -3`
Expected: `"ready": true`

- [ ] **Step 2: 균일 시나리오 heuristic 2회 측정**

```bash
SCR=/tmp/claude-1000/-home-skkai-AI-Accelerator/23bbed8c-8903-48a4-98da-b0094db7da3b/scratchpad
mkdir -p $SCR/a8 && cd /home/skkai/AI_Accelerator/kernelscope-design && HF_HUB_OFFLINE=1 \
.venv/bin/python -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario scenarios/graduation_uniform.yaml --policy heuristic --kv-gib 10 \
  --repeats 2 --warmup-runs 1 --warmup-steps 2 --out $SCR/a8/before
```

- [ ] **Step 3: step_us 중앙값 기록**

```bash
.venv/bin/python - <<'PY'
import glob, json, pandas as pd
SCR="/tmp/claude-1000/-home-skkai-AI-Accelerator/23bbed8c-8903-48a4-98da-b0094db7da3b/scratchpad"
s = pd.concat(pd.read_parquet(p) for p in glob.glob(f"{SCR}/a8/before/heuristic/repeat_*/steps.parquet"))
json.dump({"step_us_median": float(s.step_us.median()), "attn_us_median": float(s.attn_us.median()), "n": len(s)},
          open(f"{SCR}/a8/before.json", "w"), indent=2)
print(open(f"{SCR}/a8/before.json").read())
PY
```
Expected: JSON with `step_us_median` around 15,000–25,000 µs.

---

### Task 1: `OpTimer`와 `_forward`/`decode`/`prefill`의 영역 계측

**Files:**
- Modify: `kernelscope/serve/model.py` (imports 9–16, `AttentionTimer` 29–62, `_forward` 195–229, `prefill` 236–258, `decode` 260–281)
- Test: `tests/test_serve_model.py` (파일 끝에 추가)

**Interfaces:**
- Produces: `OP_CLASSES`, `NULL_REGION`, `class OpTimer(device=None, classes=OP_CLASSES)` with `region(op_class, layer=-1)`, `start(op_class, layer=-1)`, `stop(op_class, layer=-1)`, `rows() -> list[dict(layer, op_class, gpu_us)]`, `total_us(op_class="attention") -> float`; `class AttentionTimer(OpTimer)` with unchanged `start(layer)/stop(layer)/total_us()`; `DecoderModel.prefill(seq_id, token_ids, pool, chunk=4096, *, timer=None)`.

- [ ] **Step 1: 실패하는 테스트 추가** (`tests/test_serve_model.py` 끝)

```python
from kernelscope.serve.model import NULL_REGION, OP_CLASSES, OpTimer


def _prefilled(seed=0):
    model = DecoderModel.random(TINY, device="cpu", seed=seed)
    pool = _pool(model)
    model.prefill(0, list(range(1, 6)), pool)
    model.prefill(1, list(range(1, 9)), pool)
    return model, pool


def test_op_timer_records_every_class_for_every_layer_on_decode():
    model, pool = _prefilled()
    timer = OpTimer(device="cpu")
    model.decode([0, 1], [3, 4], pool, timer=timer)
    rows = timer.rows()
    layers = {}
    for row in rows:
        layers.setdefault(row["op_class"], set()).add(row["layer"])
    assert set(layers) == set(OP_CLASSES)
    for op_class in ("norm", "qkv_proj", "rope", "attention", "o_proj", "mlp"):
        assert layers[op_class] == set(range(TINY.n_layers))
    assert layers["embed"] == {-1} and layers["lm_head"] == {-1}
    assert all(row["gpu_us"] >= 0 for row in rows)
    attention = sum(row["gpu_us"] for row in rows if row["op_class"] == "attention")
    assert timer.total_us("attention") == pytest.approx(attention)


def test_prefill_accepts_a_timer_and_records_one_embed_per_chunk():
    model = DecoderModel.random(TINY, device="cpu")
    pool = _pool(model)
    timer = OpTimer(device="cpu")
    model.prefill(0, list(range(1, 12)), pool, chunk=4, timer=timer)
    rows = timer.rows()
    assert {row["op_class"] for row in rows} == set(OP_CLASSES)
    assert sum(row["op_class"] == "embed" for row in rows) == 3      # 11 tokens in chunks of 4


def test_attention_timer_keeps_layer_api_and_skips_other_classes():
    timer = AttentionTimer(device="cpu")
    assert timer.region("mlp", 0) is NULL_REGION
    timer.start(0)
    timer.stop(0)
    with timer.region("attention", 1):
        pass
    assert timer.total_us() >= 0
    assert {row["layer"] for row in timer.rows()} == {0, 1}
    timer.start(2)
    with pytest.raises(RuntimeError, match="already running"):
        timer.start(2)


def test_op_timer_rejects_unknown_classes_and_unfinished_reads():
    with pytest.raises(ValueError, match="unknown op class"):
        OpTimer(device="cpu", classes=("attention", "softmax"))
    timer = OpTimer(device="cpu")
    timer.start("mlp", 0)
    with pytest.raises(RuntimeError, match="unfinished"):
        timer.rows()


def test_instrumentation_does_not_change_decode_logits():
    model, pool = _prefilled()
    plain = model.decode([0, 1], [3, 4], pool)
    model2, pool2 = _prefilled()
    timed = model2.decode([0, 1], [3, 4], pool2, timer=OpTimer(device="cpu"))
    assert torch.equal(plain, timed)
```

- [ ] **Step 2: 실패 확인**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_model.py -k "op_timer or prefill_accepts or attention_timer_keeps or instrumentation"`
Expected: ImportError `cannot import name 'NULL_REGION'`.

- [ ] **Step 3: 타이머 구현** — `model.py` 9–16행 import에 `import contextlib` 추가, 29–62행 `AttentionTimer`를 아래로 교체

```python
OP_CLASSES = ("embed", "norm", "qkv_proj", "rope", "attention", "o_proj", "mlp", "lm_head")
NULL_REGION = contextlib.nullcontext()


def _null_region(op_class, layer=-1):
    return NULL_REGION


class _Region:
    __slots__ = ("timer", "op_class", "layer")

    def __init__(self, timer, op_class, layer):
        self.timer, self.op_class, self.layer = timer, op_class, layer

    def __enter__(self):
        self.timer._start(self.op_class, self.layer)

    def __exit__(self, *exc):
        self.timer._stop(self.op_class, self.layer)
        return False


class OpTimer:
    """GPU time per (operation class, layer) from CUDA event pairs, or a CPU monotonic clock.

    Regions of classes outside ``classes`` cost one dict lookup and a shared null context, so the
    attention-only subclass used by performance campaigns adds no events for the other classes.
    """

    def __init__(self, device=None, classes=OP_CLASSES):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classes = tuple(classes)
        unknown = set(self.classes) - set(OP_CLASSES)
        if unknown:
            raise ValueError(f"unknown op class {sorted(unknown)}; expected a subset of {OP_CLASSES}")
        self._open = {}
        self._regions = []

    def _stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter_ns()

    def _start(self, op_class, layer):
        key = (op_class, layer)
        if key in self._open:
            raise RuntimeError(f"{op_class} layer {layer} timer is already running")
        self._open[key] = self._stamp()

    def _stop(self, op_class, layer):
        key = (op_class, layer)
        if key not in self._open:
            raise RuntimeError(f"{op_class} layer {layer} timer was not started")
        self._regions.append((op_class, layer, self._open.pop(key), self._stamp()))

    def start(self, op_class, layer=-1):
        self._start(op_class, layer)

    def stop(self, op_class, layer=-1):
        self._stop(op_class, layer)

    def region(self, op_class, layer=-1):
        if op_class not in self.classes:
            return NULL_REGION
        return _Region(self, op_class, layer)

    def _elapsed_us(self, start, end):
        if self.device.type == "cuda":
            end.synchronize()
            return start.elapsed_time(end) * 1000
        return (end - start) / 1000

    def rows(self) -> list:
        if self._open:
            raise RuntimeError("cannot read an unfinished op timer")
        return [{"layer": layer, "op_class": op_class, "gpu_us": self._elapsed_us(start, end)}
                for op_class, layer, start, end in self._regions]

    def total_us(self, op_class="attention") -> float:
        if self._open:
            raise RuntimeError("cannot read an unfinished op timer")
        return sum(self._elapsed_us(start, end) for cls, _, start, end in self._regions if cls == op_class)


class AttentionTimer(OpTimer):
    """Attention-only timer of the performance campaigns; keeps the layer-only start/stop API."""

    def __init__(self, device=None):
        super().__init__(device, classes=("attention",))

    def start(self, layer):
        self._start("attention", layer)

    def stop(self, layer):
        self._stop("attention", layer)

    def total_us(self) -> float:
        return super().total_us("attention")
```

- [ ] **Step 4: `_forward` 교체** (195–229행 전체)

```python
    def _forward(self, x, positions, pool, seq_ids, cache_lens, num_splits, timer=None):
        cfg, weights = self.cfg, self.w
        batch, tokens, _ = x.shape
        reg = timer.region if timer is not None else _null_region
        cos, sin = self._rope(positions)
        table = pool.block_table(seq_ids)
        for layer in range(cfg.n_layers):
            prefix = f"model.layers.{layer}."
            with reg("norm", layer):
                h = _rms(x, weights[prefix + "input_layernorm.weight"], cfg.rms_eps)
            with reg("qkv_proj", layer):
                q = F.linear(h, weights[prefix + "self_attn.q_proj.weight"]).view(batch, tokens, cfg.n_heads, cfg.head_dim)
                k = F.linear(h, weights[prefix + "self_attn.k_proj.weight"]).view(batch, tokens, cfg.n_kv_heads, cfg.head_dim)
                v = F.linear(h, weights[prefix + "self_attn.v_proj.weight"]).view(batch, tokens, cfg.n_kv_heads, cfg.head_dim)
            if cfg.qk_norm:
                with reg("norm", layer):
                    q = _rms(q, weights[prefix + "self_attn.q_norm.weight"], cfg.rms_eps)
                    k = _rms(k, weights[prefix + "self_attn.k_norm.weight"], cfg.rms_eps)
            with reg("rope", layer):
                q = q * cos + _rotate_half(q) * sin
                k = k * cos + _rotate_half(k) * sin
            with reg("attention", layer):
                if self._flash_attention is not None:
                    attn = self._flash_attention(
                        q, pool.k[layer], pool.v[layer], k=k, v=v,
                        cache_seqlens=cache_lens, block_table=table,
                        softmax_scale=cfg.head_dim ** -0.5, causal=True, num_splits=num_splits,
                    )
                else:
                    attn = self._cpu_attention(q, k, v, pool, layer, cache_lens, table)
            with reg("o_proj", layer):
                x = x + F.linear(attn.reshape(batch, tokens, cfg.n_heads * cfg.head_dim),
                                 weights[prefix + "self_attn.o_proj.weight"])
            with reg("norm", layer):
                h = _rms(x, weights[prefix + "post_attention_layernorm.weight"], cfg.rms_eps)
            with reg("mlp", layer):
                gate = F.silu(F.linear(h, weights[prefix + "mlp.gate_proj.weight"]))
                up = F.linear(h, weights[prefix + "mlp.up_proj.weight"])
                x = x + F.linear(gate * up, weights[prefix + "mlp.down_proj.weight"])
        return x
```

- [ ] **Step 5: `prefill`·`decode`에 embed·lm_head 영역 추가**

`prefill` 시그니처를 `def prefill(self, seq_id, token_ids: list[int], pool, chunk=4096, *, timer=None):`로 바꾸고, 청크 루프를:

```python
        reg = timer.region if timer is not None else _null_region
        for start in range(0, ids.numel(), chunk):
            stop = min(ids.numel(), start + chunk)
            with reg("embed"):
                x = F.embedding(ids[start:stop], self.w["model.embed_tokens.weight"]).unsqueeze(0)
            positions = torch.arange(start, stop, device=self.device).unsqueeze(0)
            h = self._forward(x, positions, pool, [seq_id], pool.lengths([seq_id]), num_splits=0, timer=timer)
            pool.set_length(seq_id, stop)
        with reg("lm_head"):
            return self._logits(h[0, -1])
```

`decode`의 마지막 네 줄을:

```python
        reg = timer.region if timer is not None else _null_region
        with reg("embed"):
            x = F.embedding(ids, self.w["model.embed_tokens.weight"]).unsqueeze(1)
        h = self._forward(x, positions, pool, seq_ids, cache_lens, num_splits, timer)
        with reg("lm_head"):
            logits = self._logits(h[:, -1])
        for seq_id, length in zip(seq_ids, lengths):
            pool.set_length(seq_id, length + 1)
        return logits
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_model.py tests/test_serve_engine.py tests/test_serve_equivalence.py -m "not gpu"`
Expected: all PASS (기존 테스트 포함).

---

### Task 2: `MachineSpec.tc_tflops`와 ridge

**Files:**
- Modify: `kernelscope/model/machine.py`
- Test: `tests/test_model_machine.py` (끝에 추가)

**Interfaces:**
- Produces: `MachineSpec.tc_tflops: float | None` (마지막 필드, 기본 None), `MachineSpec.ridge_flop_per_byte() -> float` (없으면 `ValueError("... tc_tflops ...")`).

- [ ] **Step 1: 실패하는 테스트**

```python
def test_tensor_core_ceiling_and_ridge(m):
    assert m.tc_tflops == 168.6
    assert m.ridge_flop_per_byte() == pytest.approx(168.6e12 / 952.6e9)
    assert m.scaled(dram=0.5).ridge_flop_per_byte() == pytest.approx(2 * m.ridge_flop_per_byte())


def test_missing_tensor_core_ceiling_is_none_and_ridge_refuses(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({k: v for k, v in SPEC.items() if k != "tc_tflops"}))
    m = MachineSpec.from_json(p)
    assert m.tc_tflops is None
    with pytest.raises(ValueError, match="tc_tflops"):
        m.ridge_flop_per_byte()
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_model_machine.py -k ceiling` → AttributeError.

- [ ] **Step 3: 구현** — dataclass 끝에 `tc_tflops: float | None = None` 추가; `from_json`의 `cls(...)` 호출에 `tc_tflops=float(d["tc_tflops"]) if d.get("tc_tflops") is not None else None` 추가; 메서드 추가:

```python
    def ridge_flop_per_byte(self) -> float:
        """Arithmetic intensity where the tensor-core roof meets the DRAM roof (FLOP per byte)."""
        if self.tc_tflops is None:
            raise ValueError("machine spec has no tc_tflops ceiling")
        return self.tc_tflops * 1e12 / (self.dram_gbps * 1e9)
```

- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_model_machine.py tests/test_model_predict.py` → PASS.

---

### Task 3: `Engine.run(ops_mode)`와 `RunResult.ops`

**Files:**
- Modify: `kernelscope/serve/engine.py`
- Test: `tests/test_serve_engine.py` (끝에 추가; 파일 상단에 `import contextlib` 추가)

**Interfaces:**
- Consumes: `OpTimer`, `AttentionTimer` (Task 1).
- Produces: `OPS_COLUMNS = ["phase", "step", "rid", "layer", "op_class", "gpu_us"]`; `RunResult.ops: pd.DataFrame`; `Engine.run(requests, vocab, record_logits_steps=0, seed=0, ops_mode=None)`; `RunResult.metadata["ops_mode"]`.

- [ ] **Step 1: 실패하는 테스트**

```python
from kernelscope.serve.engine import OPS_COLUMNS, STEP_COLUMNS


class RegionModel(ArithmeticModel):
    """Scheduler double that opens op regions the way DecoderModel does."""

    def prefill(self, rid, tokens, pool, timer=None):
        reg = timer.region if timer is not None else (lambda *a: contextlib.nullcontext())
        with reg("embed"):
            pool.reserve(rid, len(tokens))
            pool.set_length(rid, len(tokens))
        with reg("lm_head"):
            return self._logits(sum(tokens))

    def decode(self, seq_ids, tokens, pool, num_splits=0, timer=None):
        with timer.region("embed"):
            for rid in seq_ids:
                length = pool.length(rid) + 1
                pool.reserve(rid, length)
                pool.set_length(rid, length)
        with timer.region("attention", 0):
            output = torch.stack([self._logits(token + 1) for token in tokens])
        with timer.region("mlp", 0):
            pass
        return output


def test_event_mode_collects_op_rows_and_keeps_step_columns():
    result = Engine(RegionModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2), Request(1, 2, 2)], 16, ops_mode="event")
    assert list(result.steps.columns) == STEP_COLUMNS
    assert list(result.ops.columns) == OPS_COLUMNS
    decode = result.ops[result.ops.phase == "decode"]
    prefill = result.ops[result.ops.phase == "prefill"]
    assert (decode.rid == "").all() and set(decode.op_class) == {"embed", "attention", "mlp"}
    assert set(prefill.rid) == {"0", "1"} and set(prefill.op_class) == {"embed", "lm_head"}
    per_step = decode[decode.op_class == "attention"].groupby("step").gpu_us.sum()
    assert per_step.index.tolist() == result.steps.step.tolist()
    assert result.metadata["ops_mode"] == "event"


def test_default_mode_has_no_op_rows_and_accepts_the_old_double():
    result = Engine(ArithmeticModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2)], 16)
    assert result.ops.empty and list(result.ops.columns) == OPS_COLUMNS
    assert result.metadata["ops_mode"] is None


def test_unknown_ops_mode_is_rejected():
    with pytest.raises(ValueError, match="ops_mode"):
        Engine(ArithmeticModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2)], 16, ops_mode="trace")
```

`Request(...)`의 위치 인자 순서는 파일 안 기존 호출(`grep -n "Request(" tests/test_serve_engine.py | head -3`)과 같게 맞춘다 (rid, prompt_len, max_new_tokens).

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_engine.py -k "event_mode or default_mode or unknown_ops"` → ImportError `OPS_COLUMNS`.

- [ ] **Step 3: 구현** — `engine.py`:

상수·데이터클래스:
```python
from kernelscope.serve.model import AttentionTimer, OpTimer   # AttentionTimer import가 이미 있으면 OpTimer만 추가
OPS_COLUMNS = ["phase", "step", "rid", "layer", "op_class", "gpu_us"]
OPS_MODES = (None, "event")


@dataclass
class RunResult:
    steps: pd.DataFrame
    tokens: pd.DataFrame
    prefill: pd.DataFrame
    logits: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    ops: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=OPS_COLUMNS))
```

`run` 시그니처 `def run(self, requests, vocab, record_logits_steps=0, seed=0, ops_mode=None) -> RunResult:` 첫 검사 뒤에:
```python
        if ops_mode not in OPS_MODES:
            raise ValueError(f"ops_mode must be one of {OPS_MODES}, got {ops_mode!r}")
```
`steps, tokens, prefills, ... = [], ...` 줄에 `ops_rows = []` 추가. prefill 호출부(`timer.start(); output = self.model.prefill(...)`)를:
```python
                    timer.start()
                    if ops_mode == "event":
                        ops = OpTimer(device=device)
                        output = self.model.prefill(request.rid, ids, self.pool, timer=ops)
                    else:
                        output = self.model.prefill(request.rid, ids, self.pool)
                    elapsed = timer.stop()
                    if ops_mode == "event":
                        ops_rows.extend({"phase": "prefill", "step": step, "rid": str(request.rid), **row} for row in ops.rows())
```
decode 부분 `attention = AttentionTimer(device=device)`를 `attention = OpTimer(device=device) if ops_mode == "event" else AttentionTimer(device=device)`로; `step_us = timer.stop()` 뒤에:
```python
                if ops_mode == "event":
                    ops_rows.extend({"phase": "decode", "step": step, "rid": "", **row} for row in attention.rows())
```
반환 metadata dict에 `"ops_mode": ops_mode,` 추가하고 `RunResult(..., {...}, ops=pd.DataFrame(ops_rows, columns=OPS_COLUMNS))`.

- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_engine.py tests/test_serve_equivalence.py tests/test_serve_cli.py` → PASS.

---

### Task 4: `kernelscope/diagnose/opmodel.py`

**Files:**
- Create: `kernelscope/diagnose/__init__.py` (docstring 한 줄: `"""Op-class time attribution and ceiling verdicts for the serving decoder (GPU-free analysis)."""`)
- Create: `kernelscope/diagnose/opmodel.py`
- Test: `tests/test_diagnose_opmodel.py`

**Interfaces:**
- Produces: `OP_CLASSES`, `LAYER_CLASSES`, `OpCost(weight_bytes, act_bytes, flops)` with `.bytes`, `dtype_bytes(dtype: str) -> int`, `op_costs(cfg, dtype: str, phase: str, tokens: int, lens) -> dict[str, OpCost]` (`cfg`는 `hidden, intermediate, vocab, n_layers, n_heads, n_kv_heads, head_dim, qk_norm` 속성만 사용).

- [ ] **Step 1: 실패하는 테스트**

```python
from types import SimpleNamespace

import pytest

from kernelscope.analytic import attention_flops, attention_traffic
from kernelscope.diagnose.opmodel import LAYER_CLASSES, OP_CLASSES, OpCost, dtype_bytes, op_costs
from kernelscope.workload import Workload

CFG = SimpleNamespace(n_layers=2, hidden=24, intermediate=48, n_heads=4, n_kv_heads=2, head_dim=8, vocab=128, qk_norm=True)


def test_decode_costs_match_hand_calculation():
    c = op_costs(CFG, "bfloat16", "decode", 3, (5, 7, 9))
    assert set(c) == set(OP_CLASSES)
    b, T, H, I, V, L = 2, 3, 24, 48, 128, 2
    assert c["qkv_proj"].flops == L * 2 * T * H * (4 + 2 * 2) * 8
    assert c["qkv_proj"].weight_bytes == L * H * (4 + 4) * 8 * b
    assert c["mlp"] == OpCost(L * 3 * H * I * b, L * T * b * (3 * H + 6 * I), L * 2 * T * 3 * H * I)
    assert c["embed"] == OpCost(T * H * b, T * H * b, 0)
    assert c["lm_head"] == OpCost(H * b + V * H * b, 2 * T * H * b + T * V * 4, 2 * T * V * H)
    assert c["norm"].act_bytes == L * (2 * (2 * T * H * b) + 2 * T * (4 + 2) * 8 * b) and c["norm"].flops == 0
    w = Workload("decode", 3, 1, 9, 4, 2, 8, "bfloat16", kv_lens=(5, 7, 9))
    assert c["attention"] == OpCost(0, L * attention_traffic(w)["total_bytes"], L * attention_flops(w))
    assert c["attention"].bytes == c["attention"].act_bytes


def test_uniform_decode_uses_a_plain_workload_key():
    c = op_costs(CFG, "float16", "decode", 2, (6, 6))
    w = Workload("decode", 2, 1, 6, 4, 2, 8, "float16")
    assert c["attention"].flops == 2 * attention_flops(w)


def test_prefill_chunk_costs_use_visible_keys():
    c = op_costs(CFG, "float32", "prefill", 5, (13,))
    w = Workload("prefill", 1, 5, 13, 4, 2, 8, "float32")
    assert c["attention"].flops == 2 * attention_flops(w)
    assert c["qkv_proj"].flops == 2 * 2 * 5 * 24 * 8 * 8


def test_rejects_bad_inputs():
    assert dtype_bytes("torch.bfloat16") == 2
    with pytest.raises(ValueError, match="dtype"):
        dtype_bytes("int7")
    with pytest.raises(ValueError, match="phase"):
        op_costs(CFG, "float16", "train", 1, (1,))
    with pytest.raises(ValueError, match="one KV length"):
        op_costs(CFG, "float16", "decode", 2, (4,))
    with pytest.raises(ValueError, match="single"):
        op_costs(CFG, "float16", "prefill", 2, (4, 4))


def test_layer_classes_are_the_per_layer_subset():
    assert set(LAYER_CLASSES) < set(OP_CLASSES) and "embed" not in LAYER_CLASSES


def test_op_classes_match_the_serving_timer():
    torch = pytest.importorskip("torch")
    from kernelscope.serve.model import OP_CLASSES as TIMER_CLASSES
    assert TIMER_CLASSES == OP_CLASSES
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_opmodel.py` → ModuleNotFoundError.

- [ ] **Step 3: 구현** (`kernelscope/diagnose/opmodel.py`)

```python
"""Compulsory bytes and FLOPs of one decode step (or one prefill chunk) per operation class.

Weights are read once and activations are read and written once per use, so the totals are lower
bounds on real traffic; an achieved bandwidth derived from them is a lower bound too. Attention reuses
kernelscope.analytic. No torch here: this module also runs on the GPU-less verification path.
"""
from dataclasses import dataclass

from kernelscope.analytic import DTYPE_BYTES, attention_flops, attention_traffic
from kernelscope.workload import Workload

OP_CLASSES = ("embed", "norm", "qkv_proj", "rope", "attention", "o_proj", "mlp", "lm_head")
LAYER_CLASSES = ("norm", "qkv_proj", "rope", "attention", "o_proj", "mlp")
LOGIT_BYTES = 4      # DecoderModel._logits returns float32


@dataclass(frozen=True)
class OpCost:
    weight_bytes: float
    act_bytes: float
    flops: float

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.act_bytes


def dtype_bytes(dtype: str) -> int:
    name = str(dtype).removeprefix("torch.")
    if name not in DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; expected one of {sorted(DTYPE_BYTES)}")
    return DTYPE_BYTES[name]


def op_costs(cfg, dtype: str, phase: str, tokens: int, lens) -> dict:
    """Decode: tokens = batch size, lens = KV length of each row after the append.
    Prefill: tokens = chunk length, lens = (keys visible after the chunk,)."""
    if phase not in ("decode", "prefill"):
        raise ValueError(f"phase must be 'decode' or 'prefill', got {phase!r}")
    lens = tuple(int(n) for n in lens)
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1 or not lens or min(lens) < 1:
        raise ValueError("tokens must be a positive int and lens a nonempty tuple of positive ints")
    name = str(dtype).removeprefix("torch.")
    b = dtype_bytes(name)
    T, H, I, V, L = tokens, cfg.hidden, cfg.intermediate, cfg.vocab, cfg.n_layers
    Hq, Hkv, d = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    if phase == "decode":
        if len(lens) != T:
            raise ValueError("decode needs one KV length per batch row")
        w = Workload("decode", T, 1, max(lens), Hq, Hkv, d, name, kv_lens=lens if len(set(lens)) > 1 else None)
    else:
        if len(lens) != 1:
            raise ValueError("prefill chunk needs a single visible-key count")
        w = Workload("prefill", 1, T, lens[0], Hq, Hkv, d, name)
    qk = 2 * T * (Hq + Hkv) * d * b if cfg.qk_norm else 0
    per_layer = {
        "norm": OpCost(2 * H * b, 2 * (2 * T * H * b) + qk, 0),
        "qkv_proj": OpCost(H * (Hq + 2 * Hkv) * d * b, T * H * b + T * (Hq + 2 * Hkv) * d * b, 2 * T * H * (Hq + 2 * Hkv) * d),
        "rope": OpCost(0, 2 * T * (Hq + Hkv) * d * b, 0),
        "attention": OpCost(0, attention_traffic(w)["total_bytes"], attention_flops(w)),
        "o_proj": OpCost(Hq * d * H * b, T * Hq * d * b + 2 * T * H * b, 2 * T * Hq * d * H),
        "mlp": OpCost(3 * H * I * b, T * b * (3 * H + 6 * I), 2 * T * 3 * H * I),
    }
    costs = {"embed": OpCost(T * H * b, T * H * b, 0)}
    costs.update({k: OpCost(c.weight_bytes * L, c.act_bytes * L, c.flops * L) for k, c in per_layer.items()})
    costs["lm_head"] = OpCost(H * b + V * H * b, 2 * T * H * b + T * V * LOGIT_BYTES, 2 * T * V * H)
    return costs
```

- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_opmodel.py` → PASS.

---

### Task 5: `kernelscope/diagnose/report.py` — 실행 폴더 → 진단

**Files:**
- Create: `kernelscope/diagnose/report.py`
- Test: `tests/test_diagnose_report.py`

**Interfaces:**
- Consumes: `op_costs`, `OpCost`, `OP_CLASSES`, `LAYER_CLASSES` (Task 4); `MachineSpec.ridge_flop_per_byte`, `.tc_tflops`, `.dram_gbps` (Task 2); `kernelscope.serve.report.tpot_us`; `kernelscope.model.hybrid.variant_for_splits`(함수 안에서 lazy import).
- Produces: `VERDICTS`, `LAUNCH_US = 5.0`, `OPS_CSV_COLUMNS`, `verdict(...)`, `amdahl_bound(share, chosen_over_best)`, `load_table(csv_path) -> DataFrame`(열 `lens_list` 추가), `load_measured(data_root) -> dict[(workload_key, plugin), kernel_time_us]`, `nearest_cell(table, lens, n_heads, n_kv_heads) -> (row | None, distance)`, `decode_table(...)`, `prefill_table(...)`, `attention_steps(...)`, `attention_summary(...)`, `@dataclass Diagnosis(summary: dict, ops: DataFrame, attention_steps: DataFrame)`, `diagnose(run_dir, machine, table_csv, data_root, params=None, threshold=0.7) -> Diagnosis`, `write(run_dir, diagnosis) -> None`.

- [ ] **Step 1: 실패하는 테스트** (`tests/test_diagnose_report.py`)

```python
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kernelscope.diagnose import report
from kernelscope.diagnose.opmodel import OP_CLASSES
from kernelscope.model.machine import MachineSpec

ROOT = Path(__file__).resolve().parents[1]
CFG = dict(arch="qwen3", n_layers=2, hidden=24, intermediate=48, n_heads=4, n_kv_heads=2, head_dim=8, vocab=128,
           rope_theta=10000.0, rope_scaling=None, rms_eps=1e-6, tie_embeddings=True, qk_norm=True)
KEY = "decode_B2_Lq1_Lkv8+4x1_Hq4_Hkv2_d8_float16_causal"
# per-layer gpu_us chosen so that the verdict rules fire deterministically (see spec §2.3):
# mlp ≈ 33 KB per step in 0.04 µs → 835 GB/s ≥ 0.7·952.6 → memory_bound; rope 1 µs/layer → launch_bound;
# attention 150 µs/layer → parallelism_candidate; qkv 50 µs/layer → below_ceiling_unknown.
LAYER_US = {"norm": 3.0, "qkv_proj": 50.0, "rope": 1.0, "attention": 150.0, "o_proj": 20.0, "mlp": 0.02}
STEP_US, CONTROL_STEP_US = 1000.0, 950.0


def _steps(step_us, lens=(8, 4), n=2, num_splits=0):
    return pd.DataFrame([dict(step=i, policy="heuristic", B=len(lens), len_max=max(lens), len_sum=sum(lens), n_long=0,
                              num_splits=num_splits, attn_us=300.0, step_us=step_us, policy_us=10.0, policy_cache_hit=True,
                              decode_wall_us=step_us + 100.0, seq_ids=json.dumps([0, 1]), lens=json.dumps(list(lens)))
                         for i in range(n)])


def _tokens(n_steps=2):
    rows = [dict(rid=r, step=0, position=0, token=1, t_us=1000.0, phase="prefill") for r in (0, 1)]
    rows += [dict(rid=r, step=s, position=s + 1, token=2, t_us=1000.0 * (s + 2), phase="decode") for r in (0, 1) for s in range(n_steps)]
    return pd.DataFrame(rows)


def _prefill():
    return pd.DataFrame([dict(rid=0, prompt_len=8, prefill_us=500.0, arrival_us=0.0, admitted_us=0.0, first_token_us=500.0, prompt_sha256="a"),
                         dict(rid=1, prompt_len=4, prefill_us=300.0, arrival_us=0.0, admitted_us=500.0, first_token_us=800.0, prompt_sha256="b")])


def _ops(n_steps=2, drop=()):
    rows = []
    for step in range(n_steps):
        rows.append(dict(phase="decode", step=step, rid="", layer=-1, op_class="embed", gpu_us=2.0))
        for layer in range(CFG["n_layers"]):
            for op_class, us in LAYER_US.items():
                if op_class not in drop:
                    rows.append(dict(phase="decode", step=step, rid="", layer=layer, op_class=op_class, gpu_us=us))
        rows.append(dict(phase="decode", step=step, rid="", layer=-1, op_class="lm_head", gpu_us=5.0))
    for rid in ("0", "1"):
        rows.append(dict(phase="prefill", step=0, rid=rid, layer=-1, op_class="embed", gpu_us=1.0))
        for layer in range(CFG["n_layers"]):
            for op_class in LAYER_US:
                rows.append(dict(phase="prefill", step=0, rid=rid, layer=layer, op_class=op_class, gpu_us=10.0))
        rows.append(dict(phase="prefill", step=0, rid=rid, layer=-1, op_class="lm_head", gpu_us=1.0))
    return pd.DataFrame(rows)


def _run_dir(tmp_path, control=True, drop=()):
    run = tmp_path / "run"
    (run / "heuristic" / "event_000").mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"evidence_kind": "diagnostic_op_breakdown", "performance_claim": False,
                                                   "model": "tiny", "scenario": "s.yaml", "created_at": "t",
                                                   "model_config": CFG, "model_dtype": "float32"}))
    event = run / "heuristic" / "event_000"
    _steps(STEP_US).to_parquet(event / "steps.parquet", index=False)
    _tokens().to_parquet(event / "tokens.parquet", index=False)
    _prefill().to_parquet(event / "prefill.parquet", index=False)
    _ops(drop=drop).to_parquet(event / "ops.parquet", index=False)
    if control:
        ctrl = run / "heuristic" / "control_000"
        ctrl.mkdir()
        _steps(CONTROL_STEP_US).to_parquet(ctrl / "steps.parquet", index=False)
        _tokens().to_parquet(ctrl / "tokens.parquet", index=False)
        _prefill().to_parquet(ctrl / "prefill.parquet", index=False)
    return run


def _table(tmp_path, h_q=4, h_kv=2):
    csv = tmp_path / "table.csv"
    pd.DataFrame([dict(workload_key=KEY, B=2, L_kv=8, H_q=h_q, H_kv=h_kv, ragged=True, lens="8+4x1", n_variants=3,
                       best_kernel="fd_s2_paged", best_us=10.0, heuristic_us=30.0, heuristic_regret=2.0, fa2_us=30.0, fa2_regret=2.0)]
                 ).to_csv(csv, index=False)
    return csv


def _data(tmp_path):
    d = tmp_path / "data" / "hw_4090" / "ragged_s1_paged"
    d.mkdir(parents=True)
    rows = [dict(status="ok", plugin=p, workload_key=KEY, cache_state="cold", kernel_time_us=us)
            for p, us in (("flashdecoding_paged", 30.0), ("fd_s2_paged", 10.0), ("fa2_paged", 30.0))]
    rows.append(dict(status="ok", plugin="fd_s2_paged", workload_key=KEY, cache_state="warm", kernel_time_us=1.0))
    (d / "summaries.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path / "data"


@pytest.fixture
def machine():
    return MachineSpec.from_json(ROOT / "machines" / "rtx4090.json")


def test_verdict_rules_in_order(machine):
    ridge = machine.ridge_flop_per_byte()
    assert report.verdict("mlp", ridge / 2, 0.8 * machine.dram_gbps, 0.0, 100.0, machine, 0.7) == "memory_bound"
    assert report.verdict("mlp", 2 * ridge, 0.0, 0.8 * machine.tc_tflops, 100.0, machine, 0.7) == "compute_bound"
    assert report.verdict("rope", 0.0, 1.0, 0.0, 1.0, machine, 0.7) == "launch_bound"
    assert report.verdict("attention", 1.0, 1.0, 0.0, 100.0, machine, 0.7) == "parallelism_candidate"
    assert report.verdict("qkv_proj", 1.0, 1.0, 0.0, 100.0, machine, 0.7) == "below_ceiling_unknown"


def test_amdahl_bound_formula():
    assert report.amdahl_bound(0.72, 3.84) == pytest.approx(1 / (0.28 + 0.72 / 3.84))


def test_decode_table_shares_costs_and_verdicts(tmp_path, machine):
    run = _run_dir(tmp_path)
    steps = pd.read_parquet(run / "heuristic/event_000/steps.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    t = report.decode_table(steps, ops, report.config_from(CFG), "float32", machine, 0.7, "heuristic").set_index("op_class")
    assert list(t.index) == list(OP_CLASSES)
    assert t.loc["attention", "gpu_us"] == pytest.approx(300.0) and t.loc["attention", "share"] == pytest.approx(0.3)
    assert t.loc["mlp", "verdict"] == "memory_bound" and t.loc["rope", "verdict"] == "launch_bound"
    assert t.loc["attention", "verdict"] == "parallelism_candidate" and t.loc["qkv_proj", "verdict"] == "below_ceiling_unknown"
    assert t.loc["mlp", "layer_mean_us"] == pytest.approx(0.02) and t.loc["embed", "layer_mean_us"] == pytest.approx(2.0)
    assert (t.bytes > 0).all() and t.loc["norm", "flops"] == 0 and t.loc["norm", "ai"] == 0
    assert t.loc["mlp", "pct_dram"] > 70 and 0 < t.loc["qkv_proj", "pct_dram"] < 70


def test_missing_class_rows_are_zero_not_error(tmp_path, machine):
    run = _run_dir(tmp_path, drop=("rope",))
    steps = pd.read_parquet(run / "heuristic/event_000/steps.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    t = report.decode_table(steps, ops, report.config_from(CFG), "float32", machine, 0.7, "heuristic").set_index("op_class")
    assert t.loc["rope", "gpu_us"] == 0 and t.loc["rope", "verdict"] == "launch_bound"
    assert math.isnan(t.loc["rope", "achieved_gbps"]) and math.isnan(t.loc["rope", "pct_dram"])


def test_prefill_costs_sum_over_chunks(tmp_path, machine):
    run = _run_dir(tmp_path)
    prefill = pd.read_parquet(run / "heuristic/event_000/prefill.parquet")
    ops = pd.read_parquet(run / "heuristic/event_000/ops.parquet")
    cfg = report.config_from(CFG)
    whole = report.prefill_table(prefill, ops, cfg, "float32", machine, 0.7, "heuristic", chunk=4096).set_index("op_class")
    chunked = report.prefill_table(prefill, ops, cfg, "float32", machine, 0.7, "heuristic", chunk=3).set_index("op_class")
    assert whole.loc["attention", "gpu_us"] == pytest.approx(20.0)                       # 2 layers × 10 µs, mean over 2 prefills
    assert whole.loc["mlp", "flops"] == chunked.loc["mlp", "flops"]                       # GEMM flops do not depend on chunking
    assert chunked.loc["attention", "flops"] == whole.loc["attention", "flops"]           # causal pairs do not depend on chunking
    assert chunked.loc["attention", "bytes"] > whole.loc["attention", "bytes"]             # but K/V are re-read per chunk
    assert whole.loc["attention", "share"] == pytest.approx(20.0 / 400.0)                 # base = mean prefill_us


def test_attention_steps_keep_one_row_per_step_and_find_the_measured_cell(tmp_path, machine):
    run = _run_dir(tmp_path)
    steps = pd.concat([_steps(STEP_US, lens=(8, 4), n=1), _steps(STEP_US, lens=(9, 4), n=1).assign(step=1)])
    table = report.load_table(_table(tmp_path))
    measured = report.load_measured(_data(tmp_path))
    assert measured[(KEY, "fd_s2_paged")] == 10.0 and len(measured) == 3                    # the warm row is skipped
    rows = report.attention_steps(steps, report.config_from(CFG), table, measured)
    assert len(rows) == 2 and rows.iloc[0].neighbor_distance == pytest.approx(0.0) and rows.iloc[1].neighbor_distance > 0
    assert (rows.source == "nearest_measured").all() and (rows.chosen_variant == "flashdecoding_paged").all()
    assert (rows.best_alternative == "fd_s2_paged").all() and rows.regret.iloc[0] == pytest.approx(2.0)
    summary = report.attention_summary(rows, 0.3)
    assert summary["chosen_over_best"] == pytest.approx(3.0) and summary["regret"] == pytest.approx(2.0)
    assert summary["amdahl_bound"] == pytest.approx(report.amdahl_bound(0.3, 3.0)) and summary["neighbor_key"] == KEY


def test_attention_falls_back_when_the_table_lacks_the_head_shape(tmp_path):
    steps = _steps(STEP_US)
    table = report.load_table(_table(tmp_path, h_q=32, h_kv=8))
    rows = report.attention_steps(steps, report.config_from(CFG), table, {})
    assert (rows.source == "unavailable").all() and rows.regret.isna().all()
    summary = report.attention_summary(rows, 0.3)
    assert math.isnan(summary["amdahl_bound"]) and summary["chosen_variant"] == "flashdecoding_paged"


def test_diagnose_and_write_round_trip(tmp_path, machine):
    run = _run_dir(tmp_path)
    result = report.diagnose(run, machine, _table(tmp_path), _data(tmp_path))
    p = result.summary["policies"]["heuristic"]
    expected_unattributed = 100 * (STEP_US - (2 + 5 + 2 * sum(LAYER_US.values()))) / STEP_US
    assert p["unattributed_pct"] == pytest.approx(expected_unattributed)
    assert p["timer_overhead_pct"] == pytest.approx(100 * (STEP_US / CONTROL_STEP_US - 1))
    assert p["step_waterfall"]["host_residual_us"] == pytest.approx(90.0) and p["tpot_waterfall"]["tpot_us_mean"] == pytest.approx(1000.0)
    assert p["attention"]["amdahl_bound"] == pytest.approx(report.amdahl_bound(0.3, 3.0))
    assert result.summary["ceilings"]["threshold"] == 0.7 and result.summary["identity"]["performance_claim"] is False
    assert len(result.ops) == 2 * len(OP_CLASSES) and set(result.ops.phase) == {"decode", "prefill"}
    report.write(run, result)
    loaded = json.loads((run / "diagnosis.json").read_text())
    assert loaded["policies"]["heuristic"]["n_steps"] == 2
    assert list(pd.read_csv(run / "ops.csv").columns) == report.OPS_CSV_COLUMNS
    assert len(pd.read_csv(run / "attention_steps.csv")) == 2


def test_missing_control_run_gives_nan_overhead(tmp_path, machine):
    run = _run_dir(tmp_path, control=False)
    result = report.diagnose(run, machine, _table(tmp_path), _data(tmp_path))
    assert math.isnan(result.summary["policies"]["heuristic"]["timer_overhead_pct"])


def test_write_serializes_nan_as_null(tmp_path, machine):
    run = _run_dir(tmp_path, control=False)
    result = report.diagnose(run, machine, _table(tmp_path, h_q=32, h_kv=8), tmp_path / "nowhere")
    report.write(run, result)
    loaded = json.loads((run / "diagnosis.json").read_text())
    assert loaded["policies"]["heuristic"]["timer_overhead_pct"] is None
    assert loaded["policies"]["heuristic"]["attention"]["source"] == "unavailable"


def test_diagnose_refuses_a_folder_without_event_runs(tmp_path, machine):
    (tmp_path / "manifest.json").write_text(json.dumps({"model_config": CFG, "model_dtype": "float32"}))
    with pytest.raises(FileNotFoundError, match="event_000"):
        report.diagnose(tmp_path, machine, None, None)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_report.py` → ModuleNotFoundError.

- [ ] **Step 3: 구현** (`kernelscope/diagnose/report.py`)

```python
"""Turn a `serve diagnose` run folder into a per-op-class ceiling report. GPU-free, no torch.

Reads the event run (op-class CUDA event rows) and the control run (no op timers) of every policy,
joins the measured GPU time per class with the compulsory bytes/FLOPs model and the machine
ceilings, and states per class whether it sits at the DRAM roof, the tensor-core roof, is launch
dominated, or is below both roofs for a reason this report cannot see. Attention also gets the
policy's chosen split variant against the nearest measured cell of the dispatch table.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from kernelscope.diagnose.opmodel import LAYER_CLASSES, OP_CLASSES, OpCost, op_costs
from kernelscope.serve.report import tpot_us

VERDICTS = ("memory_bound", "compute_bound", "launch_bound", "parallelism_candidate", "below_ceiling_unknown")
LAUNCH_US = 5.0
OPS_CSV_COLUMNS = ["policy", "phase", "op_class", "gpu_us", "share", "weight_bytes", "act_bytes", "bytes", "flops", "ai",
                   "achieved_gbps", "achieved_tflops", "pct_dram", "pct_tc", "layer_mean_us", "verdict"]
ATTENTION_COLUMNS = ["step", "B", "num_splits", "chosen_variant", "source", "neighbor_key", "neighbor_distance",
                     "best_alternative", "best_us", "chosen_us", "regret"]


@dataclass
class Diagnosis:
    summary: dict
    ops: pd.DataFrame
    attention_steps: pd.DataFrame


def config_from(model_config: dict):
    """Attribute view of manifest['model_config']; avoids importing torch-backed serve.hf here."""
    return SimpleNamespace(**model_config)


def verdict(op_class, ai, achieved_gbps, achieved_tflops, layer_mean_us, machine, threshold) -> str:
    ridge = machine.ridge_flop_per_byte()
    if ai < ridge and achieved_gbps >= threshold * machine.dram_gbps:
        return "memory_bound"
    if ai >= ridge and achieved_tflops >= threshold * machine.tc_tflops:
        return "compute_bound"
    if layer_mean_us < LAUNCH_US:
        return "launch_bound"
    if op_class == "attention":
        return "parallelism_candidate"
    return "below_ceiling_unknown"


def amdahl_bound(share: float, chosen_over_best: float) -> float:
    """Step speedup if attention alone reached the best variant (same formula as dashboard.data.amdahl_speedup)."""
    return 1.0 / (1.0 - share + share / chosen_over_best)


def _mean_costs(cost_dicts):
    return {k: OpCost(float(np.mean([c[k].weight_bytes for c in cost_dicts])),
                      float(np.mean([c[k].act_bytes for c in cost_dicts])),
                      float(np.mean([c[k].flops for c in cost_dicts]))) for k in cost_dicts[0]}


def _op_table(gpu_us, costs, base_us, n_layers, machine, threshold, policy, phase) -> pd.DataFrame:
    rows = []
    for op_class in OP_CLASSES:
        us, c = float(gpu_us.get(op_class, 0.0)), costs[op_class]
        ai = c.flops / c.bytes if c.bytes else math.inf
        gbps = c.bytes / us * 1e-3 if us > 0 else math.nan
        tflops = c.flops / us * 1e-6 if us > 0 else math.nan
        layer_mean = us / n_layers if op_class in LAYER_CLASSES else us
        rows.append({"policy": policy, "phase": phase, "op_class": op_class, "gpu_us": us,
                     "share": us / base_us if base_us > 0 else math.nan,
                     "weight_bytes": c.weight_bytes, "act_bytes": c.act_bytes, "bytes": c.bytes, "flops": c.flops, "ai": ai,
                     "achieved_gbps": gbps, "achieved_tflops": tflops,
                     "pct_dram": 100 * gbps / machine.dram_gbps if us > 0 else math.nan,
                     "pct_tc": 100 * tflops / machine.tc_tflops if us > 0 else math.nan,
                     "layer_mean_us": layer_mean,
                     "verdict": verdict(op_class, ai, gbps if us > 0 else 0.0, tflops if us > 0 else 0.0,
                                        layer_mean, machine, threshold)})
    return pd.DataFrame(rows, columns=OPS_CSV_COLUMNS)


def decode_table(steps, ops, cfg, dtype, machine, threshold, policy) -> pd.DataFrame:
    decode = ops[ops.phase == "decode"]
    per_step = decode.groupby(["step", "op_class"]).gpu_us.sum().unstack(fill_value=0.0)
    gpu_us = per_step.reindex(columns=OP_CLASSES, fill_value=0.0).mean().to_dict() if len(per_step) else {}
    costs = _mean_costs([op_costs(cfg, dtype, "decode", int(s.B), tuple(json.loads(s.lens))) for s in steps.itertuples()])
    return _op_table(gpu_us, costs, float(steps.step_us.mean()), cfg.n_layers, machine, threshold, policy, "decode")


def prefill_table(prefill, ops, cfg, dtype, machine, threshold, policy, chunk=4096) -> pd.DataFrame:
    rows = ops[ops.phase == "prefill"]
    if prefill.empty or rows.empty:
        return pd.DataFrame(columns=OPS_CSV_COLUMNS)
    per_rid = rows.groupby(["rid", "op_class"]).gpu_us.sum().unstack(fill_value=0.0)
    gpu_us = per_rid.reindex(columns=OP_CLASSES, fill_value=0.0).mean().to_dict()
    per_request = []
    for n in prefill.prompt_len.astype(int):
        chunks = [op_costs(cfg, dtype, "prefill", min(chunk, n - start), (min(chunk, n - start) + start,))
                  for start in range(0, n, chunk)]
        per_request.append({k: OpCost(sum(c[k].weight_bytes for c in chunks), sum(c[k].act_bytes for c in chunks),
                                      sum(c[k].flops for c in chunks)) for k in chunks[0]})
    return _op_table(gpu_us, _mean_costs(per_request), float(prefill.prefill_us.mean()), cfg.n_layers, machine,
                     threshold, policy, "prefill")


def _expand_lens(spec) -> list:
    """'32768+512x25' -> [32768, 512, ...]; '512x8' -> [512] * 8."""
    out = []
    for part in str(spec).split("+"):
        n, _, times = part.partition("x")
        out.extend([int(n)] * (int(times) if times else 1))
    return out


def _features(lens) -> np.ndarray:
    return np.log2([len(lens), max(lens), float(np.mean(lens))])       # TablePolicy's features


def load_table(csv_path) -> pd.DataFrame:
    if csv_path is None or not Path(csv_path).exists():
        return pd.DataFrame(columns=["workload_key", "H_q", "H_kv", "lens", "best_kernel", "best_us", "lens_list"])
    table = pd.read_csv(csv_path)
    table["lens_list"] = table.lens.map(_expand_lens)
    return table


def load_measured(data_root) -> dict:
    """(workload_key, plugin) -> cold kernel time of every recorded paged cell under data_root/hw_4090."""
    measured = {}
    if data_root is None or not Path(data_root).exists():
        return measured
    for path in sorted(Path(data_root).glob("hw_4090/*_paged/summaries.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("status") == "ok" and r.get("cache_state") == "cold":
                measured[(r["workload_key"], r["plugin"])] = float(r["kernel_time_us"])
    return measured


def nearest_cell(table, lens, n_heads, n_kv_heads):
    same = table[(table.H_q == n_heads) & (table.H_kv == n_kv_heads)]
    if same.empty:
        return None, math.nan
    d = np.sqrt([np.square(_features(l) - _features(lens)).sum() for l in same.lens_list])
    i = int(np.argmin(d))
    return same.iloc[i], float(d[i])


def attention_steps(steps, cfg, table, measured, machine=None, params=None, cache_state="cold") -> pd.DataFrame:
    from kernelscope.model.hybrid import variant_for_splits
    rows = []
    for s in steps.itertuples():
        lens, splits = json.loads(s.lens), int(s.num_splits)
        chosen = variant_for_splits(splits)
        cell, dist = nearest_cell(table, lens, cfg.n_heads, cfg.n_kv_heads)
        row = {"step": int(s.step), "B": int(s.B), "num_splits": splits, "chosen_variant": chosen, "source": "unavailable",
               "neighbor_key": None, "neighbor_distance": dist, "best_alternative": None, "best_us": math.nan, "chosen_us": math.nan}
        if cell is not None:
            chosen_us = float(cell.best_us) if chosen == cell.best_kernel else measured.get((cell.workload_key, chosen), math.nan)
            row.update(source="nearest_measured", neighbor_key=cell.workload_key, best_alternative=cell.best_kernel,
                       best_us=float(cell.best_us), chosen_us=float(chosen_us))
        elif params is not None and machine is not None:
            from kernelscope.model.predict import PAGED_VARIANTS, rank_variants
            from kernelscope.workload import Workload
            w = Workload("decode", len(lens), 1, max(lens), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, "float16",
                         kv_lens=tuple(lens) if len(set(lens)) > 1 else None)
            ranked = rank_variants(w, PAGED_VARIANTS, machine, params, cache_state)
            row.update(source="model", best_alternative=ranked[0].plugin, best_us=ranked[0].time_us,
                       chosen_us={p.plugin: p.time_us for p in ranked}.get(chosen, math.nan))
        row["regret"] = row["chosen_us"] / row["best_us"] - 1 if row["best_us"] > 0 else math.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=ATTENTION_COLUMNS)


def _mode(values):
    s = pd.Series([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))])
    return None if s.empty else s.mode().iloc[0]


def attention_summary(att_steps: pd.DataFrame, share_attention: float) -> dict:
    ratio = (att_steps.chosen_us / att_steps.best_us).replace([np.inf, -np.inf], np.nan).dropna()
    over = float(ratio.mean()) if len(ratio) else math.nan
    return {"n_steps": int(len(att_steps)), "num_splits": _mode(att_steps.num_splits), "chosen_variant": _mode(att_steps.chosen_variant),
            "source": _mode(att_steps.source), "neighbor_key": _mode(att_steps.neighbor_key),
            "neighbor_distance_mean": float(att_steps.neighbor_distance.mean()) if len(att_steps) else math.nan,
            "best_alternative": _mode(att_steps.best_alternative), "share": share_attention, "chosen_over_best": over,
            "regret": over - 1 if math.isfinite(over) else math.nan,
            "amdahl_bound": amdahl_bound(share_attention, over) if math.isfinite(over) and over > 0 else math.nan}


def _frames(directory: Path, ops=False):
    if not (directory / "steps.parquet").exists():
        return None
    frames = {name: pd.read_parquet(directory / f"{name}.parquet") for name in ("steps", "tokens", "prefill")}
    if ops:
        frames["ops"] = pd.read_parquet(directory / "ops.parquet")
    return frames


def diagnose(run_dir, machine, table_csv, data_root, params=None, threshold=0.7) -> Diagnosis:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    cfg, dtype = config_from(manifest["model_config"]), manifest["model_dtype"]
    table, measured = load_table(table_csv), load_measured(data_root)
    policies, ops_tables, att_tables = {}, [], []
    for policy_dir in sorted(p for p in run_dir.iterdir() if (p / "event_000" / "ops.parquet").exists()):
        name = policy_dir.name
        event, control = _frames(policy_dir / "event_000", ops=True), _frames(policy_dir / "control_000")
        steps = event["steps"]
        decode = decode_table(steps, event["ops"], cfg, dtype, machine, threshold, name)
        pre = prefill_table(event["prefill"], event["ops"], cfg, dtype, machine, threshold, name)
        step_us = float(steps.step_us.mean())
        share = float(decode.set_index("op_class").loc["attention", "share"])
        att = attention_steps(steps, cfg, table, measured, machine, params)
        control_step = float(control["steps"].step_us.mean()) if control is not None else math.nan
        tpot = tpot_us(event["tokens"]).dropna()
        policies[name] = {
            "n_steps": int(len(steps)),
            "tpot_waterfall": {"tpot_us_mean": float(tpot.mean()) if len(tpot) else math.nan,
                                "decode_wall_us_mean": float(steps.decode_wall_us.mean())},
            "step_waterfall": {"policy_us_mean": float(steps.policy_us.mean()), "step_us_mean": step_us,
                               "host_residual_us": float((steps.decode_wall_us - steps.step_us - steps.policy_us).mean())},
            "ops": decode.to_dict("records"), "prefill_ops": pre.to_dict("records"),
            "attention": attention_summary(att, share),
            "unattributed_pct": 100 * (step_us - float(decode.gpu_us.sum())) / step_us if step_us > 0 else math.nan,
            "timer_overhead_pct": 100 * (step_us / control_step - 1) if control_step > 0 else math.nan,
        }
        ops_tables += [decode, pre]
        att_tables.append(att.assign(policy=name))
    if not policies:
        raise FileNotFoundError(f"no <policy>/event_000/ops.parquet under {run_dir}")
    summary = {"schema_version": 1,
               "identity": {k: manifest.get(k) for k in ("evidence_kind", "performance_claim", "model", "scenario", "created_at")},
               "ceilings": {"machine": machine.name, "dram_gbps": machine.dram_gbps, "tc_tflops": machine.tc_tflops,
                            "ridge_flop_per_byte": machine.ridge_flop_per_byte(), "threshold": threshold, "launch_us": LAUNCH_US},
               "inputs": {"table": str(table_csv), "data_root": str(data_root), "run_dir": str(run_dir)},
               "policies": policies}
    return Diagnosis(summary, pd.concat(ops_tables, ignore_index=True), pd.concat(att_tables, ignore_index=True))


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def write(run_dir, diagnosis: Diagnosis) -> None:
    run_dir = Path(run_dir)
    (run_dir / "diagnosis.json").write_text(json.dumps(_jsonable(diagnosis.summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    diagnosis.ops.to_csv(run_dir / "ops.csv", index=False)
    diagnosis.attention_steps.to_csv(run_dir / "attention_steps.csv", index=False)
```

- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_report.py` → PASS. 실패하면 판정용 상수(`LAYER_US`)가 아니라 구현을 고친다. `test_decode_table_shares_costs_and_verdicts`의 `pct_dram` 경계가 어긋나면 spec §2.3 공식과 `_op_table`을 대조한다.

- [ ] **Step 5: torch 없이 import 되는지 확인**

Run: `.venv/bin/python -c "import sys; sys.modules['torch']=None; import kernelscope.diagnose.report, kernelscope.diagnose.opmodel; print('ok')"`
Expected: `ok`.

---

### Task 6: `kernelscope/diagnose/figures.py`

**Files:**
- Create: `kernelscope/diagnose/figures.py`
- Test: `tests/test_diagnose_figures.py`

**Interfaces:**
- Consumes: `Diagnosis.summary` 구조 (Task 5), `OP_CLASSES` (Task 4), `kernelscope.dashboard.style.CATEGORICAL`.
- Produces: `op_breakdown(summary: dict, out_png, out_svg=None, title=...) -> Path`.

- [ ] **Step 1: 실패하는 테스트**

```python
import pytest

pytest.importorskip("matplotlib")

from kernelscope.diagnose.figures import op_breakdown
from kernelscope.diagnose.opmodel import OP_CLASSES


def _summary():
    def policy(scale):
        ops = [{"op_class": c, "gpu_us": scale * (i + 1) * 10.0, "share": 0.05 * (i + 1), "verdict": "below_ceiling_unknown"}
               for i, c in enumerate(OP_CLASSES)]
        return {"ops": ops, "step_waterfall": {"step_us_mean": scale * 500.0}}
    return {"policies": {"heuristic": policy(1.0), "table": policy(0.5)}}


def test_renders_png_and_svg(tmp_path):
    png = op_breakdown(_summary(), tmp_path / "b.png", tmp_path / "b.svg")
    assert png.exists() and png.stat().st_size > 1000 and (tmp_path / "b.svg").stat().st_size > 1000


def test_svg_is_optional(tmp_path):
    assert op_breakdown(_summary(), tmp_path / "only.png").exists()
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_figures.py` → ModuleNotFoundError.

- [ ] **Step 3: 구현**

```python
"""Stacked op-class bars from a diagnosis summary. matplotlib/Agg so it renders on any host."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kernelscope.dashboard.style import CATEGORICAL  # noqa: E402
from kernelscope.diagnose.opmodel import OP_CLASSES  # noqa: E402

UNATTRIBUTED = "#9a9a9a"


def op_breakdown(summary: dict, out_png, out_svg=None, title="Decode step time by operation class") -> Path:
    policies = list(summary["policies"])
    colors = CATEGORICAL["light"]
    fig, ax = plt.subplots(figsize=(9, 1.4 + 0.75 * len(policies)))
    for i, name in enumerate(policies):
        p = summary["policies"][name]
        by = {row["op_class"]: row["gpu_us"] / 1e3 for row in p["ops"]}
        left = 0.0
        for j, op_class in enumerate(OP_CLASSES):
            width = by.get(op_class, 0.0)
            ax.barh(i, width, left=left, color=colors[j % len(colors)], label=op_class if i == 0 else None)
            left += width
        rest = max(p["step_waterfall"]["step_us_mean"] / 1e3 - left, 0.0)
        ax.barh(i, rest, left=left, color=UNATTRIBUTED, label="unattributed" if i == 0 else None)
        attention = next((row for row in p["ops"] if row["op_class"] == "attention"), None)
        if attention is not None:
            ax.text(left + rest, i, f"  attention {100 * attention['share']:.0f}% · {attention['verdict']}", va="center", fontsize=8)
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels(policies)
    ax.invert_yaxis()
    ax.set_xlabel("ms per decode step (CUDA events; unattributed = step − Σ classes)")
    ax.set_title(title)
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    if out_svg is not None:
        fig.savefig(out_svg)
    plt.close(fig)
    return out_png
```

- [ ] **Step 4: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_diagnose_figures.py` → PASS.

---

### Task 7: `kernelscope/diagnose/run.py`와 CLI `serve diagnose` / `serve diagnose-report`

**Files:**
- Create: `kernelscope/diagnose/run.py`
- Modify: `kernelscope/serve/cli.py` (`register_parser` 376–409, 핸들러 추가)
- Test: `tests/test_serve_cli.py` (끝에 추가)

**Interfaces:**
- Consumes: `Engine.run(..., ops_mode=)` (Task 3), `diagnose/write` (Task 5), `op_breakdown` (Task 6), `serve.cli`의 헬퍼 `_validate_args, preflight, _load_model, _resolve_scenario, _policy_setup, _request_metadata, _sha256, _json, _git, _source_hashes, _default_out`.
- Produces: `run_diagnose(args) -> Path`, `run_report(args) -> Path`, `project_path(value, default) -> Path`; CLI `serve diagnose` (공용 인자 + `--table`, `--data`, `--threshold`), `serve diagnose-report <results> [--machine --params --table --data --threshold]`.

- [ ] **Step 1: 실패하는 테스트** (`tests/test_serve_cli.py` 끝)

```python
def test_diagnose_cpu_tiny_writes_control_event_and_diagnosis(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "diag"
    args = parse(["serve", "diagnose", "--device", "cpu", "--model", "tiny-random", "--scenario", str(root / "scenarios/tiny.yaml"),
                  "--policy", "heuristic", "--policy", "fa2", "--out", str(out), "--warmup-runs", "0", "--kv-gib", "0.002",
                  "--table", str(root / "demo_data/dispatch_paged_cold.csv"), "--data", str(root / "demo_data")])
    args.func(args)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["tokens_consistent"] is True
    assert manifest["evidence_kind"] == "cpu_functional" and manifest["performance_claim"] is False
    assert manifest["ops_mode_runs"] == ["control", "event"] and manifest["threshold"] == 0.7
    for policy in ("heuristic", "fa2"):
        assert (out / policy / "control_000/steps.parquet").is_file() and (out / policy / "event_000/ops.parquet").is_file()
        assert not (out / policy / "control_000/ops.parquet").exists()
        assert json.loads((out / policy / "event_000/meta.json").read_text())["ops_mode"] == "event"
    diagnosis = json.loads((out / "diagnosis.json").read_text())
    assert set(diagnosis["policies"]) == {"heuristic", "fa2"}
    heuristic = diagnosis["policies"]["heuristic"]
    assert len(heuristic["ops"]) == 8 and len(heuristic["prefill_ops"]) == 8
    assert heuristic["attention"]["source"] == "unavailable"          # tiny head shape (4, 2) is not in the measured table
    assert heuristic["unattributed_pct"] is not None
    ops = pd.read_csv(out / "ops.csv")
    assert len(ops) == 2 * 16 and (out / "op_breakdown.png").stat().st_size > 1000
    assert pd.read_csv(out / "consistency.csv").passed.all()
    with pytest.raises(SystemExit, match="already exists"):
        args.func(args)


def test_diagnose_report_regenerates_from_a_recorded_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "diag"
    run = parse(["serve", "diagnose", "--device", "cpu", "--model", "tiny-random", "--scenario", str(root / "scenarios/tiny.yaml"),
                 "--policy", "heuristic", "--out", str(out), "--warmup-runs", "0", "--kv-gib", "0.002"])
    run.func(run)
    (out / "diagnosis.json").unlink()
    report = parse(["serve", "diagnose-report", str(out), "--threshold", "0.5"])
    report.func(report)
    assert json.loads((out / "diagnosis.json").read_text())["ceilings"]["threshold"] == 0.5


def test_diagnose_report_refuses_a_folder_without_manifest(tmp_path):
    args = parse(["serve", "diagnose-report", str(tmp_path)])
    with pytest.raises(SystemExit, match="manifest.json"):
        args.func(args)
```

- [ ] **Step 2: 실패 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_cli.py -k diagnose` → `argparse` error `invalid choice: 'diagnose'` (SystemExit 2).

- [ ] **Step 3: `kernelscope/diagnose/run.py` 작성**

```python
"""`serve diagnose`: one control run and one op-timed run per policy, then the ceiling report.

Provenance (preflight, prompt resolution, manifest, source hashes) is reused from kernelscope.serve.cli.
Timings recorded here are diagnostic evidence, never a performance claim.
"""
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]


def project_path(value, default) -> Path:
    """A CLI path as given when it exists, else the same path relative to the repository root."""
    candidate = Path(value) if value else PROJECT / default
    return candidate if candidate.exists() else PROJECT / candidate


def _report(out, args):
    from kernelscope.diagnose.figures import op_breakdown
    from kernelscope.diagnose.report import diagnose, write
    from kernelscope.model.machine import MachineSpec
    machine = MachineSpec.from_json(project_path(getattr(args, "machine", None), "machines/rtx4090.json"))
    params = None
    if getattr(args, "params", None):
        from kernelscope.model.params import ModelParams
        params = ModelParams.from_json(args.params)
    result = diagnose(out, machine, project_path(args.table, "demo_data/dispatch_paged_cold.csv"),
                      project_path(args.data, "demo_data"), params=params, threshold=args.threshold)
    write(out, result)
    op_breakdown(result.summary, out / "op_breakdown.png", out / "op_breakdown.svg")
    return result


def run_report(args) -> Path:
    out = Path(args.results)
    if not (out / "manifest.json").exists():
        raise FileNotFoundError(f"{out} is not a serve diagnose run: manifest.json is missing")
    result = _report(out, args)
    print(result.ops.to_string(index=False))
    return out


def run_diagnose(args) -> Path:
    import pandas as pd
    import torch
    from kernelscope.serve import cli
    from kernelscope.serve.engine import Engine
    from kernelscope.serve.equivalence import compare_results
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.scenarios import load_scenario

    args.repeats = 1
    cli._validate_args(args)
    if not 0 < args.threshold <= 1:
        raise ValueError("--threshold must be in (0, 1]")
    requests, dataset = load_scenario(args.scenario)
    args.policy = args.policy or ["heuristic"]
    policies, policy_setup = cli._policy_setup(args, "initial")
    cuda = args.device == "cuda"
    status = cli.preflight(args.model, require_cuda=cuda)
    if not status["ready"]:
        raise RuntimeError("Diagnose preflight failed:\n- " + "\n- ".join(status["errors"]))
    out = Path(args.out) if args.out else cli._default_out("diagnose")
    if out.exists():
        raise FileExistsError(f"output already exists: {out}; choose a new directory to preserve prior evidence")
    if not cuda:
        torch.set_num_threads(args.cpu_threads)
    model = cli._load_model(args)
    requests, prompt_metadata = cli._resolve_scenario(args, requests, dataset, model.cfg.vocab)
    kv_bytes = int(args.kv_gib * 2**30)
    if cuda and torch.cuda.mem_get_info()[0] < kv_bytes + 256 * 2**20:
        raise MemoryError("not enough free GPU memory for --kv-gib plus working tensors")
    table, data_root = project_path(args.table, "demo_data/dispatch_paged_cold.csv"), project_path(args.data, "demo_data")
    manifest = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
                "evidence_kind": "diagnostic_op_breakdown" if cuda else "cpu_functional", "performance_claim": False,
                "model": args.model, "model_backend": model.backend, "model_config": asdict(model.cfg),
                "model_dtype": str(model.dtype).removeprefix("torch."),
                "scenario": str(Path(args.scenario).resolve()), "scenario_sha256": cli._sha256(args.scenario),
                "requests": [cli._request_metadata(r) for r in requests], "policy_specs": args.policy,
                "seed": args.seed, "max_batch": args.max_batch, "kv_bytes": kv_bytes,
                "warmup_runs": args.warmup_runs, "warmup_steps": args.warmup_steps,
                "ops_mode_runs": ["control", "event"], "threshold": args.threshold,
                "table": str(table), "data_root": str(data_root), "policy_setup_runs": [policy_setup], **prompt_metadata,
                "sampling": "greedy_fixed_length_no_eos_stop", "arrival_mode": "logical_decode_step",
                "timing_instrumentation": "per-layer op-class CUDA events (event run); attention-only events (control run)",
                "limitations": ["Op timers add host work; step times of this run are diagnostic, not performance evidence",
                                "The byte model counts compulsory traffic only, so achieved bandwidth is a lower bound",
                                "Kernel-internal behaviour (stalls, bank conflicts) is outside this report; that is Nsight Compute's domain"],
                "preflight": status, "git": cli._git(), "source_sha256": cli._source_hashes()}
    if not cuda:
        manifest["limitations"].append("CPU SDPA ignores num_splits; CPU op timings are functional only")
    out.mkdir(parents=True, exist_ok=False)
    (out / "scenario.yaml").write_bytes(Path(args.scenario).read_bytes())
    with (out / "prompts.jsonl").open("x") as handle:
        for request in requests:
            handle.write(json.dumps({**cli._request_metadata(request), "token_ids": list(request.token_ids)}, ensure_ascii=False) + "\n")
    cli._json(out / "manifest.json", manifest)

    def run(policy, scenario, ops_mode):
        pool = PagePool.for_budget(model.cfg, kv_bytes, device=model.device, dtype=model.dtype)
        try:
            return Engine(model, pool, policy, args.max_batch).run(scenario, model.cfg.vocab, seed=args.seed, ops_mode=ops_mode)
        finally:
            del pool

    consistency = []
    try:
        warm_requests = requests
        if args.warmup_steps is not None:
            warm_requests = [replace(r, max_new_tokens=min(r.max_new_tokens, args.warmup_steps + 1)) for r in requests]
        for policy in policies:
            for _ in range(args.warmup_runs):
                print(f"Warmup {policy.name}", flush=True)
                run(policy, warm_requests, None)
            results = {}
            for label, mode in (("control", None), ("event", "event")):
                if cuda:
                    check = cli.preflight(require_cuda=True)
                    if not check["ready"]:
                        raise RuntimeError("GPU contention/preflight changed: " + "; ".join(check["errors"]))
                print(f"{label}: {policy.name}", flush=True)
                result = run(policy, requests, mode)
                directory = out / policy.name / f"{label}_000"
                directory.mkdir(parents=True, exist_ok=False)
                for name in ("steps", "tokens", "prefill"):
                    getattr(result, name).to_parquet(directory / f"{name}.parquet", index=False)
                if mode == "event":
                    result.ops.to_parquet(directory / "ops.parquet", index=False)
                cli._json(directory / "meta.json", {**result.metadata, "policy": policy.name, "run": label,
                                                    "evidence_kind": manifest["evidence_kind"], "performance_claim": False,
                                                    "created_at": datetime.now(timezone.utc).isoformat()})
                results[label] = result
            consistency.append({**compare_results(results["control"], results["event"], "control", "event"), "policy": policy.name})
        frame = pd.DataFrame(consistency)
        frame.to_csv(out / "consistency.csv", index=False)
        manifest.update(status="complete", tokens_consistent=bool(frame.passed.all()),
                        completed_at=datetime.now(timezone.utc).isoformat())
        cli._json(out / "manifest.json", manifest)
        result = _report(out, args)
        print(result.ops.to_string(index=False))
        print(f"Artifacts: {out.resolve()}")
        if not manifest["tokens_consistent"]:
            raise RuntimeError("control and op-timed runs generated different tokens; see consistency.csv")
        return out
    except BaseException as error:
        if manifest["status"] != "complete":
            manifest.update(status="failed", error=str(error))
            cli._json(out / "manifest.json", manifest)
        raise
```

- [ ] **Step 4: `serve/cli.py`에 핸들러와 파서 추가**

`_compare` 앞에 핸들러:
```python
def _diagnose(args):
    from kernelscope.diagnose.run import run_diagnose
    try:
        return run_diagnose(args)
    except (ValueError, RuntimeError, OSError, MemoryError) as error:
        raise SystemExit(str(error)) from None


def _diagnose_report(args):
    from kernelscope.diagnose.run import run_report
    try:
        return run_report(args)
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
```

`register_parser`의 루프를 `parsers = {}`로 파서를 보관하도록 바꾸고 `("diagnose", _diagnose)`를 튜플에 추가한다:
```python
    parsers = {}
    for name, handler in (("run", _run), ("demo", _demo), ("equivalence", _equivalence), ("diagnose", _diagnose)):
        parser = parsers[name] = sub.add_parser(name)
        ...(기존 인자 그대로)...
        parser.set_defaults(func=handler)
    diag = parsers["diagnose"]
    diag.add_argument("--table", default="demo_data/dispatch_paged_cold.csv", help="measured dispatch table for the attention row")
    diag.add_argument("--data", default="demo_data", help="results root holding hw_4090/*_paged/summaries.jsonl")
    diag.add_argument("--threshold", type=float, default=0.7, help="ceiling fraction that counts as bound")
    report = sub.add_parser("diagnose-report", help="recompute diagnosis.json / ops.csv / figure from a recorded diagnose run (no GPU)")
    report.add_argument("results")
    report.add_argument("--machine", default="machines/rtx4090.json")
    report.add_argument("--params")
    report.add_argument("--table", default="demo_data/dispatch_paged_cold.csv")
    report.add_argument("--data", default="demo_data")
    report.add_argument("--threshold", type=float, default=0.7)
    report.set_defaults(func=_diagnose_report)
```
(`diagnose`는 루프에서 `--repeats/--warmup-runs/--warmup-steps`를 이미 받는다. `run_diagnose`는 `repeats`를 1로 덮어쓴다.)

- [ ] **Step 5: 통과 확인** — Run: `.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_serve_cli.py` → PASS (기존 포함).

- [ ] **Step 6: 전체 CPU 스위트** — Run: `make test` → 기존 485개 + 신규 전부 PASS, 실패 0.

---

### Task 8: GPU 실행 D1·D2·D3, 번들 복사, 합격 기준 A1~A6·A8 (메인 세션)

**Files:**
- Create (리포 밖): `../kernelscope/results/serve_4090/diagnose_20260927/{ragged,uniform,heldout_ragged}/`
- Create (리포 안, `package_demo`가 복사): `demo_data/serve_4090/diagnose_20260927/**`
- Create (스크래치): `$SCR/a8/after/`, `$SCR/a8/after.json`

- [ ] **Step 1: 유휴 확인** — `serve doctor` → `"ready": true`.

- [ ] **Step 2: D1 혼합 길이**

```bash
cd /home/skkai/AI_Accelerator/kernelscope-design && export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
R=../kernelscope/results/serve_4090/diagnose_20260927
.venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario scenarios/graduation_ragged.yaml --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
  --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --out $R/ragged
```

- [ ] **Step 3: D2 균일** — 같은 명령에 `--scenario scenarios/graduation_uniform.yaml --out $R/uniform`.

- [ ] **Step 4: D3 자연어 혼합**

```bash
.venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 \
  --scenario scenarios/heldout_text_ragged.yaml --policy heuristic --policy model \
  --machine machines/rtx4090.json --params models/rtx4090.json \
  --kv-gib 10 --warmup-runs 1 --warmup-steps 2 --out $R/heldout_ragged
```

- [ ] **Step 5: 합격 기준 A1~A6 출력**

```bash
.venv/bin/python - <<'PY'
import json, pandas as pd
R="../kernelscope/results/serve_4090/diagnose_20260927"
table = pd.read_csv("demo_data/dispatch_paged_cold.csv").set_index("workload_key")
for scen in ("ragged", "uniform", "heldout_ragged"):
    d = json.load(open(f"{R}/{scen}/diagnosis.json")); m = json.load(open(f"{R}/{scen}/manifest.json"))
    print(f"== {scen}  tokens_consistent(A6)={m['tokens_consistent']}")
    for name, p in d["policies"].items():
        a = p["attention"]; ops = {o["op_class"]: o for o in p["ops"]}
        print(f"  {name:10s} unattributed(A1)={p['unattributed_pct']:.2f}%  overhead(A2)={p['timer_overhead_pct']:.2f}%  "
              f"attn share={a['share']:.3f} chosen={a['chosen_variant']} best={a['best_alternative']} regret={a['regret']} "
              f"source={a['source']} amdahl(A5)={a['amdahl_bound']}")
        print("     verdicts(A4):", {k: (round(v['pct_dram'] or 0, 1), v['verdict']) for k, v in ops.items()})
    if scen == "ragged":
        key = d["policies"]["heuristic"]["attention"]["neighbor_key"]
        print("  A3 table row:", key, table.loc[key, ["best_kernel", "heuristic_regret"]].to_dict())
        h = pd.read_parquet(f"{R}/ragged/heuristic/control_000/steps.parquet").step_us.mean()
        t = pd.read_parquet(f"{R}/ragged/table/control_000/steps.parquet").step_us.mean()
        print(f"  A5 measured control step ratio heuristic/table = {h/t:.3f}")
PY
```
Expected: A1 ≤ 10, A2 ≤ 5 (초과 시 STATUS에 원인 기록, 실패 아님), A3 `best_alternative == best_kernel`, `regret ≈ heuristic_regret`(D1 heuristic), A4 attention만 `parallelism_candidate`, A5 `amdahl_bound` ≥ 측정 비율과 같은 방향, A6 True. 예상과 다른 값은 그대로 기록한다.

- [ ] **Step 6: A8 변경 후 측정** — Task 0의 명령을 `--out $SCR/a8/after`로 다시 실행하고 같은 스니펫으로 `after.json` 생성. `after.step_us_median / before.step_us_median − 1`이 ±2% 안인지 출력.

- [ ] **Step 7: 번들 복사** — `.venv/bin/python scripts/package_demo.py --results ../kernelscope/results` 실행 뒤 `git status --short demo_data | head`로 새 파일이 `demo_data/serve_4090/diagnose_20260927/` 아래에만 생기고 기존 파일은 바뀌지 않았는지 확인(`git diff --stat demo_data`가 비어 있어야 함).

- [ ] **Step 8: 그림 복사** — `cp ../kernelscope/results/serve_4090/diagnose_20260927/ragged/op_breakdown.png docs/img/op_breakdown_ragged.png`.

---

### Task 9: verify 항목, 실험 노트, PLAN·STATUS·README

**Files:**
- Modify: `kernelscope/verify.py` (헬퍼 + `CHECKS` 끝에 `diagnose.*` 7개)
- Create: `docs/experiments/2026-09-27-op-breakdown.md`
- Modify: `PLAN.md` (§1 ② 근거), `docs/STATUS.md` (최상단 항목), `README.md` (`serve demo` 블록 다음)
- Test: `tests/test_verify.py` (기존 패턴에 맞춰 1개 추가), `tests/test_cli.py`(변경 없음)

- [ ] **Step 1: 수치 뽑기** — Task 8 Step 5의 출력에서 D1 heuristic·table attention share, D1 heuristic unattributed·mlp pct_dram·amdahl_bound, D2 heuristic attention share, 세 실행의 timer_overhead 최댓값을 적는다.

- [ ] **Step 2: 실패하는 테스트** (`tests/test_verify.py` 끝; 기존 테스트가 `run_checks`를 어떻게 부르는지 위쪽 함수를 따라간다)

```python
def test_diagnose_checks_recompute_from_the_bundle():
    from kernelscope import verify
    repo = Path(__file__).resolve().parents[1]
    checks = [c for c in verify.CHECKS if c.id.startswith("diagnose.")]
    assert len(checks) == 7
    t = verify.run_checks(checks, repo=repo, data=repo / "demo_data")
    assert set(t.status) == {"PASS"}, t.to_string()
```

- [ ] **Step 3: verify 헬퍼와 항목** (`verify.py`, `_sim_ratio_s1` 뒤)

```python
DIAG = "diagnose_20260927"
OPB = "docs/experiments/2026-09-27-op-breakdown.md"


@lru_cache(maxsize=None)
def _diagnosis(repo: Path, data: Path, scenario: str) -> dict:
    """Recompute the op-class report from the recorded parquet files (never read diagnosis.json)."""
    from kernelscope.diagnose.report import diagnose
    from kernelscope.model.machine import MachineSpec
    return diagnose(data / "serve_4090" / DIAG / scenario, MachineSpec.from_json(repo / "machines" / "rtx4090.json"),
                    repo / "demo_data" / "dispatch_paged_cold.csv", data).summary


def _diag_value(scenario, policy, *path):
    def f(repo, data):
        node = _diagnosis(repo, data, scenario)["policies"][policy]
        for key in path:
            node = node[key]
        return float(node)
    return f


def _diag_op(scenario, policy, op_class, column, scale=1.0):
    def f(repo, data):
        ops = _diagnosis(repo, data, scenario)["policies"][policy]["ops"]
        return scale * float(next(r for r in ops if r["op_class"] == op_class)[column])
    return f


def _diag_overhead_max(repo, data):
    return max(p["timer_overhead_pct"] for s in ("ragged", "uniform", "heldout_ragged")
               for p in _diagnosis(repo, data, s)["policies"].values())
```

`CHECKS` 끝에 (`<...>`는 Step 1의 값과 노트에 적은 문구로 채운다; `tol`은 share 0.5%p, pct_dram 1.0%p, amdahl 0.01):
```python
    Check("diagnose.ragged_attention_share_heuristic", "혼합 길이 연산 분해(D1): 휴리스틱 step의 attention 비중",
          <share_h_pct>, 0.5, "%", _diag_op("ragged", "heuristic", "attention", "share", 100), OPB, "<share_h_pct 표기>"),
    Check("diagnose.ragged_attention_share_table", "혼합 길이 연산 분해(D1): 측정 테이블 정책 step의 attention 비중",
          <share_t_pct>, 0.5, "%", _diag_op("ragged", "table", "attention", "share", 100), OPB, "<share_t_pct 표기>"),
    Check("diagnose.ragged_unattributed_heuristic", "혼합 길이 연산 분해(D1): 미귀속 시간 비율(상한 10%, 5±5)",
          5.0, 5.0, "%", _diag_value("ragged", "heuristic", "unattributed_pct")),
    Check("diagnose.ragged_mlp_pct_dram_heuristic", "혼합 길이 연산 분해(D1): mlp 클래스의 DRAM 상한 대비 달성률(하한 추정)",
          <mlp_pct>, 1.0, "%", _diag_op("ragged", "heuristic", "mlp", "pct_dram"), OPB, "<mlp_pct 표기>"),
    Check("diagnose.ragged_amdahl_bound_heuristic", "혼합 길이 연산 분해(D1): attention만 최적으로 바꿀 때 step 상한 배율",
          <amdahl>, 0.01, "x", _diag_value("ragged", "heuristic", "attention", "amdahl_bound"), OPB, "<amdahl 표기>"),
    Check("diagnose.uniform_attention_share_heuristic", "균일 길이 연산 분해(D2): 휴리스틱 step의 attention 비중",
          <share_u_pct>, 0.5, "%", _diag_op("uniform", "heuristic", "attention", "share", 100), OPB, "<share_u_pct 표기>"),
    Check("diagnose.timer_overhead_max", "연산 분해 타이머 오버헤드 최댓값(세 실행, 상한 5%, 0±5)",
          0.0, 5.0, "%", _diag_overhead_max),
```

- [ ] **Step 4: 실험 노트** `docs/experiments/2026-09-27-op-breakdown.md` — 구성: 한 문단 결론(굵게), 방법(event/control 두 실행, 8클래스, 바이트 모델은 하한, 임계 0.7, GPU 없이 `serve diagnose-report`로 재생성), 표 1 D1 heuristic vs table의 클래스별 gpu_us(ms)·share·pct_dram·verdict(ops.csv에서), 표 2 D2·D3 요약(attention share, unattributed, overhead), attention 행(chosen/best/regret/amdahl vs control step 비율), 합격 기준 A1~A6·A8 결과(초과 시 그대로), 한계(하한 추정·미귀속·커널 내부는 ncu 영역·CPU 타이머 불가), 그림 `../img/op_breakdown_ragged.png`, 재현 명령. verify가 찾는 문구(`doc_text`)가 그대로 들어가야 한다.

- [ ] **Step 5: PLAN·STATUS·README**
  - `PLAN.md` §1 ② 행 근거 끝에: `; step 연산 분해(\`serve diagnose\`, 2026-09-27): 혼합 길이 attention 비중 <h>%→<t>%, 나머지 GEMM 클래스는 DRAM 상한 <x>~<y>%로 커널 선택의 여지 없음`.
  - `docs/STATUS.md` 최상단에 `## 2026-09-27 — Op-class breakdown of the decode step (branch design-1-3)` 항목: 무엇을 만들었고(OpTimer, diagnose 패키지, CLI), D1~D3 핵심 수치, A1~A8 결과, verify 47개, 한계.
  - `README.md` `serve demo` 예시 뒤에 블록:
    ```bash
    # 연산 클래스 분해 진단(대조 실행 + 계측 실행, 성능 주장 아님)
    .venv/bin/python -m kernelscope.cli serve diagnose --model Qwen/Qwen3-4B-Instruct-2507 \
      --scenario scenarios/graduation_ragged.yaml --policy heuristic --policy table:demo_data/dispatch_paged_cold.csv \
      --kv-gib 10 --out ../kernelscope/results/serve_4090/diagnose_new/ragged
    .venv/bin/python -m kernelscope.cli serve diagnose-report demo_data/serve_4090/diagnose_20260927/ragged   # GPU 없이 재생성
    ```
    와 한 문장 설명(8클래스, 상한 판정, 하한 추정, `docs/experiments/2026-09-27-op-breakdown.md` 링크).

- [ ] **Step 6: 검증** — Run: `make verify` → `{"PASS": 47}`; `make test` → 전부 PASS; `grep -rn "더 나은 프로파일러\|better profiler" README.md docs/graduation.md docs/STATUS.md docs/experiments/2026-09-27-op-breakdown.md` → 0건.

- [ ] **Step 7: 사용자에게 보고** — 변경 파일 목록, A1~A8 결과, 커밋 여부 질문(CLAUDE.md 규칙).
