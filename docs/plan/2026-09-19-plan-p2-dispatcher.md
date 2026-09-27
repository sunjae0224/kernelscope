# Output-Preserving Kernel Dispatcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A minimal continuous-batching decode engine for a real LLM on the RTX 4090 that, at every decode step, chooses the flash-attn `num_splits` for the paged-KV attention call — fixed, the library heuristic, a measured-table lookup, or the surrogate model's argmin — and measures what the choice does to attention time, step time and per-request TPOT, while showing the outputs stay the same.

**Architecture:** `kernelscope/serve/`: `hf.py` (local Hugging Face snapshot → config, RoPE frequencies, bf16 weights), `kvcache.py` (paged KV pool, 256-token pages, shared block tables across layers), `model.py` (Llama / Qwen3 decoder forward in plain PyTorch with `flash_attn_with_kvcache` for attention, RoPE applied in PyTorch), `dispatch.py` (policies), `engine.py` (deterministic continuous batching with per-step CUDA-event timing), `scenarios.py` (request traces), `equivalence.py` (output comparison across policies). CLI `kernelscope serve {run,compare,equivalence}`.

**Tech Stack:** torch 2.8.0+cu128, flash-attn 2.8.3.post1, safetensors, tokenizers, transformers (test oracle only), numpy, pandas.

**Spec:** `docs/plan/2026-09-19-design-surrogate-dispatcher.md` §5 (and §1 facts F14–F19). The case for the dispatcher: under cold L2, uniform batches leave the heuristic within 16 % of the best split, but ragged batches (long + short sequences) cost up to 12.5× on the dense path and 3.8× on the paged path (campaign note `docs/plan/2026-09-19-p0-campaign.md`). The engine uses the paged path (a dense cache cannot hold ragged batches in 24 GB).

## Global Constraints

- **Worktree:** `/home/skkai/AI_Accelerator/kernelscope-design`, branch `design-1-3`. Never edit `/home/skkai/AI_Accelerator/kernelscope`. Experiment output goes to `/home/skkai/AI_Accelerator/kernelscope/results/serve_4090/` (git-ignored, shared).
- **Python:** `PY=/home/skkai/miniforge3/envs/gradkernel/bin/python`, absolute path. Whole suite: `$PY -m pytest -q -p no:cacheprovider` (GPU tests first via tests/conftest.py).
- **Installs:** only in Task 1, only `transformers==4.57.6`, `tokenizers`, `safetensors`, and only if `pip install --dry-run` shows that torch, numpy, triton and flash-attn stay at their current versions. Otherwise stop and report.
- **Models** (local HF cache, never download others): default `Qwen/Qwen3-4B-Instruct-2507` (36 layers, hidden 2560, H_q 32, H_kv 8, head_dim 128, rope θ 5e6, tied embeddings, q/k RMSNorm); secondary `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (32 layers, hidden 4096, H_q 32, H_kv 8, llama3 RoPE scaling factor 8). Both have the attention shape of the measured grid S1 (32/8/128).
- **Numerics:** weights and activations bf16; RMSNorm and RoPE tables computed in float32 and cast like Hugging Face does; softmax scale `head_dim ** -0.5`.
- **Paged KV:** page 256 tokens (flash-attn requires a multiple of 256); one pool tensor pair per layer `[n_pages, 256, H_kv, 128]`; the same page ids and block table serve every layer.
- **GPU hygiene / contention:** the user runs their own GPU jobs (e.g. `.venv/bin/python -m core.meaning_segmentator...`). Never stop them. Before any GPU test or measurement check `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`; only `rerun` (~440 MiB) is acceptable for timing measurements (Task 8). For functional GPU tests, a foreign process is acceptable if at least 18 GiB are free (`torch.cuda.mem_get_info`); otherwise run `-m "not gpu"` and say so. Model-loading tests skip themselves when free memory is insufficient.
- **Codex-owned, never edit:** `kernelscope/backends/accelsim/**`, `_cmd_simsweep` / `p_sim` in `kernelscope/cli.py`, `tests/test_accelsim_*.py`, `tests/fixtures/SM*`, `docs/setup/**`.
- **Commits:** `git add` new files, then `git commit -m "<msg>" -- <paths>`; end with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`. Run your task's tests, then the whole suite; report failures in files you did not touch.

## Execution order

Sequential: Task 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8.

---

### Task 1: Dependencies and the Hugging Face snapshot loader

**Files:**
- Create: `env/requirements-serve.txt`, `kernelscope/serve/__init__.py`, `kernelscope/serve/hf.py`
- Test: `tests/test_serve_hf.py`

**Interfaces:**
- Produces:
  - `hf.snapshot_dir(repo_id: str, cache=None) -> Path` (`~/.cache/huggingface/hub/models--{org}--{name}/snapshots/<newest>/`; `FileNotFoundError` if absent)
  - `hf.ModelConfig` (frozen dataclass): `arch: str` (`"llama"` | `"qwen3"`), `n_layers, hidden, intermediate, n_heads, n_kv_heads, head_dim, vocab: int`, `rope_theta: float`, `rope_scaling: dict | None`, `rms_eps: float`, `tie_embeddings: bool`, `qk_norm: bool`; `ModelConfig.from_dir(path)`
  - `hf.rope_inv_freq(cfg) -> torch.Tensor` (float32 `[head_dim // 2]`, llama3 scaling applied when `rope_scaling["rope_type"] == "llama3"`)
  - `hf.load_weights(path, device="cuda", dtype=torch.bfloat16) -> dict[str, torch.Tensor]`
  - `hf.load_tokenizer(path)` → `tokenizers.Tokenizer`

- [ ] **Step 1: Install** (check first):
```bash
$PY -m pip install --dry-run "transformers==4.57.6" tokenizers safetensors 2>&1 | tail -5
```
Confirm the "Would install" line contains none of torch, numpy, triton, flash-attn / flash_attn. Then install the same set without `--dry-run`, and write `env/requirements-serve.txt` from `$PY -m pip freeze | grep -iE "^(transformers|tokenizers|safetensors|huggingface.hub)=="`.

- [ ] **Step 2: Failing tests** — `tests/test_serve_hf.py`:
```python
import math

import pytest
import torch

from kernelscope.serve.hf import ModelConfig, rope_inv_freq, snapshot_dir

QWEN = "Qwen/Qwen3-4B-Instruct-2507"
LLAMA = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


def test_snapshot_dir_finds_local_models():
    assert (snapshot_dir(QWEN) / "config.json").exists()
    with pytest.raises(FileNotFoundError):
        snapshot_dir("nobody/nothing")


def test_qwen3_config():
    c = ModelConfig.from_dir(snapshot_dir(QWEN))
    assert (c.arch, c.n_layers, c.hidden, c.n_heads, c.n_kv_heads, c.head_dim) == ("qwen3", 36, 2560, 32, 8, 128)
    assert c.qk_norm and c.tie_embeddings and c.rope_scaling is None and c.rope_theta == 5e6


def test_llama_config_derives_head_dim():
    c = ModelConfig.from_dir(snapshot_dir(LLAMA))
    assert (c.arch, c.n_layers, c.hidden, c.head_dim) == ("llama", 32, 4096, 128)
    assert not c.qk_norm and not c.tie_embeddings and c.rope_scaling["rope_type"] == "llama3"


def test_plain_rope_frequencies():
    c = ModelConfig.from_dir(snapshot_dir(QWEN))
    f = rope_inv_freq(c)
    assert f.shape == (64,) and f.dtype == torch.float32
    assert f[1].item() == pytest.approx(5e6 ** (-2 / 128))


def test_llama3_rope_frequencies_match_transformers():
    from transformers import AutoConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    path = snapshot_dir(LLAMA)
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](AutoConfig.from_pretrained(path), "cpu")
    assert torch.allclose(rope_inv_freq(ModelConfig.from_dir(path)), ref.float(), rtol=1e-6)
```

- [ ] **Step 3: Implement** — `kernelscope/serve/__init__.py`: `"""Minimal decode engine and runtime kernel dispatcher (spec §5)."""`. `kernelscope/serve/hf.py`:
```python
"""Local Hugging Face snapshots: config, RoPE frequencies, weights, tokenizer. No network access."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

HUB = Path.home() / ".cache" / "huggingface" / "hub"
_ARCH = {"LlamaForCausalLM": "llama", "Qwen3ForCausalLM": "qwen3"}


def snapshot_dir(repo_id: str, cache=None) -> Path:
    root = Path(cache or HUB) / ("models--" + repo_id.replace("/", "--")) / "snapshots"
    snaps = sorted(root.glob("*/"), key=lambda p: p.stat().st_mtime) if root.exists() else []
    if not snaps:
        raise FileNotFoundError(f"no local snapshot of {repo_id} under {root}")
    return snaps[-1]


@dataclass(frozen=True)
class ModelConfig:
    arch: str
    n_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    rope_theta: float
    rope_scaling: dict | None
    rms_eps: float
    tie_embeddings: bool
    qk_norm: bool

    @classmethod
    def from_dir(cls, path) -> "ModelConfig":
        c = json.loads((Path(path) / "config.json").read_text())
        arch = _ARCH.get(c["architectures"][0])
        if arch is None:
            raise ValueError(f"unsupported architecture {c['architectures'][0]}")
        return cls(arch=arch, n_layers=c["num_hidden_layers"], hidden=c["hidden_size"],
                   intermediate=c["intermediate_size"], n_heads=c["num_attention_heads"],
                   n_kv_heads=c["num_key_value_heads"],
                   head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                   vocab=c["vocab_size"], rope_theta=float(c["rope_theta"]), rope_scaling=c.get("rope_scaling"),
                   rms_eps=float(c["rms_norm_eps"]), tie_embeddings=bool(c.get("tie_word_embeddings", False)),
                   qk_norm=arch == "qwen3")


def rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.int64).float() / d))
    s = cfg.rope_scaling
    if not s or s.get("rope_type", s.get("type")) != "llama3":
        return inv
    factor, lo, hi, old = s["factor"], s["low_freq_factor"], s["high_freq_factor"], s["original_max_position_embeddings"]
    lo_wl, hi_wl = old / lo, old / hi
    wavelen = 2 * math.pi / inv
    scaled = torch.where(wavelen > lo_wl, inv / factor, inv)
    smooth = (old / wavelen - lo) / (hi - lo)
    smoothed = (1 - smooth) * scaled / factor + smooth * scaled
    medium = ~(wavelen < hi_wl) * ~(wavelen > lo_wl)
    return torch.where(medium, smoothed, scaled)


def load_weights(path, device="cuda", dtype=torch.bfloat16) -> dict:
    from safetensors.torch import load_file
    out = {}
    for f in sorted(Path(path).glob("*.safetensors")):
        for k, v in load_file(str(f), device=str(device)).items():
            out[k] = v.to(dtype)
    return out


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
```
- [ ] **Step 4: Run** → PASS (CPU only). If `ROPE_INIT_FUNCTIONS["llama3"]` has a different signature in the installed transformers, adapt only the test's call (not `rope_inv_freq`) and note it.
- [ ] **Step 5: Whole suite, commit** — `git add env/requirements-serve.txt kernelscope/serve/__init__.py kernelscope/serve/hf.py tests/test_serve_hf.py && git commit -m "Add serving dependencies and the local HF snapshot loader" -- env/requirements-serve.txt kernelscope/serve/__init__.py kernelscope/serve/hf.py tests/test_serve_hf.py`

---

### Task 2: Paged KV pool

**Files:**
- Create: `kernelscope/serve/kvcache.py`
- Test: `tests/test_serve_kvcache.py`

**Interfaces:**
- Produces: `PAGE = 256`; `PagePool(n_layers, n_pages, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda")` with attributes `k: list[Tensor]`, `v: list[Tensor]` (each `[n_pages, PAGE, H_kv, d]`), methods `reserve(seq_id, total_len)` (ensure pages for `total_len` tokens; `MemoryError` when the pool is exhausted), `release(seq_id)`, `length(seq_id) -> int`, `set_length(seq_id, n)`, `block_table(seq_ids) -> Tensor[int32, B × max_pages]` (on the pool's device, rows padded with 0), `lengths(seq_ids) -> Tensor[int32]`, `free_pages -> int`; `PagePool.bytes_per_page(n_layers, n_kv_heads, head_dim, dtype) -> int`; `PagePool.for_budget(cfg, budget_bytes, device, dtype)`.

- [ ] **Step 1: Failing tests** — `tests/test_serve_kvcache.py`:
```python
import pytest
import torch

from kernelscope.serve.kvcache import PAGE, PagePool


def _pool(n_pages=8):
    return PagePool(n_layers=2, n_pages=n_pages, n_kv_heads=2, head_dim=8, dtype=torch.float32, device="cpu")


def test_reserve_grows_by_whole_pages_and_tracks_length():
    p = _pool()
    p.reserve("a", 300)
    assert p.free_pages == 6
    p.set_length("a", 300)
    p.reserve("a", 512)
    assert p.free_pages == 6
    p.reserve("a", 513)
    assert p.free_pages == 5 and p.length("a") == 300


def test_release_returns_pages():
    p = _pool()
    p.reserve("a", 1000)
    p.release("a")
    assert p.free_pages == 8


def test_block_table_pads_rows_and_keeps_order():
    p = _pool()
    p.reserve("a", 600); p.reserve("b", 10)
    bt = p.block_table(["b", "a"])
    assert bt.dtype == torch.int32 and bt.shape == (2, 3)
    assert bt[0, 1:].tolist() == [0, 0]
    assert len(set(bt[1].tolist())) == 3


def test_exhaustion_raises_memory_error():
    p = _pool(n_pages=2)
    with pytest.raises(MemoryError):
        p.reserve("a", 3 * PAGE)


def test_bytes_per_page_counts_k_and_v_in_every_layer():
    assert PagePool.bytes_per_page(36, 8, 128, torch.bfloat16) == 36 * 2 * 256 * 8 * 128 * 2
```
GPU test (append; `@pytest.mark.gpu`): allocate a pool on CUDA (2 layers, H_kv 8, d 128, bf16, 16 pages), reserve two sequences of lengths 300 and 40, call `flash_attn_with_kvcache(q, pool.k[0], pool.v[0], k=k_new, v=v_new, cache_seqlens=lengths_before, block_table=bt, causal=True)` with `q, k_new, v_new` of shape `[2, 1, H, d]` / `[2, 1, H_kv, d]`, and assert that `pool.k[0][bt[0, 300 // 256], 300 % 256]` equals `k_new[0, 0]` (the append landed where the table says).

- [ ] **Step 2: Implement `kernelscope/serve/kvcache.py`:**
```python
"""Paged KV cache for flash_attn_with_kvcache: 256-token pages, one pool tensor pair per layer,
one block table shared by all layers (a sequence owns the same page ids in every layer)."""
import math

import torch

PAGE = 256


class PagePool:
    def __init__(self, n_layers, n_pages, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"):
        self.device = device
        self.k = [torch.zeros(n_pages, PAGE, n_kv_heads, head_dim, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v = [torch.zeros_like(t) for t in self.k]
        self._free = list(range(n_pages - 1, -1, -1))
        self._pages: dict = {}
        self._len: dict = {}

    @staticmethod
    def bytes_per_page(n_layers, n_kv_heads, head_dim, dtype) -> int:
        return n_layers * 2 * PAGE * n_kv_heads * head_dim * torch.tensor([], dtype=dtype).element_size()

    @classmethod
    def for_budget(cls, cfg, budget_bytes, device="cuda", dtype=torch.bfloat16) -> "PagePool":
        n = int(budget_bytes // cls.bytes_per_page(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, dtype))
        return cls(cfg.n_layers, n, cfg.n_kv_heads, cfg.head_dim, dtype, device)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def reserve(self, seq_id, total_len: int) -> None:
        pages = self._pages.setdefault(seq_id, [])
        self._len.setdefault(seq_id, 0)
        need = math.ceil(total_len / PAGE) - len(pages)
        if need > len(self._free):
            raise MemoryError(f"KV pool exhausted: need {need} pages, {len(self._free)} free")
        for _ in range(max(0, need)):
            pages.append(self._free.pop())

    def release(self, seq_id) -> None:
        self._free.extend(reversed(self._pages.pop(seq_id, [])))
        self._len.pop(seq_id, None)

    def length(self, seq_id) -> int:
        return self._len[seq_id]

    def set_length(self, seq_id, n: int) -> None:
        self._len[seq_id] = n

    def block_table(self, seq_ids) -> torch.Tensor:
        width = max(len(self._pages[s]) for s in seq_ids)
        rows = [self._pages[s] + [0] * (width - len(self._pages[s])) for s in seq_ids]
        return torch.tensor(rows, dtype=torch.int32, device=self.device)

    def lengths(self, seq_ids) -> torch.Tensor:
        return torch.tensor([self._len[s] for s in seq_ids], dtype=torch.int32, device=self.device)
```
- [ ] **Step 3: Run CPU + GPU tests → PASS; whole suite. Step 4: Commit** — `git add kernelscope/serve/kvcache.py tests/test_serve_kvcache.py && git commit -m "Add the paged KV pool" -- kernelscope/serve/kvcache.py tests/test_serve_kvcache.py`

---

### Task 3: Decoder forward (Llama / Qwen3)

**Files:**
- Create: `kernelscope/serve/model.py`
- Test: `tests/test_serve_model.py`

**Interfaces:**
- Consumes: `ModelConfig`, `rope_inv_freq`, `load_weights` (Task 1); `PagePool` (Task 2).
- Produces:
  - `DecoderModel(cfg, weights: dict, device="cuda")`; `DecoderModel.from_pretrained(repo_id, device="cuda")`; `DecoderModel.random(cfg, device="cuda", seed=0)` (random weights, for tests)
  - `model.prefill(seq_id, token_ids: list[int], pool, chunk=4096) -> torch.Tensor` (logits of the last prompt token, float32 `[vocab]`); reserves pages and sets the pool length
  - `model.decode(seq_ids, token_ids: list[int], pool, num_splits: int = 0, timer=None) -> torch.Tensor` (float32 `[B, vocab]`); reserves one more token per sequence, appends K/V through `flash_attn_with_kvcache(..., k=, v=, cache_seqlens=, block_table=, causal=True, num_splits=num_splits)`, advances pool lengths
  - `AttentionTimer` with `start(layer)` / `stop(layer)` recording CUDA events and `total_us() -> float` (synchronises)

Layer math (Hugging Face conventions): `h = rms(x)`, `q,k,v = h @ W.T`, reshape to heads; Qwen3: `q = rms_head(q, q_norm)`, `k = rms_head(k, k_norm)`; RoPE on q and k at absolute positions `pos = cache_len + i` with `cos/sin` from `cat(freqs, freqs)` in float32 cast to bf16 and `rotate_half(x) = cat(-x[..., d/2:], x[..., :d/2])`; attention via `flash_attn_with_kvcache`; `x += o_proj(attn)`; `x += down(silu(gate(h2)) * up(h2))` with `h2 = rms(x)`; logits from `rms(x_last) @ lm_head.T` (`lm_head = embed` when tied).

- [ ] **Step 1: Failing tests** — `tests/test_serve_model.py`:
```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.serve.hf import ModelConfig  # noqa: E402
from kernelscope.serve.kvcache import PagePool  # noqa: E402
from kernelscope.serve.model import DecoderModel  # noqa: E402

pytestmark = pytest.mark.gpu
TINY = ModelConfig(arch="qwen3", n_layers=2, hidden=256, intermediate=512, n_heads=4, n_kv_heads=2, head_dim=128,
                   vocab=1000, rope_theta=1e4, rope_scaling=None, rms_eps=1e-6, tie_embeddings=True, qk_norm=True)


def _free_gib():
    return torch.cuda.mem_get_info()[0] / 2**30


def test_decode_continues_prefill_consistently():
    m = DecoderModel.random(TINY)
    pool = PagePool(TINY.n_layers, 16, TINY.n_kv_heads, TINY.head_dim)
    toks = list(range(1, 40))
    full = m.prefill("a", toks, pool)                         # 39 tokens at once
    pool2 = PagePool(TINY.n_layers, 16, TINY.n_kv_heads, TINY.head_dim)
    m.prefill("b", toks[:-1], pool2)
    step = m.decode(["b"], [toks[-1]], pool2)[0]               # 38 + one decode step
    assert torch.allclose(full, step, atol=5e-2, rtol=0)
    assert pool2.length("b") == 39


@pytest.mark.parametrize("ns", [1, 4, 0])
def test_num_splits_does_not_change_the_result_beyond_rounding(ns):
    m = DecoderModel.random(TINY)
    outs = []
    for s in (1, ns):
        pool = PagePool(TINY.n_layers, 64, TINY.n_kv_heads, TINY.head_dim)
        for i, n in enumerate((700, 30, 300)):
            m.prefill(i, list(range(1, n + 1)), pool)
        outs.append(m.decode([0, 1, 2], [5, 6, 7], pool, num_splits=s))
    assert (outs[0] - outs[1]).abs().max().item() < 5e-2
    assert (outs[0].argmax(-1) == outs[1].argmax(-1)).all()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 18 * 2**30,
                    reason="needs ~18 GiB free for the model and the transformers reference")
def test_qwen3_4b_matches_transformers():
    from transformers import AutoModelForCausalLM
    from kernelscope.serve.hf import snapshot_dir
    repo = "Qwen/Qwen3-4B-Instruct-2507"
    ours = DecoderModel.from_pretrained(repo)
    ids = list(range(100, 132))
    pool = PagePool(ours.cfg.n_layers, 4, ours.cfg.n_kv_heads, ours.cfg.head_dim)
    got = ours.prefill("x", ids, pool)
    del ours, pool
    torch.cuda.empty_cache()
    ref_model = AutoModelForCausalLM.from_pretrained(snapshot_dir(repo), torch_dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
    with torch.no_grad():
        ref = ref_model(torch.tensor([ids], device="cuda")).logits[0, -1].float()
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=0).item()
    assert cos > 0.999, cos
    assert got.argmax().item() == ref.argmax().item()
```
- [ ] **Step 2: Implement `kernelscope/serve/model.py`:**
```python
"""Llama / Qwen3 decoder forward for serving experiments: plain PyTorch plus flash_attn_with_kvcache
on a paged KV pool. RoPE is applied in PyTorch (Hugging Face convention), so the attention kernel only
appends and attends; `num_splits` is passed straight through to flash-attn."""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from kernelscope.serve.hf import ModelConfig, load_weights, rope_inv_freq, snapshot_dir


def _rms(x, w, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return w * xf.to(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class AttentionTimer:
    def __init__(self):
        self._ev = []

    def start(self, layer):
        e = torch.cuda.Event(enable_timing=True); e.record(); self._ev.append([e, None])

    def stop(self, layer):
        e = torch.cuda.Event(enable_timing=True); e.record(); self._ev[-1][1] = e

    def total_us(self) -> float:
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) * 1e3 for s, e in self._ev)


class DecoderModel:
    def __init__(self, cfg: ModelConfig, weights: dict, device="cuda"):
        self.cfg, self.w, self.device = cfg, weights, device
        self.inv_freq = rope_inv_freq(cfg).to(device)
        self.lm_head = weights["model.embed_tokens.weight"] if cfg.tie_embeddings else weights["lm_head.weight"]

    @classmethod
    def from_pretrained(cls, repo_id: str, device="cuda") -> "DecoderModel":
        path = snapshot_dir(repo_id)
        return cls(ModelConfig.from_dir(path), load_weights(path, device), device)

    @classmethod
    def random(cls, cfg: ModelConfig, device="cuda", seed=0) -> "DecoderModel":
        g = torch.Generator(device=device).manual_seed(seed)
        H, Hk, d, D, I = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, cfg.hidden, cfg.intermediate

        def r(*s):
            return (torch.randn(*s, generator=g, device=device) * 0.02).to(torch.bfloat16)

        w = {"model.embed_tokens.weight": r(cfg.vocab, D), "model.norm.weight": torch.ones(D, device=device, dtype=torch.bfloat16)}
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            w.update({p + "self_attn.q_proj.weight": r(H * d, D), p + "self_attn.k_proj.weight": r(Hk * d, D),
                      p + "self_attn.v_proj.weight": r(Hk * d, D), p + "self_attn.o_proj.weight": r(D, H * d),
                      p + "mlp.gate_proj.weight": r(I, D), p + "mlp.up_proj.weight": r(I, D),
                      p + "mlp.down_proj.weight": r(D, I),
                      p + "input_layernorm.weight": torch.ones(D, device=device, dtype=torch.bfloat16),
                      p + "post_attention_layernorm.weight": torch.ones(D, device=device, dtype=torch.bfloat16)})
            if cfg.qk_norm:
                w[p + "self_attn.q_norm.weight"] = torch.ones(d, device=device, dtype=torch.bfloat16)
                w[p + "self_attn.k_norm.weight"] = torch.ones(d, device=device, dtype=torch.bfloat16)
        if not cfg.tie_embeddings:
            w["lm_head.weight"] = r(cfg.vocab, D)
        return cls(cfg, w, device)

    def _rope(self, positions):                       # positions: [B, T] int64
        freqs = positions.float()[..., None] * self.inv_freq      # [B, T, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.bfloat16)[:, :, None, :], emb.sin().to(torch.bfloat16)[:, :, None, :]

    @torch.no_grad()
    def _forward(self, x, positions, pool, seq_ids, cache_lens, num_splits, timer):
        cfg, w = self.cfg, self.w
        B, T, _ = x.shape
        cos, sin = self._rope(positions)
        bt = pool.block_table(seq_ids)
        for i in range(cfg.n_layers):
            p = f"model.layers.{i}."
            h = _rms(x, w[p + "input_layernorm.weight"], cfg.rms_eps)
            q = (h @ w[p + "self_attn.q_proj.weight"].T).view(B, T, cfg.n_heads, cfg.head_dim)
            k = (h @ w[p + "self_attn.k_proj.weight"].T).view(B, T, cfg.n_kv_heads, cfg.head_dim)
            v = (h @ w[p + "self_attn.v_proj.weight"].T).view(B, T, cfg.n_kv_heads, cfg.head_dim)
            if cfg.qk_norm:
                q = _rms(q, w[p + "self_attn.q_norm.weight"], cfg.rms_eps)
                k = _rms(k, w[p + "self_attn.k_norm.weight"], cfg.rms_eps)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            if timer is not None:
                timer.start(i)
            a = flash_attn_with_kvcache(q, pool.k[i], pool.v[i], k=k, v=v, cache_seqlens=cache_lens,
                                        block_table=bt, causal=True, num_splits=num_splits)
            if timer is not None:
                timer.stop(i)
            x = x + a.reshape(B, T, -1) @ w[p + "self_attn.o_proj.weight"].T
            h2 = _rms(x, w[p + "post_attention_layernorm.weight"], cfg.rms_eps)
            x = x + (F.silu(h2 @ w[p + "mlp.gate_proj.weight"].T) * (h2 @ w[p + "mlp.up_proj.weight"].T)) \
                @ w[p + "mlp.down_proj.weight"].T
        x = _rms(x[:, -1], w["model.norm.weight"], cfg.rms_eps)
        return (x @ self.lm_head.T).float()

    def prefill(self, seq_id, token_ids, pool, chunk=4096):
        pool.reserve(seq_id, len(token_ids))        # a new sequence starts at length 0
        logits = None
        for s in range(0, len(token_ids), chunk):
            ids = torch.tensor([token_ids[s:s + chunk]], device=self.device)
            start = pool.length(seq_id)
            pos = torch.arange(start, start + ids.shape[1], device=self.device)[None]
            x = self.w["model.embed_tokens.weight"][ids]
            logits = self._forward(x, pos, pool, [seq_id], pool.lengths([seq_id]), 0, None)[0]
            pool.set_length(seq_id, start + ids.shape[1])
        return logits

    def decode(self, seq_ids, token_ids, pool, num_splits: int = 0, timer=None):
        lens = [pool.length(s) for s in seq_ids]
        for s, n in zip(seq_ids, lens):
            pool.reserve(s, n + 1)
        cache_lens = torch.tensor(lens, dtype=torch.int32, device=self.device)
        pos = cache_lens.long()[:, None]
        x = self.w["model.embed_tokens.weight"][torch.tensor(token_ids, device=self.device)][:, None, :]
        logits = self._forward(x, pos, pool, list(seq_ids), cache_lens, num_splits, timer)
        for s, n in zip(seq_ids, lens):
            pool.set_length(s, n + 1)
        return logits
```
- [ ] **Step 3: Run** GPU tests (hygiene/contention rules) → PASS; if `test_qwen3_4b_matches_transformers` is skipped for memory, say so and run it when the GPU is free before finishing. If cosine similarity fails, report the value, the argmax tokens, and a per-layer comparison of hidden states against the reference (hook `ref_model.model.layers[i]` outputs) instead of changing tolerances.
- [ ] **Step 4: Whole suite, commit** — `git add kernelscope/serve/model.py tests/test_serve_model.py && git commit -m "Add a Llama/Qwen3 decoder forward on the paged KV pool" -- kernelscope/serve/model.py tests/test_serve_model.py`

---

### Task 4: Dispatch policies

**Files:**
- Create: `kernelscope/serve/dispatch.py`
- Test: `tests/test_serve_dispatch.py`

**Interfaces:**
- Consumes: `predict`, `PAGED_VARIANTS` (model plan Task 4); `parse_variant` (geometry); `MachineSpec`, `ModelParams`.
- Produces:
  - `Policy` protocol: `name: str`, `choose(lens: list[int], n_heads: int, n_kv_heads: int) -> int` returning the `num_splits` argument for `flash_attn_with_kvcache` (0 = library heuristic)
  - `FixedPolicy(num_splits, name=None)` (names: `fa2` for 1, `heuristic` for 0, `fixed{N}` otherwise)
  - `ModelPolicy(machine, params, cache_state="cold", candidates=PAGED_VARIANTS)`: builds `Workload(decode, B=len(lens), L_q=1, L_kv=max(lens), H_q, H_kv, d=128, dtype="bfloat16", kv_lens=lens)`, predicts every candidate, returns the argmin's `num_splits` (the heuristic candidate resolves to `resolve_splits`); caches decisions by `(n_heads, n_kv_heads, tuple(ceil(l / 256) for l in lens))` (order preserved: block placement depends on it); `.last` holds the last ranking (list of `(plugin, time_us)`)
  - `TablePolicy(csv_path)`: rows of a `dispatch-table` CSV (paged family); nearest measured workload by Euclidean distance on `(log2 B, log2 max(lens), log2 mean(lens))`; returns `parse_variant(best_kernel).num_splits`
  - `make_policy(spec, machine=None, params=None) -> Policy` for `"fa2" | "heuristic" | "fixed:N" | "model" | "table:PATH"`

- [ ] **Step 1: Failing tests** — `tests/test_serve_dispatch.py`:
```python
import pandas as pd
import pytest

from kernelscope.serve.dispatch import FixedPolicy, ModelPolicy, TablePolicy, make_policy
from tests.test_model_predict import M, P

RAGGED = [32768, 32768] + [1024] * 30


def test_fixed_policies_and_names():
    assert FixedPolicy(1).name == "fa2" and FixedPolicy(0).name == "heuristic" and FixedPolicy(8).name == "fixed8"
    assert FixedPolicy(8).choose(RAGGED, 32, 8) == 8


def test_model_policy_splits_the_long_sequences_of_a_ragged_batch():
    pol = ModelPolicy(M, P)
    n = pol.choose(RAGGED, 32, 8)
    assert n > 1                                   # the heuristic would pick 1 here (spec F15)
    assert pol.last[0][1] <= min(t for _, t in pol.last)


def test_model_policy_caches_by_page_quantised_lengths():
    pol = ModelPolicy(M, P)
    lens = [32100, 32100] + [1000] * 30                               # 32100 and 32101 share page 126
    pol.choose(lens, 32, 8)
    calls = pol.predictions
    pol.choose([32101, 32101] + [1000] * 30, 32, 8)
    assert pol.predictions == calls
    pol.choose([32300, 32300] + [1000] * 30, 32, 8)                   # crosses into page 127
    assert pol.predictions == 2 * calls


def test_table_policy_picks_the_nearest_measured_workload(tmp_path):
    t = pd.DataFrame([{"workload_key": "decode_B32_Lq1_Lkv32768x2+1024x30_Hq32_Hkv8_d128_float16_causal",
                       "B": 32, "L_kv": 32768, "best_kernel": "fd_s8_paged"},
                      {"workload_key": "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal",
                       "B": 1, "L_kv": 1024, "best_kernel": "fd_s4_paged"}])
    f = tmp_path / "t.csv"; t.to_csv(f, index=False)
    assert TablePolicy(f).choose(RAGGED, 32, 8) == 8
    assert TablePolicy(f).choose([1000], 32, 8) == 4


def test_make_policy_parses_specs(tmp_path):
    assert make_policy("fa2").choose([10], 32, 8) == 1
    assert make_policy("heuristic").choose([10], 32, 8) == 0
    assert make_policy("fixed:16").choose([10], 32, 8) == 16
    assert isinstance(make_policy("model", M, P), ModelPolicy)
    with pytest.raises(ValueError):
        make_policy("bogus")
```
- [ ] **Step 2: Implement `kernelscope/serve/dispatch.py`:**
```python
"""Per-step choice of flash-attn's num_splits for the paged decode attention (spec §5.1-5.2).
Every policy returns an argument for the same kernel call, so outputs differ only by rounding."""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kernelscope.model.geometry import parse_variant, resolve_splits
from kernelscope.model.predict import PAGED_VARIANTS, predict
from kernelscope.workload import Workload

PAGE = 256


class FixedPolicy:
    def __init__(self, num_splits: int, name: str | None = None):
        self.num_splits = num_splits
        self.name = name or {0: "heuristic", 1: "fa2"}.get(num_splits, f"fixed{num_splits}")

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        return self.num_splits


def _workload(lens, n_heads, n_kv_heads):
    return Workload(phase="decode", B=len(lens), L_q=1, L_kv=max(lens), H_q=n_heads, H_kv=n_kv_heads, d=128,
                    dtype="bfloat16", kv_lens=list(lens))


class ModelPolicy:
    name = "model"

    def __init__(self, machine, params, cache_state="cold", candidates=PAGED_VARIANTS):
        self.machine, self.params, self.cache_state, self.candidates = machine, params, cache_state, list(candidates)
        self._cache = {}
        self.predictions = 0
        self.last = []

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        key = (n_heads, n_kv_heads, tuple(math.ceil(l / PAGE) for l in lens))
        if key not in self._cache:
            w = _workload(lens, n_heads, n_kv_heads)
            ranked = sorted(((c, predict(c, w, self.machine, self.params, self.cache_state).time_us)
                             for c in self.candidates), key=lambda t: t[1])
            self.predictions += len(self.candidates)
            best = parse_variant(ranked[0][0])
            self._cache[key] = (resolve_splits(best, w, self.machine.n_sm), ranked)
        n, self.last = self._cache[key]
        return n


class TablePolicy:
    name = "table"

    def __init__(self, csv_path):
        t = pd.read_csv(csv_path)
        ws = [Workload.from_key(k) for k in t.workload_key]
        self._feat = np.array([[math.log2(w.B), math.log2(max(w.lens())), math.log2(np.mean(w.lens()))] for w in ws])
        self._best = list(t.best_kernel)

    def choose(self, lens, n_heads, n_kv_heads) -> int:
        f = np.array([math.log2(len(lens)), math.log2(max(lens)), math.log2(np.mean(lens))])
        i = int(np.argmin(((self._feat - f) ** 2).sum(axis=1)))
        return parse_variant(self._best[i]).num_splits


def make_policy(spec: str, machine=None, params=None):
    if spec == "fa2":
        return FixedPolicy(1)
    if spec == "heuristic":
        return FixedPolicy(0)
    if spec.startswith("fixed:"):
        return FixedPolicy(int(spec.split(":", 1)[1]))
    if spec == "model":
        return ModelPolicy(machine, params)
    if spec.startswith("table:"):
        return TablePolicy(Path(spec.split(":", 1)[1]))
    raise ValueError(f"unknown policy {spec!r}; use fa2, heuristic, fixed:N, model, table:PATH")
```
- [ ] **Step 3: Run → PASS (CPU); whole suite. Step 4: Commit** — `git add kernelscope/serve/dispatch.py tests/test_serve_dispatch.py && git commit -m "Add num_splits dispatch policies" -- kernelscope/serve/dispatch.py tests/test_serve_dispatch.py`

---

### Task 5: Engine and scenarios

**Files:**
- Create: `kernelscope/serve/engine.py`, `kernelscope/serve/scenarios.py`, `scenarios/ragged_1long.yaml`, `scenarios/ragged_2long.yaml`, `scenarios/uniform.yaml`, `scenarios/arrivals.yaml`, `scenarios/tiny.yaml`
- Test: `tests/test_serve_engine.py`

**Interfaces:**
- Consumes: `DecoderModel`, `AttentionTimer` (Task 3), `PagePool` (Task 2), policies (Task 4).
- Produces:
  - `scenarios.Request(rid: int, prompt_len: int, max_new_tokens: int, arrival_step: int)`; `scenarios.load(path) -> list[Request]`; YAML format: `requests: [{count, prompt_len, max_new_tokens, arrival_step}]` expanded in order (rids 0..n-1); `scenarios.prompt_ids(req, vocab, seed=0) -> list[int]` (seeded random token ids in `[1000, vocab - 1000)`)
  - `engine.Engine(model, pool, policy, max_batch=64)`; `engine.run(requests, vocab, record_logits_steps=0) -> RunResult` with `steps: pd.DataFrame` (`step, policy, B, len_max, len_sum, n_long, num_splits, attn_us, step_us`), `tokens: pd.DataFrame` (`rid, step, token, t_us` where `t_us` is the wall time at which the token was produced, from a CUDA-synchronised clock), `prefill: pd.DataFrame` (`rid, prompt_len, prefill_us`), `logits: list[torch.Tensor]` (the first `record_logits_steps` decode steps' logits, CPU float32)
  - Semantics: at each step, first prefill every request with `arrival_step <= step` not yet admitted (in rid order; its first generated token is the prefill argmax), then run one decode step for all active requests (admission order), then retire requests that produced `max_new_tokens`; greedy argmax; `n_long` = sequences with length ≥ 4096; `step_us` = CUDA-event time of the decode call; `attn_us` = `AttentionTimer.total_us()` for that step. The token sequence produced depends only on the model and prompts (never on the policy's timing), so every policy sees the same batch composition at every step.

Scenario files:
```yaml
# scenarios/ragged_1long.yaml — one long-context request among short chats (spec §5.1)
requests:
  - {count: 1, prompt_len: 16384, max_new_tokens: 128, arrival_step: 0}
  - {count: 31, prompt_len: 1024, max_new_tokens: 128, arrival_step: 0}
```
`ragged_2long.yaml`: counts 2 × 16384 and 30 × 1024. `uniform.yaml`: 32 × 1024. `arrivals.yaml`: 1 × 16384 at step 0, then 8 groups of 4 × 1024 arriving at steps 0, 16, 32, …, 112, all `max_new_tokens: 128`. `tiny.yaml`: 1 × 600 and 3 × 40, `max_new_tokens: 8`.

- [ ] **Step 1: Failing tests** — `tests/test_serve_engine.py` (GPU, uses `DecoderModel.random(TINY)` from `tests/test_serve_model.py`'s `TINY` config):
```python
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("flash_attn")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from kernelscope.serve.dispatch import FixedPolicy  # noqa: E402
from kernelscope.serve.engine import Engine  # noqa: E402
from kernelscope.serve.kvcache import PagePool  # noqa: E402
from kernelscope.serve.model import DecoderModel  # noqa: E402
from kernelscope.serve.scenarios import Request, load  # noqa: E402
from tests.test_serve_model import TINY  # noqa: E402

pytestmark = pytest.mark.gpu
REQS = [Request(0, 600, 8, 0), Request(1, 40, 8, 0), Request(2, 40, 6, 0), Request(3, 100, 6, 3)]


def _run(policy, reqs=REQS, record=0):
    m = DecoderModel.random(TINY)
    pool = PagePool(TINY.n_layers, 32, TINY.n_kv_heads, TINY.head_dim)
    return Engine(m, pool, policy).run(reqs, TINY.vocab, record_logits_steps=record)


def test_batch_composition_follows_arrivals_and_lengths():
    r = _run(FixedPolicy(0))
    assert list(r.steps.B[:4]) == [3, 3, 3, 4]             # request 3 arrives at step 3
    assert r.tokens.groupby("rid").size().to_dict() == {0: 8, 1: 8, 2: 6, 3: 6}
    assert (r.steps.attn_us > 0).all() and (r.steps.step_us >= r.steps.attn_us).all()


def test_same_tokens_under_different_policies():
    a, b = _run(FixedPolicy(1), record=3), _run(FixedPolicy(16), record=3)
    assert a.tokens.token.tolist() == b.tokens.token.tolist()
    assert a.steps.B.tolist() == b.steps.B.tolist()
    assert max((x - y).abs().max().item() for x, y in zip(a.logits, b.logits)) < 5e-2


def test_pages_are_returned_after_the_run():
    m = DecoderModel.random(TINY)
    pool = PagePool(TINY.n_layers, 32, TINY.n_kv_heads, TINY.head_dim)
    Engine(m, pool, FixedPolicy(0)).run(REQS, TINY.vocab)
    assert pool.free_pages == 32


def test_scenario_files_load():
    rs = load("scenarios/ragged_1long.yaml")
    assert len(rs) == 32 and rs[0].prompt_len == 16384 and rs[1].prompt_len == 1024
    assert [r.rid for r in rs] == list(range(32))
```
- [ ] **Step 2: Implement `kernelscope/serve/scenarios.py`:**
```python
"""Request traces for serving experiments (YAML), with seeded synthetic prompts."""
from dataclasses import dataclass

import numpy as np
import yaml


@dataclass(frozen=True)
class Request:
    rid: int
    prompt_len: int
    max_new_tokens: int
    arrival_step: int


def load(path) -> list:
    with open(path) as f:
        spec = yaml.safe_load(f)
    out = []
    for g in spec["requests"]:
        for _ in range(g["count"]):
            out.append(Request(len(out), g["prompt_len"], g["max_new_tokens"], g["arrival_step"]))
    return out


def prompt_ids(req: Request, vocab: int, seed: int = 0) -> list:
    rng = np.random.default_rng(seed * 1_000_003 + req.rid)
    lo, hi = min(1000, vocab // 4), max(vocab - 1000, vocab // 2)
    return rng.integers(lo, hi, size=req.prompt_len).tolist()
```
`kernelscope/serve/engine.py`:
```python
"""Deterministic continuous-batching decode loop with per-step attention timing."""
import time
from dataclasses import dataclass, field

import pandas as pd
import torch

from kernelscope.serve.model import AttentionTimer
from kernelscope.serve.scenarios import prompt_ids

LONG = 4096


@dataclass
class RunResult:
    steps: pd.DataFrame
    tokens: pd.DataFrame
    prefill: pd.DataFrame
    logits: list = field(default_factory=list)


class Engine:
    def __init__(self, model, pool, policy, max_batch=64):
        self.model, self.pool, self.policy, self.max_batch = model, pool, policy, max_batch

    def run(self, requests, vocab, record_logits_steps=0) -> RunResult:
        cfg = self.model.cfg
        pending = sorted(requests, key=lambda r: (r.arrival_step, r.rid))
        active, last_tok, produced = [], {}, {}
        steps, toks, pre, logits = [], [], [], []
        t0 = time.perf_counter()
        step = 0
        while pending or active:
            while pending and pending[0].arrival_step <= step and len(active) < self.max_batch:
                r = pending.pop(0)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                lg = self.model.prefill(r.rid, prompt_ids(r, vocab), self.pool)
                e.record(); e.synchronize()
                pre.append({"rid": r.rid, "prompt_len": r.prompt_len, "prefill_us": s.elapsed_time(e) * 1e3})
                last_tok[r.rid] = int(lg.argmax())
                produced[r.rid] = 1
                toks.append({"rid": r.rid, "step": step, "token": last_tok[r.rid], "t_us": (time.perf_counter() - t0) * 1e6})
                active.append(r)
            active = [r for r in active if produced[r.rid] < r.max_new_tokens or self._retire(r)]
            if not active:
                step += 1
                continue
            ids = [r.rid for r in active]
            lens = [self.pool.length(i) for i in ids]
            ns = self.policy.choose(lens, cfg.n_heads, cfg.n_kv_heads)
            timer = AttentionTimer()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            lg = self.model.decode(ids, [last_tok[i] for i in ids], self.pool, num_splits=ns, timer=timer)
            e.record(); e.synchronize()
            now = (time.perf_counter() - t0) * 1e6
            if len(logits) < record_logits_steps:
                logits.append(lg.cpu())
            nxt = lg.argmax(-1).tolist()
            for r, t in zip(active, nxt):
                last_tok[r.rid] = t
                produced[r.rid] += 1
                toks.append({"rid": r.rid, "step": step, "token": t, "t_us": now})
            steps.append({"step": step, "policy": self.policy.name, "B": len(ids), "len_max": max(lens),
                          "len_sum": sum(lens), "n_long": sum(l >= LONG for l in lens), "num_splits": ns,
                          "attn_us": timer.total_us(), "step_us": s.elapsed_time(e) * 1e3})
            active = [r for r in active if produced[r.rid] < r.max_new_tokens or self._retire(r)]
            step += 1
        return RunResult(pd.DataFrame(steps), pd.DataFrame(toks), pd.DataFrame(pre), logits)

    def _retire(self, r) -> bool:
        self.pool.release(r.rid)
        return False
```
Scenario YAML files as specified above.
- [ ] **Step 3: Run** (GPU rules) → PASS; whole suite.
- [ ] **Step 4: Commit** — `git add kernelscope/serve/engine.py kernelscope/serve/scenarios.py scenarios tests/test_serve_engine.py && git commit -m "Add the continuous-batching engine and request scenarios" -- kernelscope/serve/engine.py kernelscope/serve/scenarios.py scenarios tests/test_serve_engine.py`

---

### Task 6: `serve run` and `serve compare`

**Files:**
- Create: `kernelscope/serve/report.py`
- Modify: `kernelscope/cli.py` (add a `serve` command group with `run` and `compare`)
- Test: `tests/test_serve_report.py`

**Interfaces:**
- Produces:
  - `report.tpot_us(tokens: pd.DataFrame) -> pd.Series` (per rid: mean gap between consecutive token timestamps after the first token)
  - `report.summarize(results_dir) -> pd.DataFrame` — one row per policy from `<dir>/<policy>/steps.parquet` and `tokens.parquet`: `policy, steps, attn_ms_per_step, step_ms_per_step, attn_share, tpot_ms_mean, tpot_ms_p50, tpot_ms_p90, speedup_vs_heuristic` (heuristic TPOT mean / policy TPOT mean)
  - `report.oracle(results_dir, fixed=("fa2", "fixed2", ..., "fixed128")) -> pd.DataFrame` — per step, the minimum `attn_us` over the fixed policies present, and per policy the ratio of its summed `attn_us` to that per-step oracle sum
  - CLI `kernelscope serve run --model REPO --scenario YAML --policy SPEC [--policy SPEC ...] --out DIR [--kv-gib 8] [--machine M --params P] [--table CSV] [--warmup-steps 0]`: loads the model once, and for each policy builds a fresh pool (budget `--kv-gib`), runs the scenario, writes `DIR/<policy.name>/{steps,tokens,prefill}.parquet` plus `DIR/<policy.name>/meta.json` (model, scenario, policy spec, GPU name, date, `nvidia-smi` compute apps at start); the model's `model` policy needs `--machine`/`--params`, `table:` needs its CSV path inside the spec
  - CLI `kernelscope serve compare --results DIR [--out summary.csv]` prints `summarize` and `oracle`

- [ ] **Step 1: Failing tests** (CPU) — `tests/test_serve_report.py`: build two fake policy directories with synthetic `steps.parquet` (e.g. heuristic attn 10 ms/step, model 4 ms/step, 4 steps) and `tokens.parquet` (tokens every 20 ms vs 14 ms), then assert `summarize` gives `speedup_vs_heuristic ≈ 20/14` for `model` and 1.0 for `heuristic`, `attn_share` = attn/step, and that `oracle` with two fixed policies takes the per-step minimum. Write the synthetic frames with `pd.DataFrame(...).to_parquet(...)` in `tmp_path`.
- [ ] **Step 2: Implement `report.py`** exactly to the interface above (group by rid, sort by `t_us`, `diff()` after the first token; `attn_ms_per_step = steps.attn_us.mean() / 1e3`, etc.; missing heuristic → `speedup_vs_heuristic` NaN) and the CLI group (pattern: `p_serve = sub.add_parser("serve", ...)`, `ssub = p_serve.add_subparsers(dest="serve_cmd", required=True)`).
- [ ] **Step 3: Run → PASS; whole suite. Step 4: Commit** — `git add kernelscope/serve/report.py tests/test_serve_report.py && git commit -m "Add serve run/compare and serving reports" -- kernelscope/serve/report.py kernelscope/cli.py tests/test_serve_report.py`

---

### Task 7: Output preservation check

**Files:**
- Create: `kernelscope/serve/equivalence.py`
- Modify: `kernelscope/cli.py` (add `serve equivalence`)
- Test: `tests/test_serve_equivalence.py` (GPU, tiny random model)

**Interfaces:**
- Produces: `equivalence.compare_policies(model, policies, requests, vocab, kv_bytes, record_steps=32) -> pd.DataFrame` with one row per non-reference policy: `policy, reference, steps_compared, max_abs_logit_diff, token_agreement` (the reference is the first policy; token agreement = fraction of generated tokens identical at the same (rid, position)); CLI `kernelscope serve equivalence --model REPO --scenario YAML --policy SPEC ... --out CSV [--kv-gib] [--machine --params]`.
- Spec §5.3: the claim is "mathematically identical, numerically within rounding"; report both numbers; never assert bit-identity.

- [ ] **Step 1: Failing test** — with `DecoderModel.random(TINY)` and the `REQS` of Task 5's tests, compare `FixedPolicy(1)`, `FixedPolicy(0)`, `FixedPolicy(16)`: every row has `token_agreement == 1.0` and `max_abs_logit_diff < 5e-2`.
- [ ] **Step 2: Implement** — run each policy on a fresh pool with `record_logits_steps=record_steps`, align logits step by step (same batch order by construction), compute the maxima and token agreement from the `tokens` frames.
- [ ] **Step 3: Run → PASS; whole suite. Step 4: Commit** — `git add kernelscope/serve/equivalence.py tests/test_serve_equivalence.py && git commit -m "Add the cross-policy output equivalence check" -- kernelscope/serve/equivalence.py kernelscope/cli.py tests/test_serve_equivalence.py`

---

### Task 8: Serving experiments

**Depends on:** Tasks 1–7 and the fitted `models/rtx4090.json` from the model plan. GPU-exclusive timing: wait for an idle GPU (only `rerun`), per the contention rule.

**Files:**
- Create: `docs/plan/2026-09-19-p2-serving-results.md`, `results/.../tables/serve_*.csv` (not committed)
- Modify: `docs/STATUS.md`, `README.md`

- [ ] **Step 1: Dispatch table for the table policy** — `$PY -m kernelscope.cli dispatch-table --results $R/uniform_s1_paged $R/ragged_s1_paged --family paged --cache-state cold --out $R/tables/serve_paged_cold.csv` with `R=/home/skkai/AI_Accelerator/kernelscope/results/hw_4090`. (Uniform and ragged paged tables together.)
- [ ] **Step 2: Runs** — for each scenario in `ragged_1long`, `ragged_2long`, `uniform`, `arrivals`, with `S=/home/skkai/AI_Accelerator/kernelscope/results/serve_4090`:
```bash
$PY -m kernelscope.cli serve run --model Qwen/Qwen3-4B-Instruct-2507 --scenario scenarios/<name>.yaml \
  --policy heuristic --policy fa2 --policy model --policy table:$R/tables/serve_paged_cold.csv \
  --policy fixed:2 --policy fixed:4 --policy fixed:8 --policy fixed:16 --policy fixed:32 \
  --machine machines/rtx4090.json --params models/rtx4090.json --kv-gib 9 --out $S/qwen3_4b/<name>
$PY -m kernelscope.cli serve compare --results $S/qwen3_4b/<name> --out $S/qwen3_4b/<name>/summary.csv
```
Run each `serve run` in the background with a log if it may exceed 10 minutes. If a run hits `MemoryError` from the pool, lower nothing silently: record it and try `--kv-gib 10`; if it still fails, skip that scenario and say so.
- [ ] **Step 3: Equivalence** — `serve equivalence` on `scenarios/tiny.yaml` scaled up: create `scenarios/equiv.yaml` with 1 × 4096 and 7 × 512 prompts, `max_new_tokens: 64`, policies `heuristic fa2 model fixed:8 fixed:32` → CSV.
- [ ] **Step 4: Secondary model (optional, only if ≥ 21 GiB free)** — `ragged_1long` with `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, policies `heuristic model fixed:8`, `--kv-gib 5`.
- [ ] **Step 5: Results note** — `docs/plan/2026-09-19-p2-serving-results.md`: GPU hygiene, per scenario the `summarize` table (TPOT mean/p50/p90, attention ms/step, attention share, speedup vs heuristic), the oracle table (each policy's attention time relative to the per-step best fixed split), the equivalence table, and 3–5 sentences of interpretation tied to spec F14–F19 (where the gain comes from; where it is zero, e.g. `uniform`). Report every number as measured; no tuning.
- [ ] **Step 6: Docs and commit** — README (serve commands), STATUS (≤ 10 lines). `git add docs/plan/2026-09-19-p2-serving-results.md scenarios/equiv.yaml && git commit -m "Run the dispatcher serving experiments on the RTX 4090" -- docs/plan/2026-09-19-p2-serving-results.md scenarios/equiv.yaml README.md docs/STATUS.md`
