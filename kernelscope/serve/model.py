"""Inference-only Llama/Qwen3 with paged flash-attn and a CPU correctness backend.

The CPU backend exists to exercise scheduling and model math without a GPU. It does
not implement split-K dispatch and its timings are not GPU performance evidence.
"""

from __future__ import annotations

import contextlib
import operator
import time

import torch
import torch.nn.functional as F

from kernelscope.serve.hf import ModelConfig, load_weights, rope_inv_freq, snapshot_dir
from kernelscope.serve.kvcache import PAGE


def _rms(x, weight, eps):
    xf = x.float()
    return weight * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


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


def _weight_shapes(cfg):
    shapes = {"model.embed_tokens.weight": (cfg.vocab, cfg.hidden), "model.norm.weight": (cfg.hidden,)}
    for layer in range(cfg.n_layers):
        prefix = f"model.layers.{layer}."
        shapes.update({
            prefix + "self_attn.q_proj.weight": (cfg.n_heads * cfg.head_dim, cfg.hidden),
            prefix + "self_attn.k_proj.weight": (cfg.n_kv_heads * cfg.head_dim, cfg.hidden),
            prefix + "self_attn.v_proj.weight": (cfg.n_kv_heads * cfg.head_dim, cfg.hidden),
            prefix + "self_attn.o_proj.weight": (cfg.hidden, cfg.n_heads * cfg.head_dim),
            prefix + "mlp.gate_proj.weight": (cfg.intermediate, cfg.hidden),
            prefix + "mlp.up_proj.weight": (cfg.intermediate, cfg.hidden),
            prefix + "mlp.down_proj.weight": (cfg.hidden, cfg.intermediate),
            prefix + "input_layernorm.weight": (cfg.hidden,),
            prefix + "post_attention_layernorm.weight": (cfg.hidden,),
        })
        if cfg.qk_norm:
            shapes[prefix + "self_attn.q_norm.weight"] = (cfg.head_dim,)
            shapes[prefix + "self_attn.k_norm.weight"] = (cfg.head_dim,)
    if not cfg.tie_embeddings:
        shapes["lm_head.weight"] = (cfg.vocab, cfg.hidden)
    return shapes


class DecoderModel:
    def __init__(self, cfg: ModelConfig, weights: dict, device="cuda"):
        expected = _weight_shapes(cfg)
        missing = set(expected) - set(weights)
        if missing:
            raise ValueError(f"checkpoint is missing required weights: {sorted(missing)}")
        requested = torch.device(device)
        embedding = weights["model.embed_tokens.weight"]
        self.device, self.dtype = embedding.device, embedding.dtype
        if self.device.type != requested.type or (requested.index is not None and self.device.index != requested.index):
            raise ValueError("weights must already be on the requested model device")
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU reference and CUDA flash-attn backends are supported")
        if not self.dtype.is_floating_point:
            raise ValueError("model weights must be floating point")
        if self.device.type == "cuda" and self.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("flash-attn requires bfloat16 or float16 model weights")
        for key, shape in expected.items():
            tensor = weights[key]
            if tuple(tensor.shape) != shape:
                raise ValueError(f"{key}: expected shape {shape}, got {tuple(tensor.shape)}")
            if tensor.device != self.device or tensor.dtype != self.dtype:
                raise ValueError(f"{key}: inconsistent model device or dtype")
        self.cfg, self.w = cfg, weights
        self.inv_freq = rope_inv_freq(cfg).to(self.device)
        self.lm_head = embedding if cfg.tie_embeddings else weights["lm_head.weight"]
        self.backend = "flash_attn_paged" if self.device.type == "cuda" else "torch_sdpa_reference"
        self._flash_attention = None
        if self.device.type == "cuda":
            try:
                from flash_attn import flash_attn_with_kvcache
            except ImportError as exc:
                raise ImportError("CUDA serving requires a compatible flash-attn installation") from exc
            self._flash_attention = flash_attn_with_kvcache

    @classmethod
    def from_pretrained(cls, repo_id: str, device="cuda", dtype=None):
        path = snapshot_dir(repo_id)
        dtype = dtype or (torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16)
        cfg = ModelConfig.from_dir(path)
        # Validate RoPE before allocating the checkpoint on the GPU.
        rope_inv_freq(cfg)
        return cls(cfg, load_weights(path, device=device, dtype=dtype), device)

    @classmethod
    def random(cls, cfg: ModelConfig, device="cuda", seed=0, dtype=None):
        """Small deterministic test model; these weights are not a trained language model."""
        dtype = dtype or (torch.float32 if torch.device(device).type == "cpu" else torch.bfloat16)
        generator = torch.Generator(device=device).manual_seed(seed)
        weights = {}
        for name, shape in _weight_shapes(cfg).items():
            if len(shape) == 1:
                weights[name] = torch.ones(shape, device=device, dtype=dtype)
            else:
                weights[name] = (torch.randn(shape, generator=generator, device=device) * 0.02).to(dtype)
        return cls(cfg, weights, device)

    def _rope(self, positions):
        # HF computes frequencies and trigonometry in float32, then casts to activations.
        frequencies = positions.float()[..., None] * self.inv_freq
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(self.dtype).unsqueeze(2), angles.sin().to(self.dtype).unsqueeze(2)

    def _check_pool(self, pool):
        if (pool.n_layers, pool.n_kv_heads, pool.head_dim) != (self.cfg.n_layers, self.cfg.n_kv_heads, self.cfg.head_dim):
            raise ValueError("KV pool geometry does not match model")
        if pool.device != self.device or pool.dtype != self.dtype:
            raise ValueError("KV pool dtype/device does not match model")

    def _token_tensor(self, token_ids):
        values = []
        for token in token_ids:
            if isinstance(token, bool):
                raise ValueError("token ids must be integers")
            try:
                token = operator.index(token)
            except TypeError as exc:
                raise ValueError("token ids must be integers") from exc
            if not 0 <= token < self.cfg.vocab:
                raise ValueError(f"token id {token} outside vocabulary [0, {self.cfg.vocab})")
            values.append(token)
        return torch.tensor(values, dtype=torch.long, device=self.device)

    def _cpu_attention(self, q, k, v, pool, layer, cache_lens, block_table):
        """Reference paged append + explicit bottom-right causal SDPA mask."""
        outputs = []
        tokens = q.shape[1]
        for row, before in enumerate(cache_lens.tolist()):
            end = before + tokens
            positions = torch.arange(before, end)
            pages = block_table[row, positions // PAGE].long()
            offsets = positions % PAGE
            pool.k[layer][pages, offsets] = k[row]
            pool.v[layer][pages, offsets] = v[row]
            page_ids = block_table[row, :(end + PAGE - 1) // PAGE].long()
            keys = pool.k[layer][page_ids].flatten(0, 1)[:end]
            values = pool.v[layer][page_ids].flatten(0, 1)[:end]
            repeats = self.cfg.n_heads // self.cfg.n_kv_heads
            keys = keys.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0)
            values = values.repeat_interleave(repeats, dim=1).transpose(0, 1).unsqueeze(0)
            mask = torch.arange(end)[None, :] <= positions[:, None]
            attn = F.scaled_dot_product_attention(
                q[row].transpose(0, 1).unsqueeze(0), keys, values,
                attn_mask=mask, dropout_p=0.0, scale=self.cfg.head_dim ** -0.5,
            )
            outputs.append(attn.squeeze(0).transpose(0, 1))
        return torch.stack(outputs)

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

    def _logits(self, hidden):
        hidden = _rms(hidden, self.w["model.norm.weight"], self.cfg.rms_eps)
        return F.linear(hidden, self.lm_head).float()

    @torch.inference_mode()
    def prefill(self, seq_id, token_ids: list[int], pool, chunk=4096, *, timer=None):
        """Fill a fresh (possibly pre-reserved) sequence, returning its last-token logits."""
        self._check_pool(pool)
        if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 1:
            raise ValueError("prefill chunk must be a positive integer")
        ids = self._token_tensor(token_ids)
        if not ids.numel():
            raise ValueError("prefill requires at least one prompt token")
        try:
            current = pool.length(seq_id)
        except KeyError:
            current = 0
        if current:
            raise ValueError("prefill requires a fresh sequence with zero cache length")
        pool.reserve(seq_id, ids.numel())
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

    @torch.inference_mode()
    def decode(self, seq_ids, token_ids: list[int], pool, num_splits=0, timer=None):
        self._check_pool(pool)
        seq_ids = list(seq_ids)
        if not seq_ids or len(set(seq_ids)) != len(seq_ids):
            raise ValueError("decode requires a nonempty batch of distinct sequence ids")
        ids = self._token_tensor(token_ids)
        if ids.numel() != len(seq_ids):
            raise ValueError("one input token is required per decode sequence")
        if isinstance(num_splits, bool) or not isinstance(num_splits, int) or not 0 <= num_splits <= 128:
            raise ValueError("num_splits must be an integer in [0, 128]")
        lengths = [pool.length(seq_id) for seq_id in seq_ids]
        if any(n == 0 for n in lengths):
            raise ValueError("decode sequences must have a nonempty prefilled cache")
        pool.reserve_many({seq_id: length + 1 for seq_id, length in zip(seq_ids, lengths)})
        cache_lens = pool.lengths(seq_ids)
        positions = cache_lens.long().unsqueeze(1)
        reg = timer.region if timer is not None else _null_region
        with reg("embed"):
            x = F.embedding(ids, self.w["model.embed_tokens.weight"]).unsqueeze(1)
        h = self._forward(x, positions, pool, seq_ids, cache_lens, num_splits, timer)
        with reg("lm_head"):
            logits = self._logits(h[:, -1])
        for seq_id, length in zip(seq_ids, lengths):
            pool.set_length(seq_id, length + 1)
        return logits
