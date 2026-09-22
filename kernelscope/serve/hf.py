"""Read local Hugging Face snapshots without downloading or executing model code."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

HUB = Path.home() / ".cache" / "huggingface" / "hub"
_ARCH = {"LlamaForCausalLM": "llama", "Qwen3ForCausalLM": "qwen3"}


def snapshot_dir(repo_id: str, cache=None) -> Path:
    """Resolve a local directory or cached repository; prefer its local main ref."""
    direct = Path(repo_id).expanduser()
    if direct.is_dir() and (direct / "config.json").is_file():
        return direct
    parts = str(repo_id).split("/")
    if len(parts) not in (1, 2) or any(p in ("", ".", "..") for p in parts):
        raise FileNotFoundError(f"not a local model directory or repository id: {repo_id}")
    default = Path(os.environ["HF_HOME"]) / "hub" if "HF_HOME" in os.environ else HUB
    hub = Path(cache or os.environ.get("HF_HUB_CACHE") or default).expanduser()
    repo = hub / ("models--" + "--".join(parts))
    root = repo / "snapshots"
    ref = repo / "refs" / "main"
    if ref.is_file():
        revision = ref.read_text().strip()
        if revision and Path(revision).name == revision:
            candidate = root / revision
            if (candidate / "config.json").is_file():
                return candidate
    snaps = [p for p in root.glob("*") if p.is_dir() and (p / "config.json").is_file()]
    if not snaps:
        raise FileNotFoundError(f"no local snapshot with config.json for {repo_id} under {root}")
    return max(snaps, key=lambda p: (p.stat().st_mtime_ns, p.name))


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

    def __post_init__(self):
        if self.arch not in ("llama", "qwen3"):
            raise ValueError(f"unsupported architecture: {self.arch}")
        for field in ("n_layers", "hidden", "intermediate", "n_heads", "n_kv_heads", "head_dim", "vocab"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if not math.isfinite(self.rope_theta) or self.rope_theta <= 0:
            raise ValueError("rope_theta must be finite and positive")
        if not math.isfinite(self.rms_eps) or self.rms_eps <= 0:
            raise ValueError("rms_eps must be finite and positive")
        if self.qk_norm != (self.arch == "qwen3"):
            raise ValueError("Qwen3 requires q/k RMSNorm; Llama does not use it")
        if self.rope_scaling:
            mode = self.rope_scaling.get("rope_type", self.rope_scaling.get("type", "default"))
            if mode not in ("default", "llama3"):
                raise ValueError(f"unsupported RoPE scaling: {mode}")

    @classmethod
    def from_dir(cls, path) -> "ModelConfig":
        c = json.loads((Path(path) / "config.json").read_text())
        arches = c.get("architectures", [])
        arch = _ARCH.get(arches[0]) if arches else c.get("model_type")
        if arch not in ("llama", "qwen3"):
            raise ValueError(f"unsupported architecture: {arches or c.get('model_type')}")
        if c.get("hidden_act", "silu") != "silu":
            raise ValueError("only SiLU gated MLPs are supported")
        if c.get("attention_bias", False) or c.get("mlp_bias", False):
            raise ValueError("biased attention/MLP projections are unsupported")
        if c.get("use_sliding_window", False) or c.get("sliding_window") is not None:
            raise ValueError("sliding-window attention is unsupported")
        if c.get("partial_rotary_factor", 1.0) != 1.0:
            raise ValueError("partial rotary embeddings are unsupported")
        heads = c["num_attention_heads"]
        return cls(
            arch=arch, n_layers=c["num_hidden_layers"], hidden=c["hidden_size"],
            intermediate=c["intermediate_size"], n_heads=heads,
            n_kv_heads=c.get("num_key_value_heads", heads),
            head_dim=c.get("head_dim") or c["hidden_size"] // heads,
            vocab=c["vocab_size"], rope_theta=float(c.get("rope_theta", 10000.0)),
            rope_scaling=c.get("rope_scaling"), rms_eps=float(c.get("rms_norm_eps", 1e-6)),
            tie_embeddings=bool(c.get("tie_word_embeddings", False)), qk_norm=arch == "qwen3",
        )


def rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    """HF default and Llama 3 wavelength-scaled rotary frequencies, in float32."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    s = cfg.rope_scaling
    if not s or s.get("rope_type", s.get("type", "default")) == "default":
        return inv
    try:
        factor, lo, hi, old = (float(s[k]) for k in (
            "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("incomplete Llama 3 RoPE scaling parameters") from exc
    if not all(math.isfinite(v) for v in (factor, lo, hi, old)) or factor < 1 or not 0 < lo < hi or old <= 0:
        raise ValueError("invalid Llama 3 RoPE scaling parameters")
    wavelength = 2 * math.pi / inv
    smooth = (old / wavelength - lo) / (hi - lo)
    medium = (1 - smooth) * inv / factor + smooth * inv
    return torch.where(wavelength > old / lo, inv / factor,
                       torch.where(wavelength < old / hi, inv, medium))


def load_weights(path, device="cuda", dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """Load safetensors only, one tensor at a time, validating a shard index first."""
    path = Path(path)
    index = path / "model.safetensors.index.json"
    mapping = None
    if index.is_file():
        mapping = json.loads(index.read_text()).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"empty or invalid weight_map in {index}")
        filenames = sorted(set(mapping.values()))
        if any(not isinstance(f, str) or Path(f).name != f or not f.endswith(".safetensors") for f in filenames):
            raise ValueError("safetensors shard names must be local filenames")
        files = [path / f for f in filenames]
    else:
        single = path / "model.safetensors"
        files = [single] if single.is_file() else sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights in {path}; pickle checkpoints are unsupported")
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"local snapshot is missing weight shards: {', '.join(missing)}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ImportError("serving weights require safetensors (env/requirements-serve.txt)") from exc
    out = {}
    for f in files:
        with safe_open(str(f), framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if mapping is not None and mapping.get(key) != f.name:
                    raise ValueError(f"weight index does not match tensor {key} in {f.name}")
                if key in out:
                    raise ValueError(f"duplicate weight tensor: {key}")
                tensor = tensors.get_tensor(key)
                out[key] = tensor.to(device=device, dtype=dtype if tensor.is_floating_point() else tensor.dtype)
    if mapping is not None and set(mapping) != set(out):
        raise ValueError(f"weight shards are missing indexed tensors: {sorted(set(mapping) - set(out))}")
    return out


def load_tokenizer(path):
    """Load a local tokenizers JSON; callers choose any model-specific chat template."""
    filename = Path(path) / "tokenizer.json"
    if not filename.is_file():
        raise FileNotFoundError(f"local tokenizer is missing: {filename}")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise ImportError("text prompts require tokenizers (env/requirements-serve.txt)") from exc
    return Tokenizer.from_file(str(filename))
