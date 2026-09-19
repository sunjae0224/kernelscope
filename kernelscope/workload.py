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
