"""Logical attention workload — layout-agnostic; each plugin materializes its own tensors."""
import re
from dataclasses import dataclass
from itertools import product

PHASES = ("prefill", "decode")


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

    def __post_init__(self):
        if self.phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {self.phase!r}")
        for name in ("B", "L_q", "L_kv", "H_q", "H_kv", "d"):
            v = getattr(self, name)
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.H_q % self.H_kv:
            raise ValueError(f"H_q ({self.H_q}) must be a multiple of H_kv ({self.H_kv})")

    def key(self) -> str:
        return (
            f"{self.phase}_B{self.B}_Lq{self.L_q}_Lkv{self.L_kv}"
            f"_Hq{self.H_q}_Hkv{self.H_kv}_d{self.d}_{self.dtype}"
            f"_{'causal' if self.causal else 'noncausal'}"
        )

    @classmethod
    def from_key(cls, key: str) -> "Workload":
        m = _KEY_RE.fullmatch(key)
        if not m:
            raise ValueError(f"malformed workload key: {key!r}")
        g = m.groupdict()
        return cls(
            phase=g["phase"], B=int(g["B"]), L_q=int(g["L_q"]), L_kv=int(g["L_kv"]),
            H_q=int(g["H_q"]), H_kv=int(g["H_kv"]), d=int(g["d"]),
            dtype=g["dtype"], causal=g["causal"] == "causal",
        )


_KEY_RE = re.compile(
    r"(?P<phase>prefill|decode)_B(?P<B>\d+)_Lq(?P<L_q>\d+)_Lkv(?P<L_kv>\d+)"
    r"_Hq(?P<H_q>\d+)_Hkv(?P<H_kv>\d+)_d(?P<d>\d+)_(?P<dtype>[a-z0-9]+)_(?P<causal>causal|noncausal)"
)


def expand_grid(spec: dict) -> list[Workload]:
    """Cartesian product over list-valued fields, in declaration order.

    ``L`` is shorthand: it sets ``L_kv`` and derives ``L_q`` from the phase
    (1 for decode, ``L`` for prefill) unless those are given explicitly.
    """
    items = [(k, list(v) if isinstance(v, (list, tuple)) else [v]) for k, v in spec.items()]
    out = []
    for combo in product(*(vals for _, vals in items)):
        kw = dict(zip((k for k, _ in items), combo))
        if "L" in kw:
            L = kw.pop("L")
            kw.setdefault("L_kv", L)
            kw.setdefault("L_q", 1 if kw["phase"] == "decode" else L)
        out.append(Workload(**kw))
    return out
