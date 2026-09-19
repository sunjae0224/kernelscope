"""Fitted constants of the surrogate model, per cache state and kernel kind (spec §3.2)."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

KINDS = ("nonsplit", "split", "split_paged")


@dataclass(frozen=True)
class KindParams:
    cost_us_per_key: float   # one CTA alone on its SM, per key of its KV head
    t0_us: float             # fixed startup of a CTA with work
    t_empty_us: float        # a CTA whose split range is past its sequence's length
    gamma: float             # per-CTA speed when two CTAs share an SM (spec F19)
    t_fixed_us: float        # per-launch constant


@dataclass(frozen=True)
class StateParams:
    kinds: dict              # kind -> KindParams
    comb_a_us: float         # combine kernel: a + b * (B * H_q * splits), only when splits > 1
    comb_b_us: float


@dataclass(frozen=True)
class ModelParams:
    machine: str
    states: dict             # "cold" | "warm" -> StateParams

    def to_json(self, path) -> None:
        d = {"machine": self.machine, "states": {s: {"kinds": {k: asdict(v) for k, v in sp.kinds.items()},
                                                     "comb_a_us": sp.comb_a_us, "comb_b_us": sp.comb_b_us}
                                                 for s, sp in self.states.items()}}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path) -> "ModelParams":
        d = json.loads(Path(path).read_text())
        return cls(machine=d["machine"], states={
            s: StateParams(kinds={k: KindParams(**v) for k, v in sp["kinds"].items()},
                           comb_a_us=sp["comb_a_us"], comb_b_us=sp["comb_b_us"])
            for s, sp in d["states"].items()})


def _state(c_ns, c_sp, t0_ns, t0_sp, gamma, comb_a, comb_b):
    ns = KindParams(c_ns, t0_ns, 0.0, gamma, 0.0)
    sp = KindParams(c_sp, t0_sp, 0.5, gamma, 0.0)
    return StateParams(kinds={"nonsplit": ns, "split": sp, "split_paged": sp}, comb_a_us=comb_a, comb_b_us=comb_b)


# Initial values from the 2026-09-19 spike fit (gamma clamped to <= 1; see spec F19).
SPIKE_DEFAULTS = ModelParams(machine="NVIDIA GeForce RTX 4090", states={
    "cold": _state(0.0544, 0.0242, 12.0, 0.83, 0.5, 14.3, 0.00104),
    "warm": _state(0.0562, 0.0282, 3.3, 0.07, 0.53, 6.95, 0.0017),
})
