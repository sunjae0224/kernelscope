"""Measured machine parameters (machines/<gpu>.json) and hypothetical variants of them.

A hypothetical machine scales one resource of the measured one: SM count, DRAM bandwidth, L2
capacity (the measured hit-rate curve is stretched along the working-set axis) or shared memory
per SM (which changes how many CTAs of a kernel fit on an SM).
"""
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

MIB = 2**20


@dataclass(frozen=True)
class MachineSpec:
    name: str
    n_sm: int
    max_threads_sm: int
    max_ctas_sm: int
    regs_sm: int
    smem_sm: int
    reserved_smem_per_block: int
    l2_bytes: int
    dram_gbps: float
    l2_gbps: float
    l2_curve: tuple            # ((working_set_bytes, hit), ...) measured at l2_curve_ref_bytes
    l2_curve_ref_bytes: int

    @classmethod
    def from_json(cls, path) -> "MachineSpec":
        d = json.loads(Path(path).read_text())
        curve = tuple((float(p["mib"]) * MIB, float(p["hit"])) for p in d["l2_hit_curve"])
        return cls(name=d["name"], n_sm=d["n_sm"], max_threads_sm=d["max_threads_sm"], max_ctas_sm=d["max_ctas_sm"],
                   regs_sm=d["regs_sm"], smem_sm=d["smem_sm"], reserved_smem_per_block=d["reserved_smem_per_block"],
                   l2_bytes=d["l2_bytes"], dram_gbps=float(d["dram_gbps"]), l2_gbps=float(d["l2_gbps"]),
                   l2_curve=curve, l2_curve_ref_bytes=d["l2_bytes"])

    def props(self) -> dict:
        return {"num_sms": self.n_sm, "max_threads_per_sm": self.max_threads_sm, "regs_per_sm": self.regs_sm,
                "smem_per_sm": self.smem_sm, "max_blocks_per_sm": self.max_ctas_sm,
                "reserved_smem_per_block": self.reserved_smem_per_block, "warp_size": 32,
                "max_warps_per_sm": self.max_threads_sm // 32}

    def sm_order(self) -> np.ndarray:
        """Order in which the first wave of blocks lands on SMs (measured: even SMs, then odd; spec F19)."""
        return np.concatenate([np.arange(0, self.n_sm, 2), np.arange(1, self.n_sm, 2)])

    def l2_hit(self, working_set_bytes: float) -> float:
        """Share of streamed bytes served by L2 for a working set, from the measured curve."""
        x = working_set_bytes * self.l2_curve_ref_bytes / self.l2_bytes
        xs = [p[0] for p in self.l2_curve]
        hs = [p[1] for p in self.l2_curve]
        return float(np.interp(x, xs, hs, left=hs[0], right=hs[-1]))

    def scaled(self, sm: float = 1.0, dram: float = 1.0, l2: float = 1.0, smem: float = 1.0) -> "MachineSpec":
        tags = [f"{k}={v:g}" for k, v in (("sm", sm), ("dram", dram), ("l2", l2), ("smem", smem)) if v != 1.0]
        return replace(self, name=self.name + (" [" + ",".join(tags) + "]" if tags else ""),
                       n_sm=max(1, round(self.n_sm * sm)), dram_gbps=self.dram_gbps * dram,
                       l2_bytes=round(self.l2_bytes * l2), smem_sm=round(self.smem_sm * smem))
