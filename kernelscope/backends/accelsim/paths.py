"""Where the built Accel-Sim tree keeps its tools (layout of accel-sim-framework v2.0.0)."""
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = os.environ.get("ACCELSIM_ROOT", "/home/skkai/accelsim/accel-sim-framework")


@dataclass(frozen=True)
class AccelSimPaths:
    root: Path = Path(DEFAULT_ROOT)

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @property
    def tracer_so(self) -> Path:
        return self.root / "util/tracer_nvbit/tracer_tool/tracer_tool.so"

    @property
    def post_process(self) -> Path:
        return self.root / "util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing"

    @property
    def sim_bin(self) -> Path:
        return self.root / "gpu-simulator/bin/release/accel-sim.out"

    def gpgpusim_config(self, arch: str) -> Path:
        return self.root / f"gpu-simulator/gpgpu-sim/configs/tested-cfgs/{arch}/gpgpusim.config"

    def trace_config(self, arch: str) -> Path:
        return self.root / f"gpu-simulator/configs/tested-cfgs/{arch}/trace.config"
