"""Plugin contracts. A plugin wraps an *existing* kernel; the harness never owns kernel code.

Both kinds share the metadata the backends need to isolate the kernel:
``kernel_regex`` (ncu ``-k regex:`` and the NVBit ``DYNAMIC_KERNEL_RANGE`` filter)
and ``num_launches`` (how many launches one ``run()`` issues, e.g. 2 for split-KV
attention + combine).
"""
from abc import ABC, abstractmethod
from typing import Any

from kernelscope.workload import Workload


class _PluginBase(ABC):
    name: str
    phases: frozenset
    kernel_regex: str | None      # None = not profilable as a single kernel (reference-only plugins)
    num_launches: int = 1         # hint only; the sweep counts real launches via `run_kernel --mode kernels`
    supports_ragged = False       # True only if build_inputs/run honour Workload.kv_lens

    def __init__(self, device: str = "cuda", **_):
        self.device = device

    def supports(self, w: Workload) -> bool:
        return w.phase in self.phases and (self.supports_ragged or not w.is_ragged)

    def kv_heads_read(self, w: Workload) -> int:
        """How many KV heads the kernel actually streams — feeds the analytic traffic model.
        Override when the kernel cannot do GQA and the plugin expands K/V to H_q."""
        return w.H_kv


class KernelPlugin(_PluginBase):
    """A kernel callable from Python (torch extension, Triton, library op)."""

    @abstractmethod
    def build_inputs(self, w: Workload) -> dict:
        """Materialize inputs in the plugin's own layout (seeded)."""

    @abstractmethod
    def run(self, inputs: dict) -> Any:
        """Issue the kernel once; may consist of ``num_launches`` launches."""

    @abstractmethod
    def to_dense_output(self, out: Any):
        """Convert ``run()``'s result to a dense ``[B, L_q, H_q, d]`` tensor for the reference check."""

    def to_dense_inputs(self, inputs: dict):
        """Return ``(q, k, v)`` as dense ``[B, L, H, d]`` tensors so the harness can compute the reference.

        Optional: a plugin that cannot provide this is still measured, but its
        correctness check is reported as an error.
        """
        raise NotImplementedError(f"{self.name}: to_dense_inputs not implemented; reference check unavailable")


class ExecutablePlugin(_PluginBase):
    """A standalone CUDA binary; ncu and the NVBit tracer wrap the process.

    The binary must print ``KERNELSCOPE {"kernel_time_us": <median>, ...}`` and, when
    ``out_path`` is given, write the dense output ``[B, L_q, H_q, d]`` as raw
    ``output_dtype``. ``reference_inputs(w)`` (optional) rebuilds the binary's inputs as
    dense torch tensors so the harness can check correctness.
    """
    output_dtype = "float32"
    reference_inputs = None      # override: (q, k, v) dense CPU tensors matching the binary's init

    @abstractmethod
    def command(self, w: Workload, iters: int = 1, out_path: str | None = None) -> list[str]:
        """argv that runs the kernel ``iters`` times for ``w`` (writing output to ``out_path`` if given)."""
