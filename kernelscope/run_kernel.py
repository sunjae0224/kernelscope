"""Single-cell subprocess entry point: ``python -m kernelscope.run_kernel``.

Each measurement runs in its own process so that ncu / the NVBit tracer can wrap
exactly one (plugin, workload) pair, and so a crashing kernel cannot take the sweep down.

Modes
  latency  CUDA-event median latency, printed as JSON on stdout
  check    reference-attention correctness, JSON on stdout
  kernels  names of CUDA kernels one ``run()`` launches (torch.profiler) and which
           match the plugin's ``kernel_regex`` — for plugin authors to find their regex
  profile  torch.profiler over ``iters`` runs: per-launch median kernel time, grid/block,
           registers, shared memory, occupancy estimate — the Nsight-free "counters"
  ncu      warm up, then run once more; prints nothing on stdout (ncu's CSV owns it).
           Optional: only works on hosts without a DCGM profiling lock
  trace    warm up with NVBit instrumentation off, then run once with it on (toggled through
           the injected tracer's enable/disable_nvbit_instrumentation) — JIT/autotune
           launches during warm-up are never recorded
"""
import argparse
import importlib
import json
import os
import re
import sys

from kernelscope.backends.realhw.latency import measure_latency
from kernelscope.check import check_plugin
from kernelscope.plugins.base import KernelPlugin
from kernelscope.workload import Workload

DEFAULT_REGISTRY = "kernelscope.plugins.builtin:REGISTRY"


def load_registry(spec: str):
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def _die(msg: str, code: int = 2):
    print(msg, file=sys.stderr)
    sys.exit(code)


def _sync(device: str):
    if device.startswith("cuda"):
        import torch
        torch.cuda.synchronize()


def _load_tracer(path):
    """Handle to the already-injected NVBit tracer (CUDA_INJECTION64_PATH), or None.
    dlopen of a loaded library returns the same image, so its exported
    enable/disable_nvbit_instrumentation() act on the live tracer state."""
    if not path:
        return None
    import ctypes
    try:
        lib = ctypes.CDLL(path)
        lib.enable_nvbit_instrumentation.restype = None
        lib.disable_nvbit_instrumentation.restype = None
        return lib
    except (OSError, AttributeError):
        return None


def launched_kernels(plugin, inputs, device: str) -> list[str]:
    import torch
    from torch.profiler import ProfilerActivity, profile
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        plugin.run(inputs)
        _sync(device)
    names = []
    for evt in prof.events():
        if getattr(evt, "device_type", None) == torch.autograd.DeviceType.CUDA:
            names.append(evt.name)
    return names


def profile_launches(plugin, inputs, device: str, iters: int) -> dict:
    """torch.profiler over ``iters`` runs -> per-launch median duration + geometry, and
    which launches match the plugin's kernel_regex (counter-free, DCGM-proof)."""
    import tempfile
    from torch.profiler import ProfilerActivity, profile
    from kernelscope.backends.realhw.kprofile import (
        kernel_events_from_chrome_trace, occupancy_estimate, props_from_torch, summarize_launches)
    activities = [ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        for _ in range(iters):
            plugin.run(inputs)
        _sync(device)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        path = tmp.name
    prof.export_chrome_trace(path)
    events = kernel_events_from_chrome_trace(path)
    os.unlink(path)
    summary = summarize_launches(events, plugin.kernel_regex, iters)
    props = props_from_torch(device) if device.startswith("cuda") else None
    if props:
        for l in summary["launches"]:
            l["occupancy"] = occupancy_estimate(l["grid"], l["block"], l["regs"], l["smem_bytes"], props)
    summary["device_props"] = props
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kernelscope.run_kernel")
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--workload", required=True, help="workload key, e.g. decode_B1_Lq1_Lkv4096_Hq32_Hkv8_d128_float16_causal")
    ap.add_argument("--mode", choices=["latency", "check", "kernels", "profile", "ncu", "trace"], required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--registry", default=DEFAULT_REGISTRY, help="module:attr of a PluginRegistry")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    registry = load_registry(args.registry)
    try:
        plugin = registry.get(args.plugin, device=args.device)
    except KeyError as e:
        _die(str(e))
    if not isinstance(plugin, KernelPlugin):
        _die(f"{args.plugin} is not a KernelPlugin; executables are driven by the sweep directly")
    w = Workload.from_key(args.workload)
    if not plugin.supports(w):
        _die(f"{plugin.name} does not support phase {w.phase!r} (supports {sorted(plugin.phases)})", 3)

    base = {"mode": args.mode, "plugin": plugin.name, "workload_key": w.key()}

    if args.mode == "check":
        print(json.dumps({**base, **check_plugin(plugin, w, atol=args.atol)}))
        return

    inputs = plugin.build_inputs(w)

    if args.mode == "latency":
        res = measure_latency(lambda: plugin.run(inputs), warmup=args.warmup, iters=args.iters)
        res.pop("samples_s")
        print(json.dumps({**base, **res}))
        return

    if args.mode == "kernels":
        for _ in range(args.warmup):
            plugin.run(inputs)
        names = launched_kernels(plugin, inputs, args.device)
        rx = re.compile(plugin.kernel_regex) if plugin.kernel_regex else None
        matching = [n for n in names if rx and rx.search(n)]
        print(json.dumps({**base, "kernel_regex": plugin.kernel_regex, "kernels": names, "matching": matching}))
        return

    if args.mode == "profile":
        for _ in range(args.warmup):
            plugin.run(inputs)
        _sync(args.device)
        summary = profile_launches(plugin, inputs, args.device, args.iters)
        print(json.dumps({**base, "kernel_regex": plugin.kernel_regex, "iters": args.iters, **summary}))
        return

    if args.mode == "ncu":
        for _ in range(args.warmup + 1):
            plugin.run(inputs)
        _sync(args.device)
        print(json.dumps({**base, "runs": args.warmup + 1, "warmup": args.warmup}), file=sys.stderr)
        return

    if args.mode == "trace":
        # warm-up (JIT, autotune) with instrumentation off; exactly one run with it on
        for _ in range(args.warmup):
            plugin.run(inputs)
        _sync(args.device)
        tracer = _load_tracer(os.environ.get("CUDA_INJECTION64_PATH"))
        if tracer is not None:
            tracer.enable_nvbit_instrumentation()
        plugin.run(inputs)
        _sync(args.device)
        if tracer is not None:
            tracer.disable_nvbit_instrumentation()
        print(json.dumps({**base, "runs": args.warmup + 1, "warmup": args.warmup,
                          "instrumentation_window": tracer is not None}), file=sys.stderr)
        return


if __name__ == "__main__":
    main()
