"""Simulation track orchestrator: trace once per cell, simulate once per what-if variant.

Pipeline per (plugin, workload):
  1. trace   — run_kernel --mode ncu --warmup 0 under the NVBit tracer, regex-filtered
  2. budget  — warp-instruction count from stats_ctx_* -> estimated sim seconds; skip if over
  3. post    — post-traces-processing -> traces/kernelslist.g
  4. sim     — accel-sim.out per variant config, stdout parsed into rows

The three external steps are injectable (``trace_fn``/``postprocess_fn``/``simulate_fn``)
so the orchestration is testable without the tools.
"""
import json
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from kernelscope.analysis.trace_mix import instruction_mix
from kernelscope.backends.accelsim.config import write_variant_config
from kernelscope.backends.accelsim.paths import AccelSimPaths
from kernelscope.backends.accelsim.stats import parse_sim_stdout
from kernelscope.backends.accelsim.trace import (
    PLANNING_RATE, estimate_sim_seconds, filter_kernelslist, find_kernelslist, find_trace_stats,
    read_trace_stats, tracer_env,
)
from kernelscope.plugins.base import ExecutablePlugin
from kernelscope.run_kernel import DEFAULT_REGISTRY, load_registry
from kernelscope.workload import Workload


def trace_command(python_exe, plugin, w: Workload, registry, device, warmup: int = 3) -> list[str]:
    return [python_exe, "-m", "kernelscope.run_kernel", "--plugin", plugin, "--workload", w.key(),
            "--mode", "trace", "--warmup", str(warmup), "--registry", registry, "--device", device]


def postprocess_command(paths: AccelSimPaths, traces_dir, jobs: int = 8) -> list[str]:
    return [str(paths.post_process), str(traces_dir), "-j", str(jobs)]


def simulate_command(paths: AccelSimPaths, kernelslist, gpgpusim_config, trace_config) -> list[str]:
    return [str(paths.sim_bin), "-trace", str(kernelslist),
            "-config", str(gpgpusim_config), "-config", str(trace_config)]


class AccelSimSweep:
    def __init__(self, store, paths: AccelSimPaths, work_dir, python_exe=sys.executable,
                 registry=DEFAULT_REGISTRY, device="cuda", device_index=0, arch="SM80_A100",
                 variants=("base",), max_sim_s=None, sim_rate=PLANNING_RATE,
                 trace_timeout_s=3600, sim_timeout_s=24 * 3600, run_id=None, trace_warmup=3,
                 trace_fn=None, postprocess_fn=None, simulate_fn=None):
        self.store = store
        self.paths = paths
        self.work_dir = Path(work_dir)
        self.python_exe = python_exe
        self.registry = registry
        self.device = device
        self.device_index = device_index
        self.arch = arch
        self.variants = list(variants)
        self.max_sim_s = max_sim_s
        self.sim_rate = sim_rate
        self.trace_timeout_s = trace_timeout_s
        self.sim_timeout_s = sim_timeout_s
        self.trace_warmup = trace_warmup
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._trace = trace_fn or self._run_trace
        self._postprocess = postprocess_fn or self._run_postprocess
        self._simulate = simulate_fn or self._run_simulate
        self._registry_obj = None

    # ---- default subprocess implementations ------------------------------------

    def _run_trace(self, argv, env, cell_dir):
        proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=self.trace_timeout_s)
        (cell_dir / "trace.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"tracer run exit {proc.returncode}: {proc.stderr.strip()[-1500:]}")

    def _run_postprocess(self, argv, cell_dir):
        proc = subprocess.run(argv, cwd=cell_dir, capture_output=True, text=True, timeout=self.trace_timeout_s)
        (cell_dir / "postprocess.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"post-traces-processing exit {proc.returncode}: {proc.stderr.strip()[-1500:]}")

    def _run_simulate(self, argv, log_path):
        t0 = time.time()
        try:
            proc = subprocess.run(argv, cwd=log_path.parent, capture_output=True, text=True,
                                  timeout=self.sim_timeout_s)
            out, err = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = f"TIMEOUT after {self.sim_timeout_s}s"
        log_path.write_text(out + "\n--- stderr ---\n" + err)
        return out, time.time() - t0

    # ---- one cell ------------------------------------------------------------

    def _plugin_meta(self, name):
        if self._registry_obj is None:
            self._registry_obj = load_registry(self.registry)
        return self._registry_obj.get(name, device=self.device)

    def run_cell(self, plugin: str, w: Workload) -> dict:
        t0 = time.time()
        summary = {"status": "ok", "plugin": plugin, "workload_key": w.key(), "sim": {}}
        meta = self._plugin_meta(plugin)
        if not meta.supports(w):
            summary["status"] = "unsupported"
            return summary
        if not meta.kernel_regex:
            summary["status"] = "unsupported"
            summary["error"] = "plugin has no kernel_regex (not traceable as a single kernel)"
            return summary

        cell_dir = self.work_dir / self.run_id / f"{plugin}__{w.key()}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        extra = {"run_id": self.run_id, "host": socket.gethostname(), "arch": self.arch}
        rows = []
        try:
            if isinstance(meta, ExecutablePlugin):
                # the tracer wraps the binary directly; regex filter from the first launch
                env = tracer_env(self.paths, cell_dir, meta.kernel_regex, self.device_index)
                argv = meta.command(w, iters=1)
            else:
                # warm-up (JIT/autotune) untraced; one run inside the cudaProfilerStart/Stop window
                env = tracer_env(self.paths, cell_dir, meta.kernel_regex, self.device_index, window=True)
                argv = trace_command(self.python_exe, plugin, w, self.registry, self.device, self.trace_warmup)
            self._trace(argv, env, cell_dir)
            listed = [k for p in find_trace_stats(cell_dir) for k in read_trace_stats(p)]
            if not listed:
                raise RuntimeError("tracer produced no trace (kernel_regex matched no launch?)")
            # stats_ctx lists every launch in the process; keep instrumented launches whose
            # name matches the plugin's regex (window mode traces everything in the window)
            rx = re.compile(meta.kernel_regex)
            stats = [k for k in listed if k["warp_insts"] > 0 and rx.search(k["kernel_name"])]
            warp_insts = sum(k["warp_insts"] for k in stats)
            if warp_insts == 0:
                names = ", ".join(k["kernel_name"][:60] for k in listed)
                raise RuntimeError(f"tracer recorded 0 warp instructions for launches [{names}] "
                                   f"(kernel_regex {meta.kernel_regex!r} matched nothing?)")
            est = estimate_sim_seconds(warp_insts, self.sim_rate)
            summary.update(warp_insts=warp_insts, est_sim_s=est, n_kernels=len(stats))
            rows += [_row(w, plugin, "trace", "warp_insts", warp_insts),
                     _row(w, plugin, "trace", "est_sim_s", est, unit="s"),
                     _row(w, plugin, "trace", "n_kernels", len(stats))]
            rows += [_row(w, plugin, "trace", "kernel_warp_insts", k["warp_insts"], launch_idx=i,
                          note=k["kernel_name"]) for i, k in enumerate(stats)]
            rows += self._tracemix_rows(plugin, w, cell_dir, stats)
            if self.max_sim_s is not None and est > self.max_sim_s:
                summary["status"] = "skipped_budget"
            else:
                self._postprocess(postprocess_command(self.paths, cell_dir / "traces"), cell_dir)
                kl = find_kernelslist(cell_dir)
                if kl is None:
                    raise RuntimeError("post-processing produced no kernelslist.g")
                filter_kernelslist(kl, [k["trace_file"] for k in stats])
                for v in self.variants:
                    rows += self._simulate_variant(plugin, w, v, cell_dir, kl, summary)
        except (RuntimeError, subprocess.TimeoutExpired, OSError) as e:
            summary["status"] = "error"
            summary["error"] = str(e)
        if rows:
            self.store.write(rows, tag=f"sim_{plugin}_{w.key()}", extra=extra)
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary

    def _tracemix_rows(self, plugin, w, cell_dir, stats) -> list[dict]:
        """Opcode mix from the raw per-kernel trace, when the tracer left it behind."""
        rows = []
        for i, k in enumerate(stats):
            raw = cell_dir / "traces" / k["trace_file"]
            if not raw.exists():
                continue
            mix = instruction_mix(raw)
            note = k["kernel_name"]
            rows.append(_row(w, plugin, "tracemix", "n_warp_insts", mix["n_warp_insts"], launch_idx=i, note=note))
            rows += [_row(w, plugin, "tracemix", f"frac_{c}", f, launch_idx=i, note=note)
                     for c, f in mix["fractions"].items()]
            rows += [_row(w, plugin, "tracemix", m, mix[m], unit="B", launch_idx=i, note=note)
                     for m in ("global_load_bytes", "global_store_bytes")]
        return rows

    def _simulate_variant(self, plugin, w, variant, cell_dir, kernelslist, summary) -> list[dict]:
        cfg = write_variant_config(self.paths.gpgpusim_config(self.arch), cell_dir / "configs", variant)
        log_path = cell_dir / f"sim_{variant}.log"
        stdout, wall = self._simulate(
            simulate_command(self.paths, kernelslist, cfg, self.paths.trace_config(self.arch)), log_path)
        parsed = parse_sim_stdout(stdout)
        status = parsed["status"]
        if parsed["unsupported_opcode"]:
            status = f"unsupported_opcode:{parsed['unsupported_opcode']}"
        summary["sim"][variant] = status
        backend = f"sim:{variant}"
        rows = [_row(w, plugin, backend, "status", 1.0 if status == "ok" else 0.0, note=status),
                _row(w, plugin, backend, "sim_wall_s", wall, unit="s")]
        rows += [_row(w, plugin, backend, k, v) for k, v in parsed["totals"].items()]
        for i, k in enumerate(parsed["kernels"]):
            for key in ("gpu_sim_cycle", "gpu_sim_insn"):
                if key in k:
                    rows.append(_row(w, plugin, backend, key, k[key], launch_idx=i, note=k["kernel_name"]))
        return rows

    def run_grid(self, plugins, workloads, log=print) -> list[dict]:
        out = []
        for w in workloads:
            for p in plugins:
                s = self.run_cell(p, w)
                out.append(s)
                if log:
                    log(json.dumps(s))
        return out


def _row(w, plugin, backend, metric, value, unit="", launch_idx=0, note=None):
    return {"workload_key": w.key(), "kernel": plugin, "backend": backend, "metric": metric,
            "unit": unit, "value": float(value), "launch_idx": launch_idx, "note": note}
