"""Real-hardware track orchestrator (Nsight-free by default).

For every (plugin, workload) cell it spawns `python -m kernelscope.run_kernel`
in isolated subprocesses:

  check     reference correctness
  latency   CUDA-event latency (launch overhead included — what a caller feels)
  profile   torch.profiler kernel time + grid/block/regs/smem + occupancy estimate
            (CUPTI activity API: no hardware-counter session, so it works next to DCGM)
  analytic  compulsory bytes / FLOPs -> achieved GB/s, TFLOPS, and utilisation
            against measured ceilings when provided (derived in-process, no subprocess)
  ncu       optional (``ncu_cmd``); launch-skip/count come from the profiled launch count

A crash only loses that cell; earlier measurements of the cell are still stored.
"""
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from kernelscope.analytic import arithmetic_intensity, attention_flops, attention_traffic
from kernelscope.backends.realhw.executable import check_executable, run_executable
from kernelscope.backends.realhw.hygiene import gpu_contention, visible_device_index
from kernelscope.backends.realhw.ncu import DEFAULT_METRICS, build_ncu_command, parse_ncu_csv
from kernelscope.plugins.base import ExecutablePlugin
from kernelscope.run_kernel import DEFAULT_REGISTRY, load_registry
from kernelscope.workload import Workload

UNSUPPORTED_EXIT = 3


def rows_from_ncu(parsed: list[dict], workload_key: str, plugin: str) -> list[dict]:
    return [{
        "workload_key": workload_key, "kernel": plugin, "backend": "ncu",
        "metric": r["metric"], "unit": r["unit"], "value": r["value"],
        "launch_idx": r["launch_idx"], "kernel_name": r["kernel_name"],
        "grid_size": r["grid_size"], "block_size": r["block_size"],
    } for r in parsed]


def profile_rows(w: Workload, plugin: str, prof: dict) -> list[dict]:
    rows = [_row(w, plugin, "profile", "launches_per_iter", prof["launches_per_iter"])]
    kt = prof.get("kernel_time_us_median")
    if kt is not None:
        rows.append(_row(w, plugin, "profile", "kernel_time_us", kt, unit="us"))
    for l in prof["launches"]:
        i, note = l["idx"], l["name"]
        rows += [
            _row(w, plugin, "profile", "dur_us", l["dur_us_median"], unit="us", launch_idx=i, note=note),
            _row(w, plugin, "profile", "grid_blocks", math.prod(l["grid"]), launch_idx=i, note=note),
            _row(w, plugin, "profile", "block_threads", math.prod(l["block"]), launch_idx=i, note=note),
            _row(w, plugin, "profile", "regs", l.get("regs") or 0, launch_idx=i, note=note),
            _row(w, plugin, "profile", "smem_bytes", l.get("smem_bytes") or 0, launch_idx=i, note=note),
        ]
        rows += [_row(w, plugin, "profile", k, v, launch_idx=i, note=note)
                 for k, v in (l.get("occupancy") or {}).items()]
    return rows


def analytic_rows(w: Workload, plugin: str, kv_heads_read: int, kernel_time_us, ceilings: dict | None) -> list[dict]:
    t = attention_traffic(w, kv_heads_read)
    f = attention_flops(w)
    rows = [_row(w, plugin, "analytic", "total_bytes", t["total_bytes"], unit="B"),
            _row(w, plugin, "analytic", "kv_bytes", t["kv_bytes"], unit="B"),
            _row(w, plugin, "analytic", "q_bytes", t["q_bytes"], unit="B"),
            _row(w, plugin, "analytic", "flops", f),
            _row(w, plugin, "analytic", "arithmetic_intensity", arithmetic_intensity(w, kv_heads_read), unit="flop/B"),
            _row(w, plugin, "analytic", "kv_heads_read", kv_heads_read)]
    if kernel_time_us:
        secs = kernel_time_us * 1e-6
        gbps = t["total_bytes"] / secs / 1e9
        tflops = f / secs / 1e12
        rows += [_row(w, plugin, "analytic", "achieved_gbps", gbps, unit="GB/s"),
                 _row(w, plugin, "analytic", "achieved_tflops", tflops, unit="TFLOP/s")]
        if ceilings:
            hbm = ceilings.get("hbm_gbps") or ceilings.get("hbm_copy_gbps")
            if hbm:
                rows.append(_row(w, plugin, "analytic", "dram_util", gbps / hbm))
            if ceilings.get("fp16_matmul_tflops"):
                rows.append(_row(w, plugin, "analytic", "tc_util", tflops / ceilings["fp16_matmul_tflops"]))
    return rows


class RealHWSweep:
    def __init__(self, store, python_exe=sys.executable, registry=DEFAULT_REGISTRY, device="cuda",
                 ncu_cmd=None, warmup=10, iters=50, metrics=None, atol=1e-2,
                 timeout_s=3600, run_id=None, ceilings=None):
        self.store = store
        self.python_exe = python_exe
        self.registry = registry
        self.device = device
        self.ncu_cmd = list(ncu_cmd) if ncu_cmd else None
        self.warmup = warmup
        self.iters = iters
        self.metrics = list(metrics) if metrics else DEFAULT_METRICS
        self.atol = atol
        self.timeout_s = timeout_s
        self.ceilings = dict(ceilings) if ceilings else None
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._registry_obj = None
        self.contention = gpu_contention(visible_device_index(device), my_pids={os.getpid()})

    def _extra(self) -> dict:
        c = self.contention or {}
        return {"run_id": self.run_id, "host": socket.gethostname(), "device": self.device,
                "gpu_util_at_start": c.get("utilization_pct"),
                "other_pids_at_start": ",".join(str(p) for p in c.get("other_pids", []))}

    def _annotate(self, summary: dict):
        if self.contention and self.contention.get("busy"):
            summary["gpu_contention"] = self.contention["note"]

    # ---- subprocess plumbing -------------------------------------------------

    def _plugin_meta(self, name: str):
        if self._registry_obj is None:
            self._registry_obj = load_registry(self.registry)
        return self._registry_obj.get(name, device=self.device)

    def _run_kernel_argv(self, plugin: str, w: Workload, mode: str) -> list[str]:
        return [self.python_exe, "-m", "kernelscope.run_kernel",
                "--plugin", plugin, "--workload", w.key(), "--mode", mode,
                "--registry", self.registry, "--device", self.device,
                "--warmup", str(self.warmup), "--iters", str(self.iters), "--atol", str(self.atol)]

    def _run_json(self, argv: list[str]) -> dict:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            raise _CellError(proc.returncode, proc.stderr.strip()[-2000:])
        return json.loads(proc.stdout)

    def profile(self, plugin: str, w: Workload) -> dict:
        return self._run_json(self._run_kernel_argv(plugin, w, "profile"))

    # ---- one cell ------------------------------------------------------------

    def run_cell(self, plugin: str, w: Workload) -> dict:
        t0 = time.time()
        summary = {"status": "ok", "plugin": plugin, "workload_key": w.key()}
        self._annotate(summary)
        meta = self._plugin_meta(plugin)
        if not meta.supports(w):
            summary["status"] = "unsupported"
            return summary
        extra = self._extra()
        if isinstance(meta, ExecutablePlugin):
            return self._run_executable_cell(meta, w, summary, extra, t0)
        rows = []
        try:
            chk = self._run_json(self._run_kernel_argv(plugin, w, "check"))
            summary["check_ok"] = chk["ok"]
            summary["max_abs_diff"] = chk["max_abs_diff"]
            rows += [_row(w, plugin, "check", "max_abs_diff", chk["max_abs_diff"], note=chk.get("error")),
                     _row(w, plugin, "check", "ok", float(bool(chk["ok"])))]

            lat = self._run_json(self._run_kernel_argv(plugin, w, "latency"))
            summary["median_s"] = lat["median_s"]
            rows += [_row(w, plugin, "latency", "median_s", lat["median_s"], unit="s", note=lat["timer"]),
                     _row(w, plugin, "latency", "min_s", lat["min_s"], unit="s", note=lat["timer"])]

            prof = self.profile(plugin, w)
            launches = prof["launches_per_iter"]
            kt = prof.get("kernel_time_us_median")
            summary["launches"] = launches
            summary["kernel_time_us"] = kt
            rows += profile_rows(w, plugin, prof)
            rows += analytic_rows(w, plugin, meta.kv_heads_read(w), kt, self.ceilings)

            if self.ncu_cmd is None:
                summary["ncu"] = "disabled"
            elif not launches:
                summary["ncu"] = "skipped: kernel_regex matched 0 launches"
            else:
                rows += self._ncu_rows(plugin, meta.kernel_regex, w, launches)
                summary["ncu"] = "ok"
        except _CellError as e:
            summary["status"] = "error"
            summary["error"] = f"exit {e.code}: {e.stderr}"
        except subprocess.TimeoutExpired:
            summary["status"] = "error"
            summary["error"] = f"timeout after {self.timeout_s}s"
        if rows:
            self.store.write(rows, tag=f"{plugin}_{w.key()}", extra=extra)
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary

    def _run_executable_cell(self, meta, w: Workload, summary: dict, extra: dict, t0: float) -> dict:
        """Binary path: correctness via the written output, timing self-reported by the binary."""
        plugin = meta.name
        rows = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                chk = check_executable(meta, w, Path(tmp) / "out.bin", atol=self.atol)
            summary["check_ok"] = chk["ok"]
            summary["max_abs_diff"] = chk["max_abs_diff"]
            rows += [_row(w, plugin, "check", "max_abs_diff", chk["max_abs_diff"], note=chk.get("error")),
                     _row(w, plugin, "check", "ok", float(bool(chk["ok"])))]

            rep = run_executable(meta, w, iters=self.iters, timeout_s=self.timeout_s)
            kt = rep.get("kernel_time_us")
            launches = int(rep.get("launches_per_iter") or 0)
            summary.update(kernel_time_us=kt, launches=launches, profile="self-reported")
            rows.append(_row(w, plugin, "profile", "launches_per_iter", launches))
            if kt is not None:
                rows.append(_row(w, plugin, "profile", "kernel_time_us", kt, unit="us", note="self-reported"))
            rows += analytic_rows(w, plugin, meta.kv_heads_read(w), kt, self.ceilings)

            if self.ncu_cmd is None:
                summary["ncu"] = "disabled"
            elif not launches:
                summary["ncu"] = "skipped: launches_per_iter unknown"
            else:
                rows += self._ncu_rows(plugin, meta.kernel_regex, w, launches,
                                       target=meta.command(w, iters=1), launch_skip=0)
                summary["ncu"] = "ok"
        except (RuntimeError, ValueError) as e:
            summary["status"] = "error"
            summary["error"] = str(e)
        except subprocess.TimeoutExpired:
            summary["status"] = "error"
            summary["error"] = f"timeout after {self.timeout_s}s"
        if rows:
            self.store.write(rows, tag=f"{plugin}_{w.key()}", extra=extra)
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary

    def _ncu_rows(self, plugin: str, kernel_regex: str, w: Workload, launches: int,
                  target=None, launch_skip=None) -> list[dict]:
        cmd = build_ncu_command(
            target=target or self._run_kernel_argv(plugin, w, "ncu"),
            kernel_regex=kernel_regex, num_launches=launches,
            launch_skip=self.warmup * launches if launch_skip is None else launch_skip,
            metrics=self.metrics, ncu_exe=self.ncu_cmd,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        if proc.returncode != 0:
            # ncu reports driver/permission problems as ==ERROR== lines on stdout
            errs = [ln for ln in proc.stdout.splitlines() if ln.startswith("==ERROR==")]
            raise _CellError(proc.returncode, ("\n".join(errs) + "\n" + proc.stderr).strip()[-2000:])
        return rows_from_ncu(parse_ncu_csv(proc.stdout), w.key(), plugin)

    # ---- grid ----------------------------------------------------------------

    def run_grid(self, plugins: list[str], workloads: list[Workload], log=print) -> list[dict]:
        summaries = []
        if log and self.contention and self.contention.get("busy"):
            log(json.dumps({"warning": "measuring on a busy GPU — kernel times will be inflated",
                            "gpu_contention": self.contention["note"]}))
        for w in workloads:
            for p in plugins:
                s = self.run_cell(p, w)
                summaries.append(s)
                if log:
                    log(json.dumps(s))
        return summaries


class _CellError(Exception):
    def __init__(self, code, stderr):
        super().__init__(f"exit {code}: {stderr}")
        self.code, self.stderr = code, stderr


def _row(w, plugin, backend, metric, value, unit="", launch_idx=0, note=None):
    return {"workload_key": w.key(), "kernel": plugin, "backend": backend, "metric": metric,
            "unit": unit, "value": float(value), "launch_idx": launch_idx, "note": note}
