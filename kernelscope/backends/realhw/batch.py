"""In-process batch runner for the Nsight-free real-HW track (`kernelscope bench`).

`sweep` isolates every measurement in its own subprocess (needed for ncu, the NVBit tracer and
crashing kernels) at ~10 s per cell. `bench` measures every (workload, cache state) of each
plugin inside this process with the same row builders and schema, at well under a second per
cell. A Python exception in one cell is recorded and the batch continues; a hard crash
(segfault) ends the process, and `resume` skips cells already measured.
"""
import json
import os
import socket
import time
import uuid
from pathlib import Path

from kernelscope.backends.realhw.cache import IterationHooks
from kernelscope.backends.realhw.hygiene import gpu_contention, visible_device_index
from kernelscope.backends.realhw.latency import measure_latency
from kernelscope.backends.realhw.sweep import _row, analytic_rows, profile_rows
from kernelscope.check import check_outputs
from kernelscope.plugins.base import KernelPlugin
from kernelscope.run_kernel import _sync, profile_launches


def done_cells(summaries_path) -> set:
    p = Path(summaries_path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        s = json.loads(line)
        if s.get("status") == "ok":
            out.add((s["plugin"], s["workload_key"], s["cache_state"]))
    return out


def _free(device):
    if str(device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()


def run_bench(plugins, workloads, cache_states, store, summaries_path, *, device="cuda", warmup=10,
              iters=30, pad=5, ceilings=None, check_max_ref_bytes=1 << 30, atol=1e-2, resume=True,
              log=print) -> list[dict]:
    summaries_path = Path(summaries_path)
    summaries_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_cells(summaries_path) if resume else set()
    c = gpu_contention(visible_device_index(device), my_pids={os.getpid()}) or {}
    base_extra = {"run_id": time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6],
                  "host": socket.gethostname(), "device": device,
                  "gpu_util_at_start": c.get("utilization_pct"),
                  "other_pids_at_start": ",".join(str(p) for p in c.get("other_pids", [])),
                  "runner": "bench"}
    out = []

    def emit(s):
        out.append(s)
        with open(summaries_path, "a") as f:
            f.write(json.dumps(s) + "\n")
        if log:
            log(json.dumps(s))

    for plugin in plugins:
        for w in workloads:
            todo = [st for st in cache_states if (plugin.name, w.key(), st) not in done]
            for st in cache_states:
                if st not in todo:
                    out.append({"status": "skipped", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st})
            if not todo:
                continue
            if not isinstance(plugin, KernelPlugin) or not plugin.supports(w):
                for st in todo:
                    emit({"status": "unsupported", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
                          "reason": "executable plugin (use sweep)" if not isinstance(plugin, KernelPlugin)
                                    else "plugin.supports() is False"})
                continue
            _bench_workload(plugin, w, todo, store, emit, base_extra, device=device, warmup=warmup, iters=iters,
                            pad=pad, ceilings=ceilings, check_max_ref_bytes=check_max_ref_bytes, atol=atol)
    return out


def _bench_workload(plugin, w, states, store, emit, base_extra, *, device, warmup, iters, pad, ceilings,
                    check_max_ref_bytes, atol):
    inputs = None
    check = None
    try:
        inputs = plugin.build_inputs(w)
        # the float32 reference repeats K/V to every query head over the full cache capacity
        if 2 * w.B * w.L_kv * w.H_q * w.d * 4 <= check_max_ref_bytes:
            check = check_outputs(plugin, w, inputs, atol)
    except Exception as e:
        for st in states:
            emit({"status": "error", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
                  "error": f"{type(e).__name__}: {e}"[-2000:]})
        del inputs
        _free(device)
        return
    for i, st in enumerate(states):
        t0 = time.time()
        s = {"status": "ok", "plugin": plugin.name, "workload_key": w.key(), "cache_state": st,
             "check_ok": None if check is None else bool(check["ok"])}
        rows = []
        try:
            if i == 0 and check is not None:
                rows += [_row(w, plugin.name, "check", "max_abs_diff", check["max_abs_diff"], note=check.get("error")),
                         _row(w, plugin.name, "check", "ok", float(bool(check["ok"])))]
            hooks = IterationHooks(device, st)
            before = (lambda: (hooks.between(), _sync(device))) if st == "cold" else None
            lat = measure_latency(lambda: plugin.run(inputs), warmup=warmup, iters=iters, before=before)
            rows += [_row(w, plugin.name, "latency", "median_s", lat["median_s"], unit="s", note=lat["timer"]),
                     _row(w, plugin.name, "latency", "min_s", lat["min_s"], unit="s", note=lat["timer"])]
            for _ in range(warmup):
                hooks.between()
                plugin.run(inputs)
            _sync(device)
            prof = profile_launches(plugin, inputs, device, iters, hooks=hooks, pad=pad)
            kt = prof.get("kernel_time_us_median")
            rows += profile_rows(w, plugin.name, prof)
            rows += analytic_rows(w, plugin.name, plugin.kv_heads_read(w), kt, ceilings, cache_state=st)
            s.update(kernel_time_us=kt, latency_us=lat["median_s"] * 1e6, launches=prof["launches_per_iter"],
                     iterations_dropped=prof.get("iterations_dropped"))
        except Exception as e:
            s["status"] = "error"
            s["error"] = f"{type(e).__name__}: {e}"[-2000:]
        if rows:
            store.write(rows, tag=f"{plugin.name}_{w.key()}_{st}", extra={**base_extra, "cache_state": st})
        s["elapsed_s"] = round(time.time() - t0, 3)
        emit(s)
    del inputs
    _free(device)
