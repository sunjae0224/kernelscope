"""Is anyone else on the GPU we are about to measure on?

A shared box will happily time-slice our kernels with a neighbour's training job and
hand back kernel times that are several times too long, with no error anywhere. So
the sweep asks nvidia-smi once at start, stores the answer next to every row, and warns.
"""
import csv
import io
import os
import subprocess

_QUERY = object()   # sentinel: "run nvidia-smi"; an explicit None means "unavailable"
BUSY_UTIL_PCT = 20


def _nvidia_smi(*args):
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None


def parse_gpu_query(text: str) -> list[dict]:
    out = []
    for rec in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        rec = {k.strip(): v.strip() for k, v in rec.items()}
        out.append({
            "index": int(rec["index"]), "uuid": rec["uuid"],
            "utilization_pct": int(rec["utilization.gpu [%]"].split()[0]),
            "memory_used_mib": int(rec["memory.used [MiB]"].split()[0]),
            "power_w": float(rec["power.draw [W]"].split()[0]),
        })
    return out


def parse_apps(text: str) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for rec in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        rec = {k.strip(): v.strip() for k, v in rec.items()}
        out.setdefault(rec["gpu_uuid"], []).append(int(rec["pid"]))
    return out


def visible_device_index(device: str) -> int | None:
    """Physical GPU index behind a torch device string, honouring CUDA_VISIBLE_DEVICES."""
    if not device.startswith("cuda"):
        return None
    ordinal = int(device.split(":")[1]) if ":" in device else 0
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        ids = [x.strip() for x in cvd.split(",") if x.strip()]
        try:
            return int(ids[ordinal])
        except (IndexError, ValueError):
            return None
    return ordinal


def gpu_contention(index: int | None, gpu_query_text=_QUERY, apps_text=_QUERY, my_pids=None) -> dict:
    if index is None:
        return {"busy": None, "note": "no GPU index"}
    if gpu_query_text is _QUERY:
        gpu_query_text = _nvidia_smi("--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw", "--format=csv")
    if apps_text is _QUERY:
        apps_text = _nvidia_smi("--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv")
    if not gpu_query_text:
        return {"busy": None, "note": "nvidia-smi unavailable"}
    gpus = {g["index"]: g for g in parse_gpu_query(gpu_query_text)}
    if index not in gpus:
        return {"busy": None, "note": f"GPU {index} not listed by nvidia-smi"}
    g = gpus[index]
    apps = parse_apps(apps_text) if apps_text else {}
    mine = set(my_pids or ())
    other = sorted(p for p in apps.get(g["uuid"], []) if p not in mine)
    busy = bool(other) or (g["utilization_pct"] >= BUSY_UTIL_PCT and not mine & set(apps.get(g["uuid"], [])))
    return {"busy": busy, "index": index, "uuid": g["uuid"], "utilization_pct": g["utilization_pct"],
            "memory_used_mib": g["memory_used_mib"], "other_pids": other,
            "note": (f"GPU {index} busy: {g['utilization_pct']}% util, other pids {other}" if busy else "idle")}
