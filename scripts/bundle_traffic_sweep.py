"""Copy the load-sweep replays' summary.json into demo_data/traffic_sweep/<run>/ and write sweep_table.csv.

    python -m scripts.bundle_traffic_sweep ../kernelscope/results/traffic_sweep_20261024 demo_data

GPU-free. Also records each copied file in demo_data/provenance.json (replacing earlier entries of the same path).
"""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pandas as pd


def flatten(run, s):
    losses, load, fixed = s["losses"], s.get("load", {}), s.get("fixed", {})
    row = {"run": run, "trace": run.rsplit("_x", 1)[0], "rate_scale": s["config"]["rate_scale"], "steps": s["steps"],
           "requests": s["trace"]["requests"], "batch_mean": losses["batch_mean"], "no_split_frac": losses["no_split_frac"],
           "loss_ge_1.1_frac": losses["loss_ge_1.1_frac"], "loss_ge_1.25_frac": losses["loss_ge_1.25_frac"],
           "loss_ge_1.5_frac": losses["loss_ge_1.5_frac"], "heuristic_ratio": losses["attention_time_ratio"]}
    row.update({f"load.{k}": v for k, v in load.items()})
    row.update({f"fixed.{k}": v for k, v in fixed.items()})
    return row


BINS = [0, 12, 25, 29, 34, 39, 44, 49, 64]


def conditional(losses):
    """Share of steps with batch >= 26 and, among those, the share with regret >= 1.25."""
    ge = losses.B >= 26
    return {"b_ge_26_frac": float(ge.mean()), "loss_ge_1.25_given_b_ge_26": float((losses.loss[ge] >= 1.25).mean()) if ge.any() else None}


def batch_bins(src, runs):
    """Pooled per-step regret by batch size over ``runs`` (steps are independent of the arrival scale given the batch)."""
    frame = pd.concat([pd.read_parquet(Path(src) / r / "losses.parquet", columns=["B", "loss"]) for r in runs])
    frame["bin"] = pd.cut(frame.B, BINS)
    out = []
    for interval, g in frame.groupby("bin", observed=True):
        out.append({"b_lo": int(interval.left) + 1, "b_hi": int(interval.right), "steps": int(len(g)),
                    "loss_ge_1.25_frac": float((g.loss >= 1.25).mean()), "loss_median": float(g.loss.median())})
    return {"runs": list(runs), "bins": out}


def main(src, data):
    src, data = Path(src), Path(data)
    out = data / "traffic_sweep"
    rows, entries = [], []
    for path in sorted(src.glob("*/summary.json")):
        run = path.parent.name
        (out / run).mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out / run / "summary.json")
        row = flatten(run, json.loads(path.read_text()))
        row.update({f"cond.{k}": v for k, v in conditional(pd.read_parquet(path.parent / "losses.parquet", columns=["B", "loss"])).items()})
        rows.append(row)
        blob = path.read_bytes()
        entries.append({"path": f"traffic_sweep/{run}/summary.json", "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    table = pd.DataFrame(rows).sort_values(["trace", "rate_scale"]).reset_index(drop=True)
    table.to_csv(out / "sweep_table.csv", index=False)
    (out / "batch_bins.json").write_text(json.dumps(batch_bins(src, ["azure_conv_x1", "azure_conv_x2"]), indent=2) + "\n")
    for name in ("sweep_table.csv", "batch_bins.json"):
        blob = (out / name).read_bytes()
        entries.append({"path": f"traffic_sweep/{name}", "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()})
    prov_path = data / "provenance.json"
    prov = json.loads(prov_path.read_text())
    kept = [f for f in prov["files"] if not f["path"].startswith("traffic_sweep/")]
    prov["files"] = kept + entries
    note = "traffic_sweep/*/summary.json are surrogate-model replays at many arrival scales with fixed-split scoring (GPU-free), not measurements."
    if note not in prov["notes"]:
        prov["notes"].append(note)
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"bundled {len(rows)} runs -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
