"""Replay model / table / hybrid kernel selection on the recorded RTX 4090 cells (no GPU needed).

    python -m scripts.evaluate_hybrid --out docs/experiments/2026-09-26-hybrid-policy.md

For every validation cell the table is built from every *other* measured cell (leave-one-out), so the
comparison cannot flatter the table. Regret is the picked variant's measured time over the best measured
variant in the same cell, minus one.
"""
import argparse
from pathlib import Path

from kernelscope.model.fit import prepare_rows
from kernelscope.model.hybrid import evaluate, markdown
from kernelscope.model.machine import MachineSpec
from kernelscope.model.params import ModelParams
from kernelscope.results.store import load_dirs
from kernelscope.verify import P0_DIRS

REPO = Path(__file__).resolve().parents[1]
SETS = (("V1", "cold"), ("V1", "warm"), ("V5", "cold"), ("V6", "cold"))


def run(repo=REPO, deltas=(0.05, 0.10, 0.20, 0.50)):
    rows = prepare_rows(load_dirs([repo / "demo_data" / "hw_4090" / d for d in P0_DIRS]),
                        MachineSpec.from_json(repo / "machines" / "rtx4090.json"))
    full = ModelParams.from_json(repo / "models" / "rtx4090.json")
    uniform = ModelParams.from_json(repo / "models" / "rtx4090_uniform.json")
    return [evaluate(rows, uniform if s == "V6" else full, s, c, deltas) for s, c in SETS]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--deltas", default="0.05,0.10,0.20,0.50")
    args = ap.parse_args(argv)
    results = run(deltas=tuple(float(x) for x in args.deltas.split(",")))
    md = markdown(results)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)


if __name__ == "__main__":
    main()
