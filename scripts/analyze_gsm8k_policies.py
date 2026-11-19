"""Per-question GSM8K table of a ``serve run`` campaign (policies x repeats), written for the portable bundle.

    python -m scripts.analyze_gsm8k_policies --campaign demo_data/serve_4090/gsm8k_20261024 --scenario gsm8k_ragged32k_n200 \
        --gold demo_data/serve_4090/gsm8k_20261024/gsm8k_ragged32k_n200_gold.json

For every question: the gold number, per policy the number extracted after the first ``####`` marker (``marker``: what the prompt asks
for) and the last number before the first stop marker (``last``: the metric ``scripts/gsm8k_accuracy.py score`` reports; it picks up
numbers from text the engine keeps generating after the answer, because it does not stop at end of sequence), whether each is correct,
whether the generated token sequence equals the reference policy's, and whether the three repeats of the policy generated the same tokens.
Statistics are computed from this table by ``kernelscope.analysis.accuracy`` (no tokenizer needed).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from kernelscope.analysis.accuracy import PERQUESTION
from scripts.gsm8k_accuracy import extract_prediction, same_number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--reference", default="heuristic")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    from kernelscope.serve.hf import load_tokenizer, snapshot_dir
    tokenizer = load_tokenizer(snapshot_dir(args.model))
    gold = {int(k): v for k, v in json.loads(args.gold.read_text())["gold_by_rid"].items()}
    root = args.campaign / args.scenario
    runs = {}
    for policy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        repeats = []
        for repeat in sorted(policy_dir.glob("repeat_*")):
            tokens = pd.read_parquet(repeat / "tokens.parquet")
            repeats.append({int(rid): tuple(group.sort_values("position").token.astype(int)) for rid, group in tokens.groupby("rid")
                            if int(rid) in gold})
        runs[policy_dir.name] = repeats
    reference = runs[args.reference][0]
    rows = []
    for rid in sorted(gold):
        row = {"rid": rid, "gold": gold[rid]}
        for policy, repeats in runs.items():
            ids = repeats[0][rid]
            last, marked = extract_prediction(tokenizer.decode(list(ids), skip_special_tokens=False))
            row.update({f"pred_last_{policy}": last, f"pred_marker_{policy}": marked,
                        f"correct_last_{policy}": same_number(last, gold[rid]), f"correct_marker_{policy}": same_number(marked, gold[rid]),
                        f"tokens_identical_{policy}": ids == reference[rid],
                        f"repeats_identical_{policy}": all(r[rid] == ids for r in repeats)})
            if ids != reference[rid]:
                row[f"first_diff_position_{policy}"] = next((i for i, (a, b) in enumerate(zip(ids, reference[rid])) if a != b), min(len(ids), len(reference[rid])))
        rows.append(row)
    out = args.out or args.campaign / PERQUESTION
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} questions, policies {sorted(runs)} -> {out}")


if __name__ == "__main__":
    main()
