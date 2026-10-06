"""Write race.json next to a serving recording so the Demo page replays it without the tokenizer.

    python scripts/export_race.py --run ../kernelscope/results/serve_4090/demo_text_20261006/ragged
    python scripts/export_race.py --run <dir> --policies heuristic,table --no-text

Policies default to every policy folder with recorded steps (the reference `heuristic` first).
Text decoding needs the local model snapshot (HF_HUB_OFFLINE) and torch; `--no-text` skips it."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernelscope.dashboard import data, demo


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="scenario directory holding <policy>/repeat_*/tokens.parquet")
    parser.add_argument("--policies", help="comma list; default: every recorded policy")
    parser.add_argument("--reference", default="heuristic")
    parser.add_argument("--no-text", action="store_true", help="skip token decoding (no torch / model needed)")
    args = parser.parse_args()
    run = Path(args.run)
    policies = args.policies.split(",") if args.policies else demo.policy_dirs(run)
    if args.reference not in policies:
        raise SystemExit(f"{run} has no {args.reference!r} recording; policies: {policies}")
    decode = None
    if not args.no_text:
        from kernelscope.serve.hf import load_tokenizer, snapshot_dir
        tokenizer = load_tokenizer(snapshot_dir(data.load_manifest(run).get("model")))
        decode = lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True)
    payload = demo.race_payload(run, policies, reference=args.reference, decode=decode)
    path = demo.save_race(run, payload)
    print(f"{path} · policies {[p['name'] for p in payload['policies']]} · natural_text {payload['scenario']['natural_text']}")


if __name__ == "__main__":
    main()
