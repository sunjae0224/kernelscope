"""Classify a serving campaign's divergence events with its teacher-forced diagnostics (no GPU needed).

    python -m scripts.classify_divergence --campaign ../kernelscope/results/serve_4090/<campaign>

Reads ``<campaign>/<scenario>/<policy>/repeat_*/tokens.parquet`` (``serve run``) and
``<campaign>/numerics/<scenario>/teacher_forced_logits.csv`` (``scripts/check_policy_numerics.py``),
writes ``<campaign>/divergence.csv`` and prints one line per scenario and policy. It exits non-zero,
after writing the table, when anything needs a look: a ``clear`` event (reference margin beyond two
bf16 spacings, treated as an implementation error), an event without a usable teacher-forced row, an
event the teacher-forced run did not reproduce, a differing token outside every decode event, or
token histories of different length. Runs that cannot be paired with the reference are refused.
"""
import argparse
from pathlib import Path

from kernelscope.serve.divergence import campaign_events, summarize_campaign_events


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--reference", default="heuristic")
    parser.add_argument("--out", type=Path, help="default: <campaign>/divergence.csv")
    args = parser.parse_args(argv)
    try:
        events = campaign_events(args.campaign, args.reference)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    if events.empty:
        raise SystemExit(f"no runs to compare under {args.campaign}: expected <scenario>/{args.reference}/repeat_*/tokens.parquet "
                         "next to at least one other policy")
    out = args.out or args.campaign / "divergence.csv"
    events.to_csv(out, index=False)
    summary = summarize_campaign_events(events)
    print(summary.to_string(index=False))
    print(f"Events: {out}")
    labels = {"clear": "clear", "unclassified": "unclassified", "not_reproduced": "not reproduced by the teacher-forced run",
              "mismatches_outside_events": "token mismatches outside decode events", "missing_tokens": "missing tokens"}
    problems = [f"{int(summary[name].sum())} {label}" for name, label in labels.items() if summary[name].sum()]
    if problems:
        raise SystemExit("divergence events need investigation: " + ", ".join(problems))

if __name__ == "__main__":
    main()
