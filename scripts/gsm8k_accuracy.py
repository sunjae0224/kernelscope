"""GSM8K accuracy under different attention split policies (mixed-length batches), paired by question.

    python -m scripts.gsm8k_accuracy build --parquet test-00000-of-00001.parquet --out-scenario scenarios/gsm8k_ragged32k_n200.yaml --out-gold GOLD.json
    python -m scripts.gsm8k_accuracy score --campaign DIR --scenario gsm8k_ragged32k_n200 --gold GOLD.json

``build`` writes a scenario in the engine's existing format (``prompt_text`` requests, no engine change): the first ``--questions``
test questions in groups of ``--group``; each group shares a decode batch with one 32K-token natural-text request (the Campus Garden
corpus of demo_text_ragged, repeated to length), groups start ``--spacing`` logical steps apart so they do not overlap. The engine
generates exactly ``max_new_tokens`` tokens (no end-of-sequence stop), so ``score`` cuts each generation at the first stop marker
before extracting the answer. The gold answers stay in a sidecar file, not in the scenario (the serve command's literal QA scoring
would otherwise run on them). ``score`` decodes ``tokens.parquet`` of every policy and repeat, extracts the prediction and compares
it with the number after ``####`` in the dataset answer. Pure CPU.
"""
import argparse
import json
import re
from pathlib import Path

NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "\nQuestion:", "\n\nProblem", "\nProblem:")
TEMPLATE = ("Solve the following math problem step by step. After the reasoning, write the final answer on its own line "
            "as '#### <number>'.\n\nQuestion: {question}\nAnswer:")
SUFFIX = "Read the notes above. Summarize one monitoring practice in a short sentence.\\nAnswer:"


def normalize_number(text):
    """float of a matched number ('1,234.50' -> 1234.5), or None."""
    try:
        return float(text.replace(",", "").rstrip("."))
    except ValueError:
        return None


def gold_answer(answer_field):
    """The number after '####' in a GSM8K answer."""
    if "####" not in answer_field:
        raise ValueError("GSM8K answer without '####'")
    found = NUMBER_RE.findall(answer_field.split("####", 1)[1])
    if not found:
        raise ValueError("no number after '####'")
    return normalize_number(found[0])


def truncate_generation(text):
    """Cut at the first stop marker (the engine does not stop at end-of-sequence)."""
    cut = len(text)
    for marker in STOP_MARKERS:
        i = text.find(marker)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def extract_prediction(text):
    """(last number in the truncated text, number after '####' in it); None where absent."""
    text = truncate_generation(text)
    last = NUMBER_RE.findall(text)
    marked = None
    if "####" in text:
        found = NUMBER_RE.findall(text.split("####", 1)[1])
        marked = normalize_number(found[0]) if found else None
    return (normalize_number(last[-1]) if last else None), marked


def same_number(a, b, tol=1e-6):
    return a is not None and b is not None and abs(a - b) <= tol * max(1.0, abs(b))


def build_requests(questions, golds, group, long_len, spacing, max_new):
    """[(arrival_step, 'text' question prompt)] per question plus one long request per group; returns (yaml lines, gold by rid)."""
    lines, gold_by_rid, rid = [], {}, 0
    for g in range(0, len(questions), group):
        start = (g // group) * spacing
        lines.append(f"  - {{count: 1, prompt_len: {long_len}, max_new_tokens: {max_new}, arrival_step: {start}, "
                     f"prompt_recipe: corpus_repeat, prompt_suffix: \"{SUFFIX}\"}}")
        rid += 1
        for q, a in zip(questions[g:g + group], golds[g:g + group]):
            lines.append(f"  - {{count: 1, max_new_tokens: {max_new}, arrival_step: {start}, prompt_text: {json.dumps(TEMPLATE.format(question=q))}}}")
            gold_by_rid[rid] = a
            rid += 1
    return lines, gold_by_rid


def build(args):
    import pandas as pd
    frame = pd.read_parquet(args.parquet).iloc[:args.questions]
    questions = frame.question.tolist()
    golds = [gold_answer(a) for a in frame.answer]
    corpus = re.search(r"corpus:\n(.*?)\nrequests:", Path(args.corpus_from).read_text(), re.S).group(1)
    lines, gold_by_rid = build_requests(questions, golds, args.group, args.long_len, args.spacing, args.max_new)
    header = (f"# GSM8K test, first {len(questions)} questions in groups of {args.group}, each group sharing one decode batch with a "
              f"{args.long_len}-token natural-text request (2026-10-24).\n"
              f"# Groups start {args.spacing} logical steps apart; {args.max_new} generated tokens per request, no end-of-sequence stop.\n"
              "# Gold answers are in the sidecar file written by scripts/gsm8k_accuracy.py build.\n"
              "dataset:\n  id: gsm8k-test-first200\n  evaluation_split: gsm8k_test\n  scenario_family: gsm8k_ragged32k\n"
              "  provenance: openai/gsm8k test split, revision in datasets/gsm8k/SOURCE.md\n"
              "  construction: question prompts as written plus corpus_repeat filler request per group\n")
    out = Path(args.out_scenario)
    if out.exists():
        raise SystemExit(f"scenario exists: {out}")
    out.write_text(header + "corpus:\n" + corpus + "\nrequests:\n" + "\n".join(lines) + "\n")
    gold_path = Path(args.out_gold)
    if gold_path.exists():
        raise SystemExit(f"gold file exists: {gold_path}")
    gold_path.write_text(json.dumps({"gold_by_rid": {str(k): v for k, v in gold_by_rid.items()}, "questions": len(questions),
                                     "group": args.group, "long_len": args.long_len, "spacing": args.spacing,
                                     "max_new_tokens": args.max_new, "parquet": str(args.parquet)}, indent=1))
    print(f"wrote {out} ({len(questions)} questions, {-(-len(questions) // args.group)} groups) and {gold_path}")


def score_run(tokens, tokenizer, gold_by_rid):
    """Per-question results of one run: {rid: (correct_last_number, correct_marker, prediction, marked)}."""
    out = {}
    for rid, group in tokens.groupby("rid"):
        rid = int(rid)
        if rid not in gold_by_rid:
            continue
        ids = group.sort_values("position").token.astype(int).tolist()
        last, marked = extract_prediction(tokenizer.decode(ids, skip_special_tokens=False))
        out[rid] = (same_number(last, gold_by_rid[rid]), same_number(marked, gold_by_rid[rid]), last, marked)
    return out


def paired_counts(reference, other):
    """Questions that flip between a reference run and another run on the last-number criterion."""
    both = sorted(set(reference) & set(other))
    return {"questions": len(both), "right_to_wrong": sum(reference[r][0] and not other[r][0] for r in both),
            "wrong_to_right": sum(not reference[r][0] and other[r][0] for r in both),
            "both_right": sum(reference[r][0] and other[r][0] for r in both),
            "same_prediction": sum(reference[r][2] == other[r][2] for r in both)}


def score(args):
    import pandas as pd
    gold = {int(k): v for k, v in json.loads(Path(args.gold).read_text())["gold_by_rid"].items()}
    from kernelscope.serve.hf import load_tokenizer, snapshot_dir
    tokenizer = load_tokenizer(snapshot_dir(args.model))
    root = Path(args.campaign) / args.scenario
    runs = {}
    for policy_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for repeat in sorted(policy_dir.glob("repeat_*")):
            path = repeat / "tokens.parquet"
            if path.exists():
                runs[(policy_dir.name, repeat.name)] = score_run(pd.read_parquet(path), tokenizer, gold)
    rows, flips = [], []
    for (policy, repeat), res in runs.items():
        n = len(res)
        rows.append({"policy": policy, "repeat": repeat, "questions": n,
                     "accuracy_last_number": sum(v[0] for v in res.values()) / n if n else None,
                     "accuracy_marker": sum(v[1] for v in res.values()) / n if n else None,
                     "marker_present": sum(v[3] is not None for v in res.values()) / n if n else None,
                     "no_number": sum(v[2] is None for v in res.values())})
        ref = runs.get((args.reference, repeat))
        if ref is not None and policy != args.reference:
            flips.append({"policy": policy, "repeat": repeat, **paired_counts(ref, res)})
    out = Path(args.out or Path(args.campaign) / f"{args.scenario}_accuracy.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(flips).to_csv(out.with_name(out.stem + "_paired.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(flips).to_string(index=False) if flips else "no paired runs")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--parquet", required=True)
    b.add_argument("--out-scenario", required=True)
    b.add_argument("--out-gold", required=True)
    b.add_argument("--questions", type=int, default=200)
    b.add_argument("--group", type=int, default=31)
    b.add_argument("--long-len", type=int, default=32768)
    b.add_argument("--spacing", type=int, default=270, help="logical steps between groups (> max_new_tokens so groups do not overlap)")
    b.add_argument("--max-new", type=int, default=256)
    b.add_argument("--corpus-from", default="scenarios/demo_text_ragged.yaml")
    s = sub.add_parser("score")
    s.add_argument("--campaign", required=True)
    s.add_argument("--scenario", required=True)
    s.add_argument("--gold", required=True)
    s.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    s.add_argument("--reference", default="heuristic")
    s.add_argument("--out")
    args = parser.parse_args(argv)
    build(args) if args.cmd == "build" else score(args)


if __name__ == "__main__":
    main()
