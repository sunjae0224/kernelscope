"""Where two greedy token histories part ways, and how close the reference logits were to a tie.

``compare_results`` counts differing tokens; that count mixes one flipped token with the whole tail
that a flip drags along, since every later step conditions on it. ``divergence_events`` reports the
first differing position per request and whether the tail cascaded. ``tie_class`` expresses a top-1
margin in bfloat16 spacing at the top logit, the resolution at which two policies' rounding differs.
``classify_events`` joins the two with a teacher-forced diagnostic, and ``campaign_events`` does that for
every run of a recorded campaign folder. Nothing here needs torch.
"""
import math
from pathlib import Path

import pandas as pd

COLUMNS = ["rid", "first_position", "tokens", "after_first", "differing_after_first", "cascade",
           "reference_token", "candidate_token"]


def _decode_tokens(frame):
    f = frame[frame.phase == "decode"] if "phase" in frame else frame
    return f.set_index(["rid", "position"]).sort_index()[["token"]]


def _token(value):
    return None if value is None or pd.isna(value) else int(value)


def divergence_events(reference, candidate) -> pd.DataFrame:
    """One row per request whose decode tokens differ, with the first differing position.

    ``cascade`` is true when every token from the first difference onward differs (the usual case
    for greedy decoding); a false value means the candidate flipped a token and then re-joined the
    reference history. ``reference_token`` / ``candidate_token`` are the two tokens at that position
    (None when one history has no token there).
    """
    joined = _decode_tokens(reference).join(_decode_tokens(candidate), how="outer", lsuffix="_ref", rsuffix="_cand")
    differs = (joined.token_ref != joined.token_cand) | joined.token_ref.isna() | joined.token_cand.isna()
    rows = []
    for rid, d in differs.groupby(level="rid", sort=True):
        d = d.sort_index()
        flags = d.to_numpy()
        if not flags.any():
            continue
        first = int(flags.argmax())
        tail = flags[first:]
        tokens = joined.loc[d.index[first]]
        rows.append({"rid": int(rid), "first_position": int(d.index[first][1]), "tokens": int(len(flags)),
                     "after_first": int(len(tail)), "differing_after_first": int(tail.sum()),
                     "cascade": bool(tail.all()), "reference_token": _token(tokens.token_ref),
                     "candidate_token": _token(tokens.token_cand)})
    return pd.DataFrame(rows, columns=COLUMNS)


def bf16_ulp(x: float) -> float:
    """Spacing between adjacent bfloat16 values at magnitude |x| (7 stored significand bits)."""
    x = abs(float(x))
    if x == 0.0 or not math.isfinite(x):
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(x)) - 7)


def tie_class(margin: float, top_logit: float) -> str:
    """'tie_1ulp' / 'tie_2ulp' when the top-1 margin is within one or two bf16 spacings, else 'clear'."""
    ulp = bf16_ulp(top_logit)
    if margin <= ulp * (1 + 1e-9):
        return "tie_1ulp"
    if margin <= 2 * ulp * (1 + 1e-9):
        return "tie_2ulp"
    return "clear"


TEACHER_COLUMNS = ["rid", "position", "argmax_equal", "reference_top1", "candidate_top1", "reference_top1_margin",
                   "reference_gap_to_candidate_token", "reference_top1_logit", "max_abs_logit_diff"]
CLASSIFIED_COLUMNS = COLUMNS + ["teacher_reproduced", "reference_top1_logit", "reference_margin", "margin_ulps",
                                "max_abs_logit_diff", "tie_class", "note"]


def classify_events(events, teacher=None) -> pd.DataFrame:
    """Each divergence event with the teacher-forced logit row at its first differing position.

    ``teacher`` is one policy's rows from ``scripts/check_policy_numerics.py``. ``teacher_reproduced``
    says whether the teacher-forced candidate produced the token free generation produced there. When
    it did, ``reference_margin`` is the reference's logit gap between its own token and that token;
    otherwise it is the reference's runner-up margin, a lower bound, and the event is flagged in
    ``note``. ``tie_class`` is that margin in bf16 spacings of the reference top logit. An event is
    'unclassified', never assumed to be a tie, when no row is recorded at its position or the row's
    reference token is not this run's (a diagnostic of other prompts or another seed).
    """
    rows_at = {}
    if teacher is not None:
        missing = [c for c in TEACHER_COLUMNS if c not in teacher]
        if missing:
            raise ValueError(f"teacher-forced rows lack {missing}; record them with scripts/check_policy_numerics.py")
        if teacher.duplicated(["rid", "position"]).any():
            raise ValueError("teacher-forced rows must be one policy's rows (duplicate rid/position)")
        rows_at = {(int(r.rid), int(r.position)): r for r in teacher.itertuples()}
    out = []
    for event in events.to_dict("records"):
        row = rows_at.get((int(event["rid"]), int(event["first_position"])))
        extra = {"teacher_reproduced": None, "reference_top1_logit": math.nan, "reference_margin": math.nan,
                 "margin_ulps": math.nan, "max_abs_logit_diff": math.nan, "tie_class": "unclassified",
                 "note": "no teacher-forced row at this position"}
        if row is not None and not (math.isfinite(row.reference_top1_margin) and math.isfinite(row.reference_top1_logit)):
            extra["note"] = "teacher-forced logits were not finite"
        elif row is not None and _token(event.get("reference_token")) != int(row.reference_top1):
            extra["note"] = "the diagnostic's reference token differs from this run's"
        elif row is not None:
            reproduced = not bool(row.argmax_equal) and _token(event.get("candidate_token")) == int(row.candidate_top1)
            margin = float(row.reference_gap_to_candidate_token if reproduced else row.reference_top1_margin)
            top = float(row.reference_top1_logit)
            extra = {"teacher_reproduced": reproduced, "reference_top1_logit": top, "reference_margin": margin,
                     "margin_ulps": margin / bf16_ulp(top), "max_abs_logit_diff": float(row.max_abs_logit_diff),
                     "tie_class": tie_class(margin, top),
                     "note": "" if reproduced else "teacher-forced run did not produce the same token"}
        out.append({**event, **extra})
    return pd.DataFrame(out, columns=CLASSIFIED_COLUMNS)


EVENT_KEYS = ["scenario", "policy", "reference", "repeat", "tokens_compared", "token_mismatches", "missing_tokens",
              "mismatches_outside_events", "events"]
INTEGER_COLUMNS = ["rid", "first_position", "tokens", "after_first", "differing_after_first", "reference_token",
                   "candidate_token"]
TIE_CLASSES = ["tie_1ulp", "tie_2ulp", "clear", "unclassified"]
SUMMARY_COLUMNS = ["scenario", "policy", "repeats", "token_mismatches_max", "events_repeat_max", "distinct_positions",
                   *TIE_CLASSES, "not_reproduced", "missing_tokens", "mismatches_outside_events",
                   "same_positions_every_repeat"]


def _recorded_repeats(policy_dir) -> set:
    return {p.parent.name for p in policy_dir.glob("repeat_*/tokens.parquet")}


def campaign_events(campaign, reference="heuristic") -> pd.DataFrame:
    """Classified divergence events of every non-reference policy run in a campaign folder.

    Layout: ``<campaign>/<scenario>/<policy>/repeat_NNN/tokens.parquet`` written by ``serve run`` and
    ``<campaign>/numerics/<scenario>/teacher_forced_logits.csv`` written by the teacher-forced
    diagnostic; a directory without the reference policy is not a scenario. One row per event; a run
    without events keeps a single row with ``events == 0``. Replacing the token count by events must
    not hide a difference, so two more counts are kept per run: ``missing_tokens`` (positions only
    one history has) and ``mismatches_outside_events`` (differing tokens no decode event covers, i.e.
    a differing prefill token). A policy whose recorded repeats differ from the reference's is an error.
    """
    campaign = Path(campaign)
    rows = []
    for scenario in sorted(p for p in campaign.iterdir() if (p / reference).is_dir()):
        teacher_path = campaign / "numerics" / scenario.name / "teacher_forced_logits.csv"
        teacher = pd.read_csv(teacher_path) if teacher_path.exists() else None
        reference_repeats = _recorded_repeats(scenario / reference)
        for policy_dir in sorted(p for p in scenario.iterdir() if p.is_dir() and p.name != reference):
            repeats = _recorded_repeats(policy_dir)
            if repeats != reference_repeats:
                raise ValueError(f"{scenario.name}/{policy_dir.name}: recorded repeats {sorted(repeats)} cannot be paired "
                                 f"with the reference's {sorted(reference_repeats)}")
            policy_rows = None
            if teacher is not None and (teacher.policy == policy_dir.name).any():
                policy_rows = teacher[teacher.policy == policy_dir.name]
            for repeat in sorted(repeats):
                ref = pd.read_parquet(scenario / reference / repeat / "tokens.parquet")
                cand = pd.read_parquet(policy_dir / repeat / "tokens.parquet")
                both = ref.merge(cand, on=["rid", "position"], how="outer", suffixes=("_ref", "_cand"))
                events = classify_events(divergence_events(ref, cand), policy_rows)
                mismatches = int((both.token_ref != both.token_cand).sum())
                base = {"scenario": scenario.name, "policy": policy_dir.name, "reference": reference,
                        "repeat": int(repeat.split("_")[1]), "tokens_compared": len(both),
                        "token_mismatches": mismatches,
                        "missing_tokens": int(both.token_ref.isna().sum() + both.token_cand.isna().sum()),
                        "mismatches_outside_events": mismatches - int(events.differing_after_first.sum()),
                        "events": len(events)}
                if events.empty:
                    rows.append(base)
                else:
                    rows.extend({**base, **event} for event in events.to_dict("records"))
    frame = pd.DataFrame(rows, columns=EVENT_KEYS + CLASSIFIED_COLUMNS)
    return frame.astype({column: "Int64" for column in INTEGER_COLUMNS})     # clean runs leave these empty


def summarize_campaign_events(events) -> pd.DataFrame:
    """Per scenario and policy: repeats, events per repeat and the distinct event positions by tie class.

    ``not_reproduced`` counts the positions where the teacher-forced run did not produce the token
    free generation produced, so their class rests on the runner-up margin alone.
    """
    rows = []
    for (scenario, policy), group in events.groupby(["scenario", "policy"], sort=True):
        real = group[group.events > 0]
        positions = real.drop_duplicates(["rid", "first_position"])
        per_repeat = {repeat: frozenset() for repeat in group.repeat.unique()}
        per_repeat.update({repeat: frozenset(zip(g.rid, g.first_position)) for repeat, g in real.groupby("repeat")})
        rows.append({"scenario": scenario, "policy": policy, "repeats": int(group.repeat.nunique()),
                     "token_mismatches_max": int(group.token_mismatches.max()),
                     "events_repeat_max": int(group.events.max()), "distinct_positions": len(positions),
                     **{name: int((positions.tie_class == name).sum()) for name in TIE_CLASSES},
                     "not_reproduced": int(positions.teacher_reproduced.eq(False).sum()),
                     "missing_tokens": int(group.missing_tokens.max()),
                     "mismatches_outside_events": int(group.mismatches_outside_events.max()),
                     "same_positions_every_repeat": len(set(per_repeat.values())) <= 1})
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
