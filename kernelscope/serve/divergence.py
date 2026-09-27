"""Where two greedy token histories part ways, and how close the reference logits were to a tie.

``compare_results`` counts differing tokens; that count mixes one flipped token with the whole tail
that a flip drags along, since every later step conditions on it. ``divergence_events`` reports the
first differing position per request and whether the tail cascaded. ``tie_class`` expresses a top-1
margin in bfloat16 spacing at the top logit, the resolution at which two policies' rounding differs.
"""
import math

import pandas as pd

COLUMNS = ["rid", "first_position", "tokens", "after_first", "differing_after_first", "cascade"]


def _decode_tokens(frame):
    f = frame[frame.phase == "decode"] if "phase" in frame else frame
    return f.set_index(["rid", "position"]).sort_index()[["token"]]


def divergence_events(reference, candidate) -> pd.DataFrame:
    """One row per request whose decode tokens differ, with the first differing position.

    ``cascade`` is true when every token from the first difference onward differs (the usual case
    for greedy decoding); a false value means the candidate flipped a token and then re-joined the
    reference history.
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
        rows.append({"rid": int(rid), "first_position": int(d.index[first][1]), "tokens": int(len(flags)),
                     "after_first": int(len(tail)), "differing_after_first": int(tail.sum()),
                     "cascade": bool(tail.all())})
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
