"""Accuracy and paired-flip statistics over the per-question table of ``scripts/analyze_gsm8k_policies.py`` (torch-free)."""
import math

import pandas as pd

PERQUESTION = "gsm8k_ragged32k_n200_perquestion.csv"


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    """95% Wilson score interval of a proportion."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (center - half, center + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p-value of the discordant pairs b (right->wrong) and c (wrong->right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def policy_stats(table: pd.DataFrame, policy: str, metric: str = "marker") -> dict:
    """Accuracy and its interval for one policy (``correct_<metric>_<policy>`` columns)."""
    col = table[f"correct_{metric}_{policy}"].astype(bool)
    k, n = int(col.sum()), len(col)
    low, high = wilson(k, n)
    return {"correct": k, "questions": n, "accuracy": k / n, "ci_low": low, "ci_high": high}


def paired(table: pd.DataFrame, policy: str, reference: str = "heuristic", metric: str = "marker") -> dict:
    ref = table[f"correct_{metric}_{reference}"].astype(bool)
    cand = table[f"correct_{metric}_{policy}"].astype(bool)
    b, c = int((ref & ~cand).sum()), int((~ref & cand).sum())
    differs = ~table[f"tokens_identical_{policy}"].astype(bool)
    same_pred = (table[f"pred_{metric}_{policy}"].fillna(-1e18) == table[f"pred_{metric}_{reference}"].fillna(-1e18))
    return {"right_to_wrong": b, "wrong_to_right": c, "p_value": mcnemar_exact(b, c), "net": c - b,
            "text_differs": int(differs.sum()), "prediction_changed": int((~same_pred).sum()),
            "text_differs_answer_same": int((differs & same_pred).sum())}
