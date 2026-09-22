"""Literal answer checks for the small original QA sanity set, not a general quality metric."""
import re
import unicodedata

import pandas as pd


def normalize_answer(text: str) -> str:
    """NFKC, case folding, and whitespace only; do not accept substring matches."""
    if not isinstance(text, str):
        raise ValueError("answers must be strings")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def score_answer(generated_text: str, expected_answer: str) -> dict:
    expected, generated = normalize_answer(expected_answer), normalize_answer(generated_text)
    if not expected:
        raise ValueError("expected answer must contain non-whitespace text")
    return {"expected_answer": expected_answer, "generated_text": generated_text,
            "normalized_expected_answer": expected, "normalized_generated_text": generated,
            "exact_match": generated_text == expected_answer, "normalized_exact_match": generated == expected}


def score_tokens(requests, tokens: pd.DataFrame, tokenizer) -> pd.DataFrame:
    if tokens.duplicated(["rid", "position"]).any():
        raise ValueError("duplicate token positions in QA result")
    rows = []
    for request in requests:
        if request.expected_answer is None:
            continue
        sequence = tokens[tokens.rid == request.rid].sort_values("position")
        output = tokenizer.decode(sequence.token.astype(int).tolist(), skip_special_tokens=True)
        rows.append({"rid": request.rid, "prompt_sha256": request.prompt_sha256,
                     "generated_tokens": len(sequence), **score_answer(output, request.expected_answer)})
    return pd.DataFrame(rows)
