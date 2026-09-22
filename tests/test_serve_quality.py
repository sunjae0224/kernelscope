from types import SimpleNamespace

import pandas as pd
import pytest

from kernelscope.serve.quality import normalize_answer, score_answer, score_tokens


def test_normalization_is_explicit_and_does_not_accept_substrings():
    row = score_answer("  BLUE\n", "blue")
    assert not row["exact_match"] and row["normalized_exact_match"]
    assert row["generated_text"] == "  BLUE\n" and row["expected_answer"] == "blue"
    assert normalize_answer("５") == "5"
    assert not score_answer("15", "5")["normalized_exact_match"]
    assert not score_answer("blue bed", "blue")["normalized_exact_match"]
    assert not score_answer("blue.", "blue")["normalized_exact_match"]
    with pytest.raises(ValueError):
        score_answer("anything", " ")


def test_scoring_retains_answers_and_order_without_claiming_token_equivalence():
    class Tokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(token) for token in ids)
    requests = [SimpleNamespace(rid=0, expected_answer="blue", prompt_sha256="test"),
                SimpleNamespace(rid=1, expected_answer=None)]
    tokens = pd.DataFrame([dict(rid=0, position=i, token=ord(character)) for i, character in enumerate("Blue")][::-1])
    row = score_tokens(requests, tokens, Tokenizer()).iloc[0]
    assert row.generated_text == "Blue" and row.normalized_exact_match
    assert not row.exact_match and row.generated_tokens == 4
