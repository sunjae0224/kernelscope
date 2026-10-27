import json

import pandas as pd
import pytest

from kernelscope.serve.scenarios import load_scenario
from scripts.gsm8k_accuracy import (build_requests, extract_prediction, gold_answer, main, normalize_number, paired_counts, same_number,
                                    score_run, truncate_generation)


def test_gold_answer_reads_the_number_after_the_marker():
    assert gold_answer("She makes 9 * 2 = $<<9*2=18>>18.\n#### 18") == 18.0
    assert gold_answer("blah\n#### 1,234") == 1234.0
    assert gold_answer("x\n#### -5") == -5.0
    with pytest.raises(ValueError):
        gold_answer("no marker 4")


def test_numbers_normalize_commas_and_trailing_dots():
    assert normalize_number("1,234.50") == 1234.5 and normalize_number("7.") == 7.0
    assert same_number(18.0, 18) and not same_number(None, 18) and not same_number(17.9, 18)


def test_generation_is_cut_at_the_first_stop_marker():
    text = "so 3 + 4 = 7\n#### 7<|im_end|>more 99"
    assert truncate_generation(text) == "so 3 + 4 = 7\n#### 7"
    assert truncate_generation("a 5\nQuestion: next 99") == "a 5"
    assert truncate_generation("plain 5") == "plain 5"


def test_prediction_is_last_number_and_marker_number():
    assert extract_prediction(" first 12 then 3 + 4 = 7\n#### 7\nQuestion: and 40") == (7.0, 7.0)
    assert extract_prediction(" the total is $1,250 dollars") == (1250.0, None)
    assert extract_prediction(" no digits here") == (None, None)
    assert extract_prediction("reasoning 5\n#### 18 apples and 3 more") == (3.0, 18.0)       # the two criteria can disagree


class _Tok:
    """decode maps token ids through a table; stands in for the tokenizers library object."""
    def __init__(self, table):
        self.table = table

    def decode(self, ids, skip_special_tokens=True):
        assert skip_special_tokens is False
        return "".join(self.table[i] for i in ids)


def test_score_run_scores_only_gold_rids_and_orders_by_position():
    tokens = pd.DataFrame({"rid": [0, 1, 1, 1, 2], "position": [0, 2, 0, 1, 0], "token": [9, 3, 1, 2, 1]})
    table = {1: "answer ", 2: "is 18", 3: "\n#### 18", 9: "filler 5"}
    res = score_run(tokens, _Tok(table), {1: 18.0, 2: 7.0})
    assert set(res) == {1, 2} and res[1][0] and res[1][1] and not res[2][0]


def test_paired_counts():
    ref = {1: (True, True, 5.0, 5.0), 2: (False, False, 3.0, None), 3: (True, True, 8.0, 8.0)}
    other = {1: (False, False, 4.0, None), 2: (True, True, 7.0, 7.0), 3: (True, True, 8.0, 8.0)}
    assert paired_counts(ref, other) == {"questions": 3, "right_to_wrong": 1, "wrong_to_right": 1, "both_right": 1, "same_prediction": 1}


def test_build_makes_a_loadable_grouped_scenario(tmp_path):
    parquet = tmp_path / "t.parquet"
    pd.DataFrame({"question": [f"What is {i} + 1?" for i in range(5)], "answer": [f"{i}+1\n#### {i + 1}" for i in range(5)]}).to_parquet(parquet)
    scenario, gold = tmp_path / "s.yaml", tmp_path / "g.json"
    main(["build", "--parquet", str(parquet), "--out-scenario", str(scenario), "--out-gold", str(gold), "--questions", "5", "--group", "2",
          "--long-len", "1024", "--spacing", "50", "--max-new", "8"])
    requests, dataset = load_scenario(scenario)
    assert len(requests) == 8                                        # 3 groups: 3 long requests + 5 questions
    assert [r.arrival_step for r in requests] == [0, 0, 0, 50, 50, 50, 100, 100]
    assert requests[0].prompt_recipe == "corpus_repeat" and requests[1].prompt_text.endswith("Answer:")
    g = json.loads(gold.read_text())["gold_by_rid"]
    assert g == {"1": 1.0, "2": 2.0, "4": 3.0, "5": 4.0, "7": 5.0}   # long requests (rids 0, 3, 6) carry no gold answer
    assert dataset["scenario_family"] == "gsm8k_ragged32k"
