from pathlib import Path

import pytest

from kernelscope.serve.scenarios import Request, load, load_scenario, prompt_ids, resolve_requests


def test_scenarios_expand_arrivals_and_seeded_prompts():
    root = Path(__file__).resolve().parents[1]
    requests = load(root / "scenarios/arrivals.yaml")
    assert len(requests) == 33
    assert requests[-1].arrival_step == 112
    a, b = Request(0, 100, 4), Request(1, 100, 4)
    assert prompt_ids(a, 128) == prompt_ids(a, 128)
    assert prompt_ids(a, 128) != prompt_ids(b, 128)
    assert prompt_ids(a, 128, 1) != prompt_ids(a, 128, 2)
    assert min(prompt_ids(a, 128)) >= 0 and max(prompt_ids(a, 128)) < 128


@pytest.mark.parametrize("body", ["requests: []", "requests: [{count: 0, prompt_len: 1, max_new_tokens: 1}]",
                                    "requests: [{prompt_len: -1, max_new_tokens: 1}]",
                                    "requests: [{prompt_len: 1, max_new_tokens: 0}]",
                                    "requests: [{prompt_len: 1, max_new_tokens: 1, arrival_step: -1}]",
                                    "requests: [{prompt_len: 1, max_new_tokens: 1, typo: 1}]"])
def test_invalid_scenarios_fail_before_running(tmp_path, body):
    path = tmp_path / "trace.yaml"
    path.write_text(body)
    with pytest.raises(ValueError):
        load(path)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        from types import SimpleNamespace
        assert add_special_tokens is False
        return SimpleNamespace(ids=[ord(character) for character in text])


def test_corpus_recipe_exact_length_suffix_seed_and_hash_identity():
    req = Request(0, 100, 4, prompt_recipe="corpus_repeat", corpus=("First observation.", "Second observation.", "Third observation."),
                  prompt_suffix="\nQuestion: why?\nAnswer:")
    tokenizer = CharacterTokenizer()
    first, first_meta = resolve_requests([req], 256, tokenizer, seed=0, dataset={"id": "local-v1"})
    again, again_meta = resolve_requests([req], 256, tokenizer, seed=0, dataset={"id": "local-v1"})
    variants = [resolve_requests([req], 256, tokenizer, seed=i)[0][0].token_ids for i in range(6)]
    assert len(set(variants)) > 1
    assert first[0].prompt_len == len(first[0].token_ids) == 100
    assert first[0].token_ids[-len(req.prompt_suffix):] == tuple(map(ord, req.prompt_suffix))
    assert first[0].token_ids == again[0].token_ids
    assert first_meta["resolved_prompts_sha256"] == again_meta["resolved_prompts_sha256"]
    assert first_meta["prompt_kind"] == "constructed_natural_text"
    assert first_meta["tokenization_setup_us"] >= 0


def test_raw_text_and_explicit_tokens_resolve_without_changing_content():
    requests = [Request(0, None, 3, prompt_text="Answer: blue", expected_answer="blue"),
                Request(1, None, 3, token_ids=(1, 2, 3))]
    resolved, meta = resolve_requests(requests, 256, CharacterTokenizer())
    assert resolved[0].token_ids == tuple(map(ord, "Answer: blue"))
    assert resolved[0].prompt_len == 12 and resolved[0].expected_answer == "blue"
    assert resolved[1].token_ids == (1, 2, 3)
    assert meta["prompts"][0]["expected_answer"] == "blue"
    with pytest.raises(ValueError, match="tokenizer"):
        resolve_requests(requests, 256)
    with pytest.raises(ValueError, match="not prompt_len"):
        resolve_requests([Request(0, 2, 2, prompt_text="longer")], 256, CharacterTokenizer())


def test_synthetic_resolution_preserves_existing_seeded_tokens():
    request = Request(7, 50, 2)
    expected = prompt_ids(request, 128, seed=3)
    resolved, meta = resolve_requests([request], 128, seed=3)
    assert list(resolved[0].token_ids) == expected
    assert meta["prompt_kind"] == "seeded_synthetic_token_ids"


def test_heldout_shapes_and_qa_annotations_are_explicit():
    root = Path(__file__).resolve().parents[1]
    for family in ("uniform", "ragged", "arrivals"):
        requests, dataset = load_scenario(root / f"scenarios/heldout_text_{family}.yaml")
        resolved, meta = resolve_requests(requests, 256, CharacterTokenizer(), dataset=dataset)
        assert len(resolved) == 28 and all(r.max_new_tokens == 32 for r in resolved)
        assert meta["evaluation_split"] == "heldout_text" and meta["scenario_family"] == family
        assert max(r.prompt_len for r in resolved) == (384 if family == "uniform" else 12288)
    qa, _ = load_scenario(root / "scenarios/heldout_text_qa.yaml")
    assert len(qa) == 8 and all(r.expected_answer is not None for r in qa)


def test_invalid_tokens_or_oversized_suffix_fail_before_execution():
    with pytest.raises(ValueError, match="only one"):
        Request(0, 1, 1, token_ids=(1,), prompt_text="x")
    with pytest.raises(ValueError, match="vocabulary"):
        resolve_requests([Request(0, 1, 1, token_ids=(999,))], 16)
    with pytest.raises(ValueError, match="leave room"):
        resolve_requests([Request(0, 4, 1, prompt_recipe="corpus_repeat", corpus=("body",), prompt_suffix="long suffix")],
                         256, CharacterTokenizer())
