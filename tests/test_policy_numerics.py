import pytest

torch = pytest.importorskip("torch")

from scripts.check_policy_numerics import TeacherForcedModel, logit_statistics, prepare_requests
from kernelscope.serve.dispatch import FixedPolicy
from kernelscope.serve.engine import Engine
from kernelscope.serve.scenarios import Request
from tests.test_serve_engine import ArithmeticModel, pool


def test_logit_stats_expose_small_margin_and_argmax_ties():
    rows = logit_statistics(torch.tensor([[1., 1., 0.], [3., 1., 0.]]),
                            torch.tensor([[.999, 1., 0.], [3.01, 1., 0.]]))
    assert rows[0]["reference_tied_top1"] and not rows[0]["argmax_equal"]
    assert rows[0]["reference_top1"] == 0 and rows[0]["candidate_top1"] == 1
    assert rows[1]["argmax_equal"] and rows[1]["guaranteed_stable_by_linf_bound"]
    assert rows[0]["cosine_similarity"] > .999


def test_logit_stats_do_not_hide_nonfinite_or_bad_shapes():
    assert not logit_statistics(torch.zeros(1, 3), torch.full((1, 3), float("nan")))[0]["finite"]
    with pytest.raises(ValueError):
        logit_statistics(torch.zeros(2, 3), torch.zeros(1, 3))


def test_teacher_forcing_prevents_candidate_tokens_from_becoming_inputs():
    class PerturbedModel(ArithmeticModel):
        seen = []
        def decode(self, seq_ids, tokens, pool, **kwargs):
            self.seen.append(list(tokens))
            super().decode(seq_ids, tokens, pool, **kwargs)
            return torch.stack([self._logits(t + 2) for t in tokens])
    requests = [Request(0, 3, 4)]
    cache = pool()
    reference = Engine(ArithmeticModel(), cache, FixedPolicy(0)).run(requests, 16, record_logits_steps=3)
    changed = PerturbedModel()
    wrapper = TeacherForcedModel(changed, reference, requests)
    candidate = Engine(wrapper, cache, FixedPolicy(8)).run(requests, 16)
    assert changed.seen == [[t] for t in reference.tokens.token.tolist()[:-1]]
    assert candidate.tokens.token.tolist() != reference.tokens.token.tolist()
    assert len(wrapper.rows) == 3 and not any(row["argmax_equal"] for row in wrapper.rows)


def test_natural_diagnostic_uses_shared_tokenization_and_dataset_identity(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import kernelscope.serve.hf as hf
    from tests.test_serve_scenarios import CharacterTokenizer
    monkeypatch.setattr(hf, "snapshot_dir", lambda name: tmp_path)
    monkeypatch.setattr(hf, "load_tokenizer", lambda path: CharacterTokenizer())
    scenario = tmp_path / "text.yaml"
    scenario.write_text('dataset: {id: diagnostic-text, evaluation_split: heldout_text}\n'
                        'requests: [{prompt_text: "Question: why? Answer:", max_new_tokens: 32, arrival_step: 4}]\n')
    args = SimpleNamespace(model="local/model", seed=2, steps=64, scenario=scenario)
    resolved, meta = prepare_requests(args, SimpleNamespace(cfg=SimpleNamespace(vocab=256)))
    assert resolved[0].token_ids == tuple(map(ord, "Question: why? Answer:"))
    assert resolved[0].max_new_tokens == 32 and resolved[0].arrival_step == 4
    assert meta["dataset_id"] == "diagnostic-text" and meta["prompt_kind"] == "constructed_natural_text"
    assert meta["resolved_prompts_sha256"] and meta["tokenization_setup_us"] >= 0
    args.steps = 2
    capped, _ = prepare_requests(args, SimpleNamespace(cfg=SimpleNamespace(vocab=256)))
    assert capped[0].max_new_tokens == 3 and capped[0].token_ids == resolved[0].token_ids
