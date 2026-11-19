import contextlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernelscope.serve.dispatch import FixedPolicy
from kernelscope.serve.engine import Engine
from kernelscope.serve.kvcache import PagePool
from kernelscope.serve.scenarios import Request


class ArithmeticModel:
    """Scheduler-only test double; these timings are never performance evidence."""
    cfg = SimpleNamespace(vocab=16, n_heads=2, n_kv_heads=1, head_dim=8)
    device, dtype = torch.device("cpu"), torch.float32
    backend = "scheduler_test_double"

    def _logits(self, token):
        output = torch.zeros(self.cfg.vocab)
        output[token % self.cfg.vocab] = 1
        return output

    def prefill(self, rid, tokens, pool):
        pool.reserve(rid, len(tokens))
        pool.set_length(rid, len(tokens))
        return self._logits(sum(tokens))

    def decode(self, seq_ids, tokens, pool, num_splits=0, timer=None):
        timer.start(0)
        for rid in seq_ids:
            length = pool.length(rid) + 1
            pool.reserve(rid, length)
            pool.set_length(rid, length)
        output = torch.stack([self._logits(token + 1) for token in tokens])
        timer.stop(0)
        return output


def pool(pages=12):
    return PagePool(1, pages, 1, 8, dtype=torch.float32, device="cpu")


def test_continuous_batching_arrivals_retires_and_token_counts():
    cache = pool()
    requests = [Request(0, 600, 8), Request(1, 40, 8), Request(2, 40, 6), Request(3, 100, 6, 3)]
    result = Engine(ArithmeticModel(), cache, FixedPolicy(0)).run(requests, 16)
    assert result.steps.B[:4].tolist() == [3, 3, 3, 4]
    assert result.tokens.groupby("rid").size().to_dict() == {0: 8, 1: 8, 2: 6, 3: 6}
    assert cache.free_pages == 12
    assert (result.steps.step_us >= result.steps.attn_us).all()
    assert (result.steps.decode_wall_us >= result.steps.policy_us).all()
    assert result.metadata["arrival_mode"] == "logical_decode_step"


def test_page_budget_backpressure_and_batch_limit_are_deterministic():
    requests = [Request(0, 300, 3), Request(1, 300, 3), Request(2, 1, 1, 8)]
    a = Engine(ArithmeticModel(), pool(2), FixedPolicy(0), 8).run(requests, 16)
    b = Engine(ArithmeticModel(), pool(2), FixedPolicy(8), 1).run(requests, 16)
    assert a.steps.B.tolist() == [1, 1, 1, 1]
    assert a.tokens[["rid", "step", "position", "token"]].equals(b.tokens[["rid", "step", "position", "token"]])
    assert a.tokens[a.tokens.rid == 1].step.min() == 2
    assert a.tokens[a.tokens.rid == 2].step.tolist() == [8]


def test_single_token_requests_release_without_decode():
    cache = pool(1)
    result = Engine(ArithmeticModel(), cache, FixedPolicy(0)).run([Request(0, 5, 1), Request(1, 5, 1)], 16)
    assert result.steps.empty and len(result.tokens) == 2 and cache.free_pages == 1


def test_impossible_request_and_failure_release_pages():
    cache = pool(1)
    with pytest.raises(MemoryError, match="increase"):
        Engine(ArithmeticModel(), cache, FixedPolicy(0)).run([Request(0, 300, 2)], 16)
    class FailingModel(ArithmeticModel):
        def decode(self, *args, **kwargs):
            raise RuntimeError("simulated failure")
    with pytest.raises(RuntimeError, match="simulated"):
        Engine(FailingModel(), cache, FixedPolicy(0)).run([Request(0, 2, 2)], 16)
    assert cache.free_pages == 1


def test_resolved_explicit_prompts_and_unresolved_text_validation():
    from kernelscope.serve.scenarios import resolve_requests
    cache = pool()
    with pytest.raises(ValueError, match="resolve"):
        Engine(ArithmeticModel(), cache, FixedPolicy(0)).run([Request(0, None, 2, prompt_text="Question")], 16)
    requests, _ = resolve_requests([Request(0, None, 3, token_ids=(1, 2))], 16)
    result = Engine(ArithmeticModel(), cache, FixedPolicy(0)).run(requests, 16)
    assert result.tokens.token.tolist() == [3, 4, 5]
    assert result.prefill.prompt_sha256.iloc[0] == requests[0].prompt_sha256
    assert result.metadata["model_dtype"] == "float32"


def test_actual_cpu_decoder_replay_preserves_tokens_across_split_parameters():
    from kernelscope.serve.hf import ModelConfig
    from kernelscope.serve.model import DecoderModel
    from kernelscope.serve.equivalence import compare_policies
    cfg = ModelConfig("qwen3", 1, 16, 32, 2, 1, 8, 32, 10000., None, 1e-6, False, True)
    model = DecoderModel.random(cfg, device="cpu", seed=5)
    rows = compare_policies(model, [FixedPolicy(0), FixedPolicy(8)], [Request(0, 5, 4), Request(1, 2, 3, 1)],
                            cfg.vocab, 2 * PagePool.bytes_per_page(1, 1, 8, torch.float32), record_steps=3)
    assert rows.passed.all() and rows.tokens_identical.all()
    assert rows.max_abs_logit_diff.max() == 0


from kernelscope.serve.engine import OPS_COLUMNS, STEP_COLUMNS


class RegionModel(ArithmeticModel):
    """Scheduler double that opens op regions the way DecoderModel does."""

    def prefill(self, rid, tokens, pool, timer=None):
        reg = timer.region if timer is not None else (lambda *a: contextlib.nullcontext())
        with reg("embed"):
            pool.reserve(rid, len(tokens))
            pool.set_length(rid, len(tokens))
        with reg("lm_head"):
            return self._logits(sum(tokens))

    def decode(self, seq_ids, tokens, pool, num_splits=0, timer=None):
        with timer.region("embed"):
            for rid in seq_ids:
                length = pool.length(rid) + 1
                pool.reserve(rid, length)
                pool.set_length(rid, length)
        with timer.region("attention", 0):
            output = torch.stack([self._logits(token + 1) for token in tokens])
        with timer.region("mlp", 0):
            pass
        return output


def test_event_mode_collects_op_rows_and_keeps_step_columns():
    result = Engine(RegionModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2), Request(1, 2, 2)], 16, ops_mode="event")
    assert list(result.steps.columns) == STEP_COLUMNS
    assert list(result.ops.columns) == OPS_COLUMNS
    decode = result.ops[result.ops.phase == "decode"]
    prefill = result.ops[result.ops.phase == "prefill"]
    assert (decode.rid == "").all() and set(decode.op_class) == {"embed", "attention", "mlp"}
    assert set(prefill.rid) == {"0", "1"} and set(prefill.op_class) == {"embed", "lm_head"}
    per_step = decode[decode.op_class == "attention"].groupby("step").gpu_us.sum()
    assert per_step.index.tolist() == result.steps.step.tolist()
    assert result.metadata["ops_mode"] == "event"


def test_default_mode_has_no_op_rows_and_accepts_the_old_double():
    result = Engine(ArithmeticModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2)], 16)
    assert result.ops.empty and list(result.ops.columns) == OPS_COLUMNS
    assert result.metadata["ops_mode"] is None


def test_unknown_ops_mode_is_rejected():
    with pytest.raises(ValueError, match="ops_mode"):
        Engine(ArithmeticModel(), pool(), FixedPolicy(1)).run([Request(0, 3, 2)], 16, ops_mode="trace")


def test_hybrid_policy_is_refused_outside_its_calibrated_head_shape():
    hybrid = FixedPolicy(0, name="hybrid")          # the real HybridPolicy is calibrated for d=128 fp16/bf16 only
    with pytest.raises(ValueError, match="calibrated for d=128"):
        Engine(ArithmeticModel(), pool(), hybrid).run([Request(0, 3, 2)], 16)


def test_engine_hands_the_policy_attention_backend_to_the_model_and_counts_plan_time_as_selection_cost():
    class BackendModel(ArithmeticModel):
        seen = []
        last_plan_us = 250.0

        def decode(self, seq_ids, tokens, pool, num_splits=0, timer=None, attention="fa2"):
            self.seen.append(attention)
            return super().decode(seq_ids, tokens, pool, num_splits, timer)

    policy = FixedPolicy(0, name="flashinfer_cudacore", attention="flashinfer_cudacore")
    result = Engine(BackendModel(), pool(), policy).run([Request(0, 3, 3)], 16)
    assert BackendModel.seen == ["flashinfer_cudacore"] * 2
    assert (result.steps.policy_us >= 250.0).all() and (result.steps.policy == "flashinfer_cudacore").all()
    plain = Engine(ArithmeticModel(), pool(), FixedPolicy(0)).run([Request(0, 3, 3)], 16)   # no backend kwarg for FA2
    assert (plain.steps.policy_us < 250.0).all()


class AttentionModel(ArithmeticModel):
    """Scheduler double whose decode records the backend and split count it was handed."""
    cfg = SimpleNamespace(vocab=16, n_heads=2, n_kv_heads=1, head_dim=8, n_layers=5)

    def __init__(self):
        self.seen = []

    def decode(self, seq_ids, tokens, pool, num_splits=0, timer=None, attention="fa2"):
        self.seen.append((attention, num_splits))
        return super().decode(seq_ids, tokens, pool, num_splits, timer)


class AlternatingPolicy:
    """Picks a split count and a backend per step, the way a library-agnostic table does."""
    name, last_cache_hit, n_layers = "alternating", False, None

    def __init__(self):
        self.attention, self.calls = "fa2", 0

    def choose(self, lens, n_heads, n_kv_heads):
        self.calls += 1
        self.attention = "flashinfer_cudacore" if self.calls % 2 == 0 else "fa2"
        return 0 if self.attention != "fa2" else 8


def test_step_rows_record_the_backend_each_step_ran_on():
    assert STEP_COLUMNS.index("attention") == STEP_COLUMNS.index("num_splits") + 1
    plain = Engine(AttentionModel(), pool(), FixedPolicy(4)).run([Request(0, 3, 3)], 16)
    assert list(plain.steps.columns) == STEP_COLUMNS
    assert plain.steps.attention.tolist() == ["fa2"] * 2 and plain.steps.num_splits.tolist() == [4, 4]
    model = AttentionModel()
    fixed = Engine(model, pool(), FixedPolicy(0, name="flashinfer", attention="flashinfer")).run([Request(0, 3, 3)], 16)
    assert fixed.steps.attention.tolist() == ["flashinfer"] * 2 == [a for a, _ in model.seen]


def test_step_attention_follows_a_policy_that_changes_backend_between_steps():
    model, policy = AttentionModel(), AlternatingPolicy()
    result = Engine(model, pool(), policy).run([Request(0, 3, 5)], 16)
    assert [a for a, _ in model.seen] == ["fa2", "flashinfer_cudacore", "fa2", "flashinfer_cudacore"]
    assert result.steps.attention.tolist() == [a for a, _ in model.seen]
    assert result.steps.num_splits.tolist() == [n for _, n in model.seen] == [8, 0, 8, 0]


def test_engine_fills_n_layers_only_when_the_policy_asks_for_it():
    asking, preset, plain = AlternatingPolicy(), AlternatingPolicy(), FixedPolicy(0)
    preset.n_layers = 12
    for policy in (asking, preset, plain):
        Engine(AttentionModel(), pool(), policy)
    assert asking.n_layers == 5 and preset.n_layers == 12 and not hasattr(plain, "n_layers")


def test_table_any_policies_are_refused_outside_their_calibrated_head_shape():
    for name in ("table_any", "table_any_p400"):
        with pytest.raises(ValueError, match="calibrated for d=128"):
            Engine(ArithmeticModel(), pool(), FixedPolicy(0, name=name)).run([Request(0, 3, 2)], 16)
