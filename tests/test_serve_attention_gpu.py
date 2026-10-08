"""FlashInfer decode inside the serving model, against flash-attn on the same KV pool (RTX 4090 only)."""
import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")]


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("flashinfer")
    from kernelscope.serve.model import DecoderModel, ModelConfig
    cfg = ModelConfig(arch="qwen3", n_layers=2, hidden=256, intermediate=512, n_heads=8, n_kv_heads=2, head_dim=128,
                      vocab=128, rope_theta=10000., rope_scaling=None, rms_eps=1e-6, tie_embeddings=False, qk_norm=True)
    return DecoderModel.random(cfg, device="cuda", seed=0, dtype=torch.bfloat16)


def _run(model, backend, steps=3):
    from kernelscope.serve.kvcache import PagePool
    pool = PagePool(model.cfg.n_layers, 64, model.cfg.n_kv_heads, model.cfg.head_dim, dtype=torch.bfloat16, device="cuda")
    prompts = {0: [i % 128 for i in range(1, 301)], 1: list(range(5, 30)), 2: [i % 128 for i in range(2, 700)]}   # 300, 25, 698 tokens
    for rid, ids in prompts.items():
        model.prefill(rid, ids, pool)
    logits, plans = [], []
    tokens = [3, 4, 5]
    for _ in range(steps):
        out = model.decode(list(prompts), tokens, pool, num_splits=0, attention=backend)
        logits.append(out.float().cpu())
        plans.append(model.last_plan_us)
        tokens = out.argmax(-1).tolist()
    return torch.stack(logits), plans, pool


@pytest.mark.parametrize("backend", ["flashinfer_cudacore", "flashinfer"])
def test_flashinfer_decode_matches_flash_attn_on_the_same_pool(model, backend):
    reference, ref_plans, _ = _run(model, "fa2")
    candidate, plans, pool = _run(model, backend)
    assert ref_plans == [0.0] * 3 and all(p > 0.0 for p in plans)
    assert torch.allclose(candidate, reference, atol=2e-2, rtol=2e-2), (candidate - reference).abs().max()
    assert candidate.argmax(-1).tolist() == reference.argmax(-1).tolist()
    assert pool.length(0) == 303 and pool.length(2) == 701


def test_flashinfer_engine_run_records_plan_time_as_policy_cost(model):
    from kernelscope.serve.dispatch import make_policy
    from kernelscope.serve.engine import Engine
    from kernelscope.serve.kvcache import PagePool
    from kernelscope.serve.scenarios import Request
    requests = [Request(0, 300, 4), Request(1, 25, 4), Request(2, 698, 4)]
    results = {}
    for spec in ("heuristic", "flashinfer_cudacore"):
        pool = PagePool(model.cfg.n_layers, 64, model.cfg.n_kv_heads, model.cfg.head_dim, dtype=torch.bfloat16, device="cuda")
        results[spec] = Engine(model, pool, make_policy(spec)).run(requests, model.cfg.vocab)
    fi = results["flashinfer_cudacore"].steps
    assert (fi.policy == "flashinfer_cudacore").all() and (fi.policy_us > 50.0).all() and (fi.attn_us > 0).all()
    assert results["heuristic"].tokens.token.tolist() == results["flashinfer_cudacore"].tokens.token.tolist()
