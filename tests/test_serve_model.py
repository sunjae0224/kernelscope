import gc
import subprocess
from dataclasses import replace

import pytest
import torch

from kernelscope.serve.hf import ModelConfig, snapshot_dir
from kernelscope.serve.kvcache import PagePool
from kernelscope.serve.model import AttentionTimer, DecoderModel

TINY = ModelConfig(arch="qwen3", n_layers=2, hidden=24, intermediate=48,
                   n_heads=4, n_kv_heads=2, head_dim=8, vocab=128,
                   rope_theta=10000, rope_scaling=None, rms_eps=1e-6,
                   tie_embeddings=True, qk_norm=True)
GPU_TINY = replace(TINY, hidden=128, intermediate=256, head_dim=128, vocab=1000)


@pytest.fixture(autouse=True)
def _small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 2))
    yield
    torch.set_num_threads(previous)


def _pool(model, pages=8):
    cfg = model.cfg
    return PagePool(cfg.n_layers, pages, cfg.n_kv_heads, cfg.head_dim,
                    dtype=model.dtype, device=model.device)


def _hf_model(model):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen3Config, Qwen3ForCausalLM
    cfg = model.cfg
    config_class, model_class = ((Qwen3Config, Qwen3ForCausalLM) if cfg.arch == "qwen3"
                                 else (LlamaConfig, LlamaForCausalLM))
    config = config_class(
        num_hidden_layers=cfg.n_layers, hidden_size=cfg.hidden, intermediate_size=cfg.intermediate,
        num_attention_heads=cfg.n_heads, num_key_value_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
        vocab_size=cfg.vocab, rope_theta=cfg.rope_theta, rope_scaling=cfg.rope_scaling,
        rms_norm_eps=cfg.rms_eps, tie_word_embeddings=cfg.tie_embeddings,
        max_position_embeddings=4096, attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    reference = model_class(config).eval()
    weights = dict(model.w)
    weights["lm_head.weight"] = model.lm_head
    reference.load_state_dict(weights, strict=True)
    return reference


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_prefill_and_ragged_decode_match_independent_hf_oracle(arch):
    config = TINY if arch == "qwen3" else replace(
        TINY, arch="llama", tie_embeddings=False, qk_norm=False,
        rope_scaling=dict(rope_type="llama3", factor=8, low_freq_factor=1,
                          high_freq_factor=4, original_max_position_embeddings=16))
    model = DecoderModel.random(config, device="cpu", seed=42)
    reference = _hf_model(model)
    pool = _pool(model)
    histories = {"long": [i % config.vocab for i in range(270)], "short": [7, 9, 14]}
    with torch.inference_mode():
        for name, ids in histories.items():
            got = model.prefill(name, ids, pool, chunk=61)
            want = reference(torch.tensor([ids]), use_cache=False).logits[0, -1]
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)
        # A new page boundary, unequal absolute RoPE positions, and reordered batch rows.
        for step in range(3):
            ids = ["short", "long"]
            inputs = [20 + step, 60 + step]
            got = model.decode(ids, inputs, pool, num_splits=8)
            for row, (name, token) in enumerate(zip(ids, inputs)):
                histories[name].append(token)
                want = reference(torch.tensor([histories[name]]), use_cache=False).logits[0, -1]
                torch.testing.assert_close(got[row], want, atol=2e-6, rtol=2e-5)
                assert pool.length(name) == len(histories[name])


def test_chunking_and_page_boundary_preserve_logits_and_tokens():
    model = DecoderModel.random(TINY, device="cpu")
    ids = [i % TINY.vocab for i in range(257)]
    full_pool, incremental_pool = _pool(model), _pool(model)
    expected = model.prefill("full", ids, full_pool)
    model.prefill("incremental", ids[:-1], incremental_pool, chunk=37)
    timer = AttentionTimer(device="cpu")
    actual = model.decode(["incremental"], [ids[-1]], incremental_pool, timer=timer)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert actual.argmax() == expected.argmax()
    assert incremental_pool.length("incremental") == 257
    assert timer.total_us() > 0
    assert model.backend == "torch_sdpa_reference"


def test_prefill_allows_capacity_reserved_by_scheduler():
    model = DecoderModel.random(TINY, device="cpu")
    pool = _pool(model)
    pool.reserve("a", 800)
    model.prefill("a", [1, 2, 3], pool)
    assert pool.capacity("a") == 1024 and pool.length("a") == 3


def test_decode_allocation_failure_does_not_advance_any_sequence():
    model = DecoderModel.random(TINY, device="cpu")
    pool = _pool(model, pages=2)
    for name in ("a", "b"):
        model.prefill(name, [1] * 256, pool)
    with pytest.raises(MemoryError):
        model.decode(["a", "b"], [2, 3], pool)
    assert pool.lengths(["a", "b"]).tolist() == [256, 256] and pool.free_pages == 0


def test_invalid_inputs_fail_before_cache_mutation():
    model = DecoderModel.random(TINY, device="cpu")
    pool = _pool(model)
    for tokens in ([], [-1], [TINY.vocab], [1.5]):
        with pytest.raises(ValueError):
            model.prefill("bad", tokens, pool)
    assert pool.free_pages == pool.n_pages
    model.prefill("ok", [1, 2], pool)
    with pytest.raises(ValueError, match="distinct"):
        model.decode(["ok", "ok"], [1, 2], pool)
    with pytest.raises(ValueError, match="one input"):
        model.decode(["ok"], [], pool)
    with pytest.raises(ValueError, match="num_splits"):
        model.decode(["ok"], [1], pool, num_splits=129)
    with pytest.raises(ValueError, match="fresh"):
        model.prefill("ok", [1], pool)
    assert pool.length("ok") == 2


def test_checkpoint_shape_validation():
    model = DecoderModel.random(TINY, device="cpu")
    bad = dict(model.w)
    bad["model.norm.weight"] = torch.ones(25)
    with pytest.raises(ValueError, match="expected shape"):
        DecoderModel(TINY, bad, device="cpu")


def test_timer_detects_missing_and_unfinished_regions():
    timer = AttentionTimer(device="cpu")
    assert timer.total_us() == 0
    with pytest.raises(RuntimeError, match="not started"):
        timer.stop(1)
    timer.start(1)
    with pytest.raises(RuntimeError, match="unfinished"):
        timer.total_us()
    timer.stop(1)
    assert timer.total_us() >= 0


def _require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    try:
        subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
                       capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("GPU process inventory is unavailable")
    if torch.cuda.mem_get_info()[0] < 18 * 2**30:
        pytest.skip("functional GPU tests require at least 18 GiB free")
    pytest.importorskip("flash_attn")


@pytest.mark.gpu
@pytest.mark.parametrize("splits", [0, 1, 4, 16])
def test_gpu_paged_split_output_equivalence(splits):
    _require_gpu()
    model = DecoderModel.random(GPU_TINY)
    results = []
    for chosen in (1, splits):
        pool = _pool(model, pages=16)
        for name, length in (("long", 700), ("short", 30), ("medium", 300)):
            model.prefill(name, list(range(1, length + 1)), pool, chunk=131)
        timer = AttentionTimer("cuda")
        results.append(model.decode(["short", "long", "medium"], [2, 3, 4], pool,
                                    num_splits=chosen, timer=timer))
        assert timer.total_us() > 0
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0.05)
    assert torch.equal(results[0].argmax(-1), results[1].argmax(-1))


@pytest.mark.gpu
def test_gpu_prefill_chunking_matches_decode_continuation():
    _require_gpu()
    model = DecoderModel.random(GPU_TINY)
    ids = list(range(1, 290))
    full_pool, step_pool = _pool(model), _pool(model)
    full = model.prefill("a", ids, full_pool, chunk=67)
    model.prefill("b", ids[:-1], step_pool)
    step = model.decode(["b"], [ids[-1]], step_pool)[0]
    torch.testing.assert_close(full, step, rtol=0, atol=0.05)
    assert step_pool.length("b") == len(ids)


@pytest.mark.gpu
def test_local_qwen4b_prefill_and_decode_match_transformers():
    _require_gpu()
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM
    try:
        path = snapshot_dir("Qwen/Qwen3-4B-Instruct-2507")
    except FileNotFoundError:
        pytest.skip("optional local Qwen3 4B checkpoint is absent")
    prompt, continuation = list(range(100, 132)), [800, 801]
    reference = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True, trust_remote_code=False,
    ).cuda().eval()
    expected = []
    with torch.inference_mode():
        for n in range(len(continuation) + 1):
            ids = torch.tensor([prompt + continuation[:n]], device="cuda")
            expected.append(reference(ids, use_cache=False).logits[0, -1].float().cpu())
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    model = DecoderModel.from_pretrained(str(path))
    pool = _pool(model, pages=2)
    actual = [model.prefill("real", prompt, pool, chunk=13).cpu()]
    for token in continuation:
        actual.append(model.decode(["real"], [token], pool, num_splits=8)[0].cpu())
    for got, want in zip(actual, expected):
        cosine = F_cosine(got, want)
        assert cosine > 0.999, cosine
        assert got.argmax() == want.argmax()
        assert torch.isfinite(got).all()
    del model, pool
    gc.collect()
    torch.cuda.empty_cache()


def F_cosine(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


from kernelscope.serve.model import NULL_REGION, OP_CLASSES, OpTimer


def _prefilled(seed=0):
    model = DecoderModel.random(TINY, device="cpu", seed=seed)
    pool = _pool(model)
    model.prefill(0, list(range(1, 6)), pool)
    model.prefill(1, list(range(1, 9)), pool)
    return model, pool


def test_op_timer_records_every_class_for_every_layer_on_decode():
    model, pool = _prefilled()
    timer = OpTimer(device="cpu")
    model.decode([0, 1], [3, 4], pool, timer=timer)
    rows = timer.rows()
    layers = {}
    for row in rows:
        layers.setdefault(row["op_class"], set()).add(row["layer"])
    assert set(layers) == set(OP_CLASSES)
    for op_class in ("norm", "qkv_proj", "rope", "attention", "o_proj", "mlp"):
        assert layers[op_class] == set(range(TINY.n_layers))
    assert layers["embed"] == {-1} and layers["lm_head"] == {-1}
    assert all(row["gpu_us"] >= 0 for row in rows)
    attention = sum(row["gpu_us"] for row in rows if row["op_class"] == "attention")
    assert timer.total_us("attention") == pytest.approx(attention)


def test_prefill_accepts_a_timer_and_records_one_embed_per_chunk():
    model = DecoderModel.random(TINY, device="cpu")
    pool = _pool(model)
    timer = OpTimer(device="cpu")
    model.prefill(0, list(range(1, 12)), pool, chunk=4, timer=timer)
    rows = timer.rows()
    assert {row["op_class"] for row in rows} == set(OP_CLASSES)
    assert sum(row["op_class"] == "embed" for row in rows) == 3      # 11 tokens in chunks of 4


def test_attention_timer_keeps_layer_api_and_skips_other_classes():
    timer = AttentionTimer(device="cpu")
    assert timer.region("mlp", 0) is NULL_REGION
    timer.start(0)
    timer.stop(0)
    with timer.region("attention", 1):
        pass
    assert timer.total_us() >= 0
    assert {row["layer"] for row in timer.rows()} == {0, 1}
    timer.start(2)
    with pytest.raises(RuntimeError, match="already running"):
        timer.start(2)


def test_op_timer_rejects_unknown_classes_and_unfinished_reads():
    with pytest.raises(ValueError, match="unknown op class"):
        OpTimer(device="cpu", classes=("attention", "softmax"))
    timer = OpTimer(device="cpu")
    timer.start("mlp", 0)
    with pytest.raises(RuntimeError, match="unfinished"):
        timer.rows()


def test_instrumentation_does_not_change_decode_logits():
    model, pool = _prefilled()
    plain = model.decode([0, 1], [3, 4], pool)
    model2, pool2 = _prefilled()
    timed = model2.decode([0, 1], [3, 4], pool2, timer=OpTimer(device="cpu"))
    assert torch.equal(plain, timed)
