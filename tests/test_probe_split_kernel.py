import importlib
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import torch

from kernelscope.serve.kvcache import PAGE

MODULE = "scripts.probe_split_kernel"


def _naive_reference(q, kcache, vcache, k_new, v_new, cache_lens, table, scale):
    """The previous implementation: expand K/V to the query-head count with repeat_interleave."""
    batch, _, hq, d = q.shape
    hk = kcache.shape[2]
    group = hq // hk
    outs = []
    for b in range(batch):
        n = int(cache_lens[b])
        pages = table[b, : (n + PAGE - 1) // PAGE].long()
        k = torch.cat([kcache[pages].reshape(-1, hk, d)[:n].float(), k_new[b].float()], dim=0)
        v = torch.cat([vcache[pages].reshape(-1, hk, d)[:n].float(), v_new[b].float()], dim=0)
        kk = k.repeat_interleave(group, dim=1)
        vv = v.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,nhd->hn", q[b, 0].float(), kk) * scale
        outs.append(torch.einsum("hn,nhd->hd", torch.softmax(scores, dim=-1), vv))
    return torch.stack(outs).unsqueeze(1)


def _fake_attention(q, kcache, vcache, k, v, cache_seqlens, block_table, softmax_scale, bump_row=None):
    """Stand-in for a paged decode kernel (bf16 output). With k/v it first appends them at cache_seqlens like the
    kernel does; `bump_row` adds 0.25 to that batch row of the output."""
    batch, _, hq, d = q.shape
    hk = kcache.shape[2]
    if k is not None:
        for b in range(batch):
            n = int(cache_seqlens[b])
            page = int(block_table[b, n // PAGE])
            kcache[page, n % PAGE], vcache[page, n % PAGE] = k[b, 0], v[b, 0]
    outs = []
    for b in range(batch):
        n = int(cache_seqlens[b]) + (1 if k is not None else 0)
        pages = block_table[b, : (n + PAGE - 1) // PAGE].long()
        kk = kcache[pages].reshape(-1, hk, d)[:n].float().repeat_interleave(hq // hk, 1)
        vv = vcache[pages].reshape(-1, hk, d)[:n].float().repeat_interleave(hq // hk, 1)
        probs = torch.softmax(torch.einsum("hd,nhd->hn", q[b, 0].float(), kk) * softmax_scale, -1)
        out = torch.einsum("hn,nhd->hd", probs, vv)
        outs.append((out + (0.25 if b == bump_row else 0.0)).to(torch.bfloat16))
    return torch.stack(outs).unsqueeze(1)


def _make_fake_flash(bump_splits=None, bump_row=2):
    """flash_attn_with_kvcache stand-in; the output of num_splits == bump_splits is off by 0.25 on `bump_row`."""
    def fake_flash(q, kcache, vcache, k=None, v=None, cache_seqlens=None, block_table=None, softmax_scale=None,
                   causal=True, num_splits=0):
        return _fake_attention(q, kcache, vcache, k, v, cache_seqlens, block_table, softmax_scale,
                               bump_row if num_splits == bump_splits else None)
    return fake_flash


class _FakeBackend:
    """FlashInferDecode stand-in: records plan / call arguments; the output is attention of q, off by 0.25 on row 2."""
    def __init__(self):
        self.plans, self.calls = [], []

    def plan(self, cache_lens, block_table, n_heads, n_kv_heads, head_dim, dtype):
        self.plans.append(dict(cache_lens=cache_lens.clone(), block_table=block_table,
                               shape=(n_heads, n_kv_heads, head_dim), dtype=dtype))
        return 0.0

    def __call__(self, q, k_cache, v_cache, *, k, v, cache_seqlens, block_table, softmax_scale, causal, num_splits):
        self.calls.append(dict(k=k, v=v, cache_seqlens=cache_seqlens, block_table=block_table, num_splits=num_splits))
        return _fake_attention(q, k_cache, v_cache, k, v, cache_seqlens, block_table, softmax_scale, bump_row=2)


@pytest.mark.parametrize("hq,hk", [(8, 2), (6, 1), (4, 4)])
def test_fp32_reference_matches_repeat_interleave_implementation(hq, hk):
    from scripts.probe_split_kernel import fp32_reference
    torch.manual_seed(0)
    d, batch, n_pages = 16, 3, 5
    kcache = torch.randn(n_pages, PAGE, hk, d).to(torch.bfloat16)
    vcache = torch.randn(n_pages, PAGE, hk, d).to(torch.bfloat16)
    q = torch.randn(batch, 1, hq, d).to(torch.bfloat16)
    k_new = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    v_new = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    cache_lens = torch.tensor([5, 300, 17], dtype=torch.int32)      # 1 page, 2 pages, 1 page
    table = torch.tensor([[2, 0], [4, 1], [3, 0]], dtype=torch.int32)
    scale = d ** -0.5
    got = fp32_reference(q, kcache, vcache, k_new, v_new, cache_lens, table, scale)
    want = _naive_reference(q, kcache, vcache, k_new, v_new, cache_lens, table, scale)
    assert got.shape == (batch, 1, hq, d) and got.dtype == torch.float32
    assert (got - want).abs().max().item() < 1e-5


def test_parse_args_defaults_reproduce_the_original_probe():
    from scripts.probe_split_kernel import parse_args
    a = parse_args([])
    campaign = Path("../kernelscope/results/serve_4090/hybrid_20261002")
    assert a.campaign == campaign and a.run == "control_uniform_fixed8"
    assert (a.target_step, a.target_rid, a.candidate_splits) == (40, 8, 8)
    assert a.candidate_backend == "fa2"
    assert int(a.kv_gib * 2**30) == 10 * 2**30
    assert a.model == "Qwen/Qwen3-4B-Instruct-2507"
    assert a.out == campaign / "numerics" / "control_uniform_fixed8" / "kernel_probe"


def test_parse_args_accepts_the_64k_event():
    from scripts.probe_split_kernel import parse_args
    campaign = "../kernelscope/results/serve_4090/longctx_20261006"
    a = parse_args(["--campaign", campaign, "--run", "ragged_64k", "--target-step", "26", "--target-rid", "0",
                    "--kv-gib", "13"])
    assert (a.run, a.target_step, a.target_rid, a.candidate_splits, a.kv_gib) == ("ragged_64k", 26, 0, 8, 13.0)
    assert a.candidate_backend == "fa2"
    assert a.out == Path(campaign) / "numerics" / "ragged_64k" / "kernel_probe"
    assert int(a.kv_gib * 2**30) == 13 * 2**30
    assert parse_args(["--out", "somewhere"]).out == Path("somewhere")


@pytest.mark.parametrize("bad", [["--candidate-splits", "1"], ["--target-step", "-1"], ["--kv-gib", "0"],
                                 ["--candidate-backend", "triton"]])
def test_parse_args_rejects_invalid_values(bad):
    from scripts.probe_split_kernel import parse_args
    with pytest.raises(SystemExit):
        parse_args(bad)


@pytest.mark.parametrize("backend", ["flashinfer", "flashinfer_cudacore"])
def test_parse_args_flashinfer_backend_renames_the_output_and_ignores_splits(backend):
    from scripts.probe_split_kernel import decode_kwargs, parse_args
    campaign = "../kernelscope/results/serve_4090/flashinfer_20261006"
    a = parse_args(["--campaign", campaign, "--run", "ragged", "--target-step", "40", "--target-rid", "8",
                    "--candidate-backend", backend, "--kv-gib", "10"])
    assert (a.candidate_backend, a.candidate_splits, a.target_step, a.target_rid) == (backend, None, 40, 8)
    assert a.out == Path(campaign) / "numerics" / "ragged" / f"kernel_probe_{backend}"
    assert decode_kwargs(a) == dict(num_splits=0, attention=backend)
    ignored = parse_args(["--candidate-backend", backend, "--candidate-splits", "1"])   # not validated, not used
    assert ignored.candidate_splits is None and ignored.out.name == f"kernel_probe_{backend}"
    assert parse_args(["--candidate-backend", backend, "--out", "somewhere"]).out == Path("somewhere")
    assert decode_kwargs(parse_args(["--candidate-splits", "4"])) == dict(num_splits=4)


def test_check_reference_requires_every_rid_at_every_position():
    from scripts.probe_split_kernel import check_reference
    tokens = {(rid, pos): 1 for rid in (0, 5) for pos in range(4)}
    check_reference(tokens, [0, 5], 3)
    del tokens[(5, 2)]
    with pytest.raises(ValueError, match=r"\(5, 2\)"):
        check_reference(tokens, [0, 5], 3)
    with pytest.raises(ValueError, match="too short"):
        check_reference({(0, 0): 1}, [0], 1)


def test_importing_the_module_does_not_need_flash_attn(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn", None)             # any `import flash_attn` now raises ImportError
    with pytest.raises(ImportError):
        import flash_attn  # noqa: F401
    monkeypatch.delitem(sys.modules, MODULE, raising=False)
    try:
        module = importlib.import_module(MODULE)
        assert callable(module.main) and callable(module.parse_args) and callable(module.fp32_reference)
        assert sys.modules["flash_attn"] is None
    finally:
        sys.modules.pop(MODULE, None)


def test_probing_attention_records_the_target_rows_by_batch_index():
    from scripts.probe_split_kernel import make_probing_attention
    torch.manual_seed(1)
    hq, hk, d, batch, candidate = 8, 2, 16, 3, 3
    kcache = torch.randn(5, PAGE, hk, d).to(torch.bfloat16)
    vcache = torch.randn(5, PAGE, hk, d).to(torch.bfloat16)
    table = torch.tensor([[2, 0], [4, 1], [3, 0]], dtype=torch.int32)
    fake_flash = _make_fake_flash(bump_splits=candidate)

    records, state = [], {"probe": True, "layer": 0}
    attention = make_probing_attention(fake_flash, candidate, 2, records, state)
    q = torch.randn(batch, 1, hq, d).to(torch.bfloat16)
    k = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    v = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    attention(q, kcache, vcache, k=k, v=v, cache_seqlens=torch.tensor([5, 300, 17], dtype=torch.int32),
              block_table=table, softmax_scale=d ** -0.5)
    by_variant = {r["variant"]: r for r in records}
    assert list(by_variant) == ["split1", "split3", "kernel_as_called"] and state["layer"] == 1
    assert by_variant["split1"]["max_abs_err"] < 0.01 and by_variant["split1"]["exact_equal_split1"] is True
    # only the batch row at index 2 differs between split 1 and the candidate, so the *_target columns see it
    assert by_variant["split3"]["max_abs_err_target"] > 0.2
    assert by_variant["split3"]["split1_vs_candidate_max_target"] == by_variant["split3"]["split1_vs_candidate_max"] > 0.2
    assert by_variant["split1"]["max_abs_err_target"] < 0.01


def test_probing_attention_with_a_backend_candidate_plans_once_per_step():
    from scripts.probe_split_kernel import BackendCandidate, make_probing_attention
    torch.manual_seed(1)
    hq, hk, d, batch = 8, 2, 16, 3
    kcache = torch.randn(5, PAGE, hk, d).to(torch.bfloat16)
    vcache = torch.randn(5, PAGE, hk, d).to(torch.bfloat16)
    table = torch.tensor([[2, 0], [4, 1], [3, 0]], dtype=torch.int32)
    lens = torch.tensor([5, 300, 17], dtype=torch.int32)
    q = torch.randn(batch, 1, hq, d).to(torch.bfloat16)
    k = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    v = torch.randn(batch, 1, hk, d).to(torch.bfloat16)
    backend = _FakeBackend()
    records, state = [], {"probe": True, "layer": 0}
    attention = make_probing_attention(_make_fake_flash(), BackendCandidate("flashinfer_cudacore", backend), 2,
                                       records, state)

    def layer_call():
        return attention(q, kcache, vcache, k=k, v=v, cache_seqlens=lens, block_table=table, softmax_scale=d ** -0.5)

    out = layer_call()
    layer_call()                                                     # the second probed layer of the same step
    assert out.shape == (batch, 1, hq, d) and state["layer"] == 2
    # one plan for the step with the pre-append lengths and the block table; one candidate call per probed layer
    assert len(backend.plans) == 1 and len(backend.calls) == 2
    plan = backend.plans[0]
    assert torch.equal(plan["cache_lens"], lens) and plan["block_table"] is table
    assert plan["shape"] == (hq, hk, d) and plan["dtype"] == torch.bfloat16
    for call in backend.calls:
        assert call["k"] is k and call["v"] is v and call["cache_seqlens"] is lens   # pre-append lengths, k/v rewritten
        assert call["block_table"] is table and call["num_splits"] == 0
    assert [(r["layer"], r["variant"]) for r in records] == [
        (layer, name) for layer in (0, 1) for name in ("split1", "flashinfer_cudacore", "kernel_as_called")]
    assert {r["num_splits_called"] for r in records} == {0}
    by_variant = {r["variant"]: r for r in records if r["layer"] == 0}
    # only batch row 2 differs between flash-attn and the backend, and the target index selects that row
    assert by_variant["split1"]["max_abs_err_target"] < 0.01 and by_variant["split1"]["exact_equal_split1"] is True
    assert by_variant["flashinfer_cudacore"]["max_abs_err_target"] > 0.2
    assert by_variant["flashinfer_cudacore"]["split1_vs_candidate_max_target"] == \
        by_variant["flashinfer_cudacore"]["split1_vs_candidate_max"] > 0.2
    # a new step starts at layer 0 again and is planned again; outside a probed step nothing is planned or recorded
    state["layer"] = 0
    layer_call()
    assert len(backend.plans) == 2 and len(backend.calls) == 3 and len(records) == 9
    state["probe"] = False
    layer_call()
    assert len(backend.plans) == 2 and len(backend.calls) == 3 and len(records) == 9
    # a different target row does not see the backend's error
    other, other_state = [], {"probe": True, "layer": 0}
    make_probing_attention(_make_fake_flash(), BackendCandidate("flashinfer", _FakeBackend()), 0, other,
                           other_state)(q, kcache, vcache, k=k, v=v, cache_seqlens=lens, block_table=table,
                                        softmax_scale=d ** -0.5)
    row = next(r for r in other if r["variant"] == "flashinfer")
    assert row["max_abs_err"] > 0.2 and row["max_abs_err_target"] < 0.01
    assert row["split1_vs_candidate_max"] > 0.2 and row["split1_vs_candidate_max_target"] < 0.01


class _FakeModel:
    """DecoderModel stand-in for main(): a 2-layer decode over fixed tensors through model._flash_attention or backend."""
    hq, hk, d = 8, 2, 16
    instance = None

    def __init__(self):
        torch.manual_seed(2)
        self.device, self.dtype, self.cfg = torch.device("cpu"), torch.bfloat16, types.SimpleNamespace()
        self.kcache = torch.randn(5, PAGE, self.hk, self.d).to(torch.bfloat16)
        self.vcache = torch.randn(5, PAGE, self.hk, self.d).to(torch.bfloat16)
        self.table = torch.tensor([[2, 0], [4, 1], [3, 0]], dtype=torch.int32)
        self.lens = torch.tensor([5, 300, 17], dtype=torch.int32)
        self._flash_attention = _make_fake_flash(bump_splits=8)
        self.backends, self.decodes = {}, []
        _FakeModel.instance = self

    @classmethod
    def from_pretrained(cls, repo_id):
        return cls()

    def attention_backend(self, name):
        if name == "fa2":
            return self._flash_attention
        return self.backends.setdefault(name, _FakeBackend())

    def prefill(self, seq_id, token_ids, pool):
        pass

    def decode(self, seq_ids, token_ids, pool, num_splits=0, timer=None, attention="fa2"):
        self.decodes.append(dict(num_splits=num_splits, attention=attention))
        fn = self.attention_backend(attention)
        if attention != "fa2":
            fn.plan(self.lens, self.table, self.hq, self.hk, self.d, self.dtype)
        q = torch.randn(3, 1, self.hq, self.d).to(torch.bfloat16)
        k = torch.randn(3, 1, self.hk, self.d).to(torch.bfloat16)
        v = torch.randn(3, 1, self.hk, self.d).to(torch.bfloat16)
        for _ in range(2):
            fn(q, self.kcache, self.vcache, k=k, v=v, cache_seqlens=self.lens, block_table=self.table,
               softmax_scale=self.d ** -0.5, causal=True, num_splits=num_splits)
        return torch.randn(3, 32)


class _FakePool:
    @classmethod
    def for_budget(cls, cfg, budget, device=None, dtype=None):
        return cls()

    def length(self, rid):
        return 5

    def set_length(self, rid, n):
        pass


@pytest.mark.parametrize("backend,label,splits", [("fa2", "split8", 8), ("flashinfer_cudacore", "flashinfer_cudacore", None)])
def test_main_runs_the_selected_candidate_and_writes_labelled_artifacts(tmp_path, monkeypatch, capsys,
                                                                       backend, label, splits):
    probe = importlib.import_module(MODULE)
    monkeypatch.setattr(probe, "DecoderModel", _FakeModel)
    monkeypatch.setattr(probe, "PagePool", _FakePool)
    for name in ("flash_attn", "flashinfer"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        sys.modules[name].__version__ = f"fake-{name}"
    rids = [4, 9, 8]                                                  # the target rid 8 is batch row 2
    campaign = tmp_path / "campaign"
    (campaign / "run").mkdir(parents=True)
    (campaign / "numerics" / "run").mkdir(parents=True)
    (campaign / "run" / "prompts.jsonl").write_text(
        "\n".join(json.dumps({"rid": rid, "token_ids": [1, 2, 3]}) for rid in rids))
    pd.DataFrame([dict(rid=rid, position=pos, token=1) for rid in rids for pos in range(3)]).to_parquet(
        campaign / "numerics" / "run" / "reference_tokens.parquet")
    argv = ["--campaign", str(campaign), "--run", "run", "--target-step", "2", "--target-rid", "8", "--kv-gib", "1",
            "--candidate-backend", backend]
    probe.main(argv)

    model = _FakeModel.instance
    kernel_probe = "kernel_probe" if backend == "fa2" else f"kernel_probe_{backend}"
    out = campaign / "numerics" / "run" / kernel_probe
    manifest = json.loads((out / "manifest.json").read_text())
    assert (manifest["candidate_backend"], manifest["candidate_splits"]) == (backend, splits)
    assert ("flashinfer" in manifest) == (backend != "fa2") and manifest["flash_attn"] == "fake-flash_attn"
    layers = pd.read_csv(out / "attention_vs_fp32.csv")
    assert sorted(layers.variant.unique()) == sorted(["split1", label, "kernel_as_called"])
    assert sorted(layers.layer.unique()) == [0, 1]
    # the target (batch row 2) is the one row where the candidate is off by 0.25 in both fake candidates
    assert (layers[layers.variant == label].max_abs_err_target > 0.2).all()
    assert (layers[layers.variant == "split1"].max_abs_err_target < 0.01).all()
    assert len(pd.read_csv(out / "single_step_logits.csv")) == 3
    # steps 0 and 1 are plain decodes; the target step runs the candidate first, then the probed default decode
    candidate_decode = dict(num_splits=8, attention="fa2") if backend == "fa2" else dict(num_splits=0, attention=backend)
    assert model.decodes == [dict(num_splits=0, attention="fa2")] * 2 + [candidate_decode,
                                                                          dict(num_splits=0, attention="fa2")]
    if backend != "fa2":    # once in the candidate decode, once on the first probed layer
        assert len(model.backends[backend].plans) == 2
    assert label in capsys.readouterr().out
