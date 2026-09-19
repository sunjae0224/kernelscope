import json

import pytest

from kernelscope.backends.realhw import batch
from kernelscope.results.store import ResultStore
from kernelscope.workload import Workload
from tests.fake_plugins import FakeExec, FaithfulCPU

W1 = Workload(phase="decode", B=2, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32")
W2 = Workload(phase="decode", B=3, L_q=1, L_kv=64, H_q=8, H_kv=2, d=16, dtype="float32", kv_lens=[64, 9, 30])
FAKE_PROF = {"launches_per_iter": 1, "kernel_time_us_median": 12.5, "unmatched": [], "device_props": None,
             "iterations_used": 3, "iterations_dropped": 1,
             "launches": [{"idx": 0, "name": "k", "dur_us_median": 12.5, "grid": (1, 8, 2), "block": (128, 1, 1),
                           "regs": 64, "smem_bytes": 0, "occupancy": {"blocks": 16, "sm_coverage": 0.125}}]}


@pytest.fixture
def fake_profile(monkeypatch):
    monkeypatch.setattr(batch, "profile_launches", lambda *a, **k: dict(FAKE_PROF))


def _run(tmp_path, plugins, workloads, states=("warm", "cold"), **kw):
    store = ResultStore(tmp_path / "res")
    summ = tmp_path / "res" / "summaries.jsonl"
    out = batch.run_bench(plugins, workloads, list(states), store, summ, device="cpu",
                          warmup=1, iters=3, pad=1, log=None, **kw)
    return out, store.load(), summ


def test_bench_writes_one_row_set_per_cache_state_with_the_state_column(tmp_path, fake_profile):
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1, W2])
    assert [s["status"] for s in out] == ["ok"] * 4
    kt = df[(df.backend == "profile") & (df.metric == "kernel_time_us")]
    assert sorted(kt.cache_state) == ["cold", "cold", "warm", "warm"]
    assert (df.runner == "bench").all()
    checks = df[df.backend == "check"]
    assert len(checks[checks.metric == "ok"]) == 2            # once per workload, not per state
    assert set(df[df.metric == "iterations_dropped"].value) == {1.0}


def test_bench_resumes_without_remeasuring(tmp_path, fake_profile):
    _run(tmp_path, [FaithfulCPU(device="cpu")], [W1])
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1])
    assert [s["status"] for s in out] == ["skipped", "skipped"]
    assert len(df[(df.backend == "profile") & (df.metric == "kernel_time_us")]) == 2


def test_executables_and_unsupported_workloads_are_reported_not_measured(tmp_path, fake_profile):
    out, _, _ = _run(tmp_path, [FakeExec(device="cpu")], [W1], states=("cold",))
    assert out[0]["status"] == "unsupported"


class Exploding(FaithfulCPU):
    name = "exploding"

    def run(self, inputs):
        raise RuntimeError("boom")


def test_a_failing_cell_does_not_stop_the_batch(tmp_path, fake_profile):
    out, _, summ = _run(tmp_path, [Exploding(device="cpu"), FaithfulCPU(device="cpu")], [W1], states=("cold",))
    assert [s["status"] for s in out] == ["error", "ok"]
    assert "boom" in out[0]["error"]
    lines = [json.loads(l) for l in summ.read_text().splitlines()]
    assert len(lines) == 2


def test_check_is_skipped_when_the_reference_would_be_too_large(tmp_path, fake_profile):
    out, df, _ = _run(tmp_path, [FaithfulCPU(device="cpu")], [W1], states=("cold",), check_max_ref_bytes=10)
    assert out[0]["check_ok"] is None
    assert df[df.backend == "check"].empty


@pytest.mark.gpu
def test_bench_on_real_flash_kernels(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("flash_attn")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from kernelscope.plugins.builtin import REGISTRY
    ws = [Workload(phase="decode", B=1, L_q=1, L_kv=4096, H_q=32, H_kv=8, d=128),
          Workload(phase="decode", B=4, L_q=1, L_kv=2048, H_q=32, H_kv=8, d=128, kv_lens=[2048, 100, 100, 100])]
    plugins = [REGISTRY.get(n, device="cuda") for n in ("fa2", "fd_s8_paged")]
    out = batch.run_bench(plugins, ws, ["cold"], ResultStore(tmp_path), tmp_path / "s.jsonl", log=None)
    assert all(s["status"] == "ok" and s["kernel_time_us"] > 0 and s["check_ok"] for s in out), out
