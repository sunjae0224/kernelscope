"""The sample CUDA binary (samples/naive_attn) registered through ExecutablePlugin."""
from pathlib import Path

import pytest
import torch

from kernelscope.plugins.builtin import REGISTRY
from kernelscope.plugins.builtin.naive_exec import NaiveDecodeExec, hash_init
from kernelscope.workload import Workload

W = Workload(phase="decode", B=2, L_q=1, L_kv=1024, H_q=32, H_kv=8, d=128, dtype="float32")
BIN = Path(__file__).resolve().parents[1] / "samples" / "naive_attn" / "naive_attn"


def _c_init(i: int, seed: int) -> float:
    # literal transcription of init_val() in naive_attn.cu (uint32 arithmetic)
    x = ((i + seed * 1000003) * 2654435761) & 0xFFFFFFFF
    return (x & 0xFFFF) / 65536.0 - 0.5


def test_registered_and_decode_fp32_only():
    assert "naive_exec" in REGISTRY.names()
    p = NaiveDecodeExec()
    assert p.supports(W)
    assert not p.supports(Workload(**{**W.__dict__, "dtype": "float16"}))
    assert not p.supports(Workload(phase="prefill", B=1, L_q=64, L_kv=64, H_q=8, H_kv=8, d=64, dtype="float32"))


def test_command_encodes_workload_iters_and_output():
    argv = NaiveDecodeExec().command(W, iters=7, out_path="/tmp/o.bin")
    assert argv[0].endswith("naive_attn")
    assert argv[1:] == ["2", "1024", "32", "8", "128", "7", "/tmp/o.bin"]
    assert NaiveDecodeExec().command(W)[-1] == "1"


def test_hash_init_matches_the_c_formula():
    t = hash_init(5, seed=2)
    assert t.dtype == torch.float32
    assert t.tolist() == pytest.approx([_c_init(i, 2) for i in range(5)])


def test_reference_inputs_shapes_and_seeds():
    q, k, v = NaiveDecodeExec().reference_inputs(W)
    assert tuple(q.shape) == (2, 1, 32, 128)
    assert tuple(k.shape) == (2, 1024, 8, 128) and tuple(v.shape) == (2, 1024, 8, 128)
    assert q.flatten()[0].item() == pytest.approx(_c_init(0, 1))
    assert k.flatten()[0].item() == pytest.approx(_c_init(0, 2))
    assert not torch.equal(k, v)


@pytest.mark.gpu
def test_binary_output_matches_reference(tmp_path):
    if not BIN.exists() or not torch.cuda.is_available():
        pytest.skip("needs the compiled sample and a GPU")
    from kernelscope.backends.realhw.executable import check_executable
    r = check_executable(NaiveDecodeExec(), W, tmp_path / "out.bin", atol=1e-3)
    assert r["ok"], r
