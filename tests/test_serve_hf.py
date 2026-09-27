import json
from dataclasses import replace

import pytest
import torch

from kernelscope.serve.hf import ModelConfig, load_tokenizer, load_weights, rope_inv_freq, snapshot_dir


def _config(tmp_path, **updates):
    config = dict(architectures=["Qwen3ForCausalLM"], num_hidden_layers=2, hidden_size=24,
                  intermediate_size=48, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, vocab_size=128, rope_theta=5000000, rms_norm_eps=1e-6,
                  tie_word_embeddings=True)
    config.update(updates)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return ModelConfig.from_dir(tmp_path)


def test_snapshot_resolves_main_then_complete_fallback(tmp_path):
    repo = tmp_path / "models--org--model"
    first, second, partial = [repo / "snapshots" / name for name in ("a", "b", "partial")]
    for path in (first, second, partial):
        path.mkdir(parents=True)
    (first / "config.json").write_text("{}")
    (second / "config.json").write_text("{}")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a\n")
    assert snapshot_dir("org/model", cache=tmp_path) == first
    (repo / "refs" / "main").write_text("missing")
    assert snapshot_dir("org/model", cache=tmp_path) in (first, second)
    assert snapshot_dir(str(second)) == second
    with pytest.raises(FileNotFoundError):
        snapshot_dir("absent/model", cache=tmp_path)


def test_snapshot_honors_hf_home(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    path = tmp_path / "hub/models--org--model/snapshots/revision"
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    assert snapshot_dir("org/model") == path


def test_qwen_projection_width_can_differ_from_hidden(tmp_path):
    config = _config(tmp_path)
    assert config.hidden == 24 and config.n_heads * config.head_dim == 32
    assert config.qk_norm and config.tie_embeddings
    assert rope_inv_freq(config)[1].item() == pytest.approx(5000000 ** (-2 / 8))


@pytest.mark.parametrize("updates", [
    {"attention_bias": True}, {"mlp_bias": True}, {"hidden_act": "gelu"},
    {"rope_scaling": {"rope_type": "dynamic", "factor": 8}}, {"sliding_window": 128},
    {"num_key_value_heads": 3}, {"partial_rotary_factor": 0.5},
])
def test_unsupported_model_semantics_fail_before_loading_weights(tmp_path, updates):
    with pytest.raises(ValueError):
        _config(tmp_path, **updates)


def test_llama_config_derives_head_dim(tmp_path):
    config = _config(tmp_path, architectures=["LlamaForCausalLM"], head_dim=None,
                     hidden_size=32, tie_word_embeddings=False)
    assert config.head_dim == 8 and not config.qk_norm and not config.tie_embeddings


def test_llama3_rope_matches_transformers(tmp_path):
    pytest.importorskip("transformers")
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    scaling = dict(rope_type="llama3", factor=8.0, low_freq_factor=1.0,
                   high_freq_factor=4.0, original_max_position_embeddings=8192)
    config = _config(tmp_path, architectures=["LlamaForCausalLM"], head_dim=128,
                     rope_theta=500000, rope_scaling=scaling)
    ref_config = LlamaConfig(hidden_size=512, num_attention_heads=4, head_dim=128,
                             rope_theta=500000, rope_scaling=scaling)
    ref, _ = ROPE_INIT_FUNCTIONS["llama3"](ref_config, torch.device("cpu"))
    torch.testing.assert_close(rope_inv_freq(config), ref, rtol=1e-6, atol=0)
    with pytest.raises(ValueError, match="invalid"):
        rope_inv_freq(replace(config, rope_scaling={**scaling, "high_freq_factor": 1}))


def test_load_weights_follows_index_and_casts(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({"x": torch.arange(6).reshape(2, 3).float()}, str(tmp_path / "a.safetensors"))
    safetensors.save_file({"y": torch.ones(2)}, str(tmp_path / "b.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "x": "a.safetensors", "y": "b.safetensors"}}))
    weights = load_weights(tmp_path, device="cpu", dtype=torch.bfloat16)
    assert set(weights) == {"x", "y"} and weights["x"].dtype == torch.bfloat16
    assert weights["x"].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_missing_shard_and_pickle_only_fail_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="safetensors"):
        load_weights(tmp_path, "cpu")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "missing.safetensors"}}))
    with pytest.raises(FileNotFoundError, match="missing weight shards"):
        load_weights(tmp_path, "cpu")


def test_index_cannot_escape_snapshot(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}))
    with pytest.raises(ValueError, match="local filenames"):
        load_weights(tmp_path, "cpu")


def test_index_missing_tensor_is_rejected(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    safetensors.save_file({"x": torch.ones(2)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "x": "model.safetensors", "y": "model.safetensors"}}))
    with pytest.raises(ValueError, match="missing indexed tensors"):
        load_weights(tmp_path, "cpu")


def test_tokenizer_load_is_local(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    assert load_tokenizer(tmp_path).encode("hello").ids == [1]


@pytest.mark.parametrize("repo,expected", [
    ("Qwen/Qwen3-4B-Instruct-2507", ("qwen3", 36, 2560, 128)),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", ("llama", 32, 4096, 128)),
])
def test_available_local_checkpoint_configs(repo, expected):
    try:
        path = snapshot_dir(repo)
    except FileNotFoundError:
        pytest.skip("optional local checkpoint is absent")
    config = ModelConfig.from_dir(path)
    assert (config.arch, config.n_layers, config.hidden, config.head_dim) == expected
