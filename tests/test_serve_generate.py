from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kernelscope.serve.dispatch import FixedPolicy
from kernelscope.serve.generate import generate
from tests.test_serve_engine import ArithmeticModel, pool


class TinyTokenizer:
    def encode(self, text):
        return SimpleNamespace(ids=[1, 2])

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(map(str, tokens))


def test_raw_generation_decodes_and_releases_at_eos():
    cache = pool()
    result = generate(ArithmeticModel(), cache, TinyTokenizer(), "A prompt", 8, FixedPolicy(0), eos_token_ids=[5])
    assert result["token_ids"] == [3, 4, 5]
    assert result["generated_text"] == "3 4 5" and result["stopped_on_eos"]
    assert not result["performance_claim"] and cache.free_pages == 12


@pytest.mark.parametrize("prompt,count", [("  ", 8), ("prompt", 0), ("prompt", -1)])
def test_invalid_generation_arguments(prompt, count):
    with pytest.raises(ValueError):
        generate(ArithmeticModel(), pool(), TinyTokenizer(), prompt, count, FixedPolicy(0))
