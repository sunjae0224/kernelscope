from pathlib import Path

import pytest

from kernelscope.backends.accelsim.config import (
    VARIANTS, derive_variant, get_option, set_option, write_variant_config,
)

BASE = (Path(__file__).parent / "fixtures" / "SM80_A100_gpgpusim.config").read_text()


def test_get_option_reads_value_after_flag():
    assert get_option(BASE, "gpgpu_n_clusters") == "108"
    assert get_option(BASE, "gpgpu_cache:dl2") == "S:128:128:16,L:B:m:L:X,A:192:4,32:0,32"
    assert get_option(BASE, "gpgpu_clock_domains") == "1410:1410:1410:1512"


def test_get_option_ignores_commented_lines():
    # "#-gpgpu_clock_domains <Core Clock>:..." precedes the real line
    assert get_option(BASE, "gpgpu_clock_domains") == "1410:1410:1410:1512"


def test_set_option_replaces_only_that_line():
    out = set_option(BASE, "gpgpu_n_clusters", "54")
    assert get_option(out, "gpgpu_n_clusters") == "54"
    assert get_option(out, "gpgpu_n_cores_per_cluster") == "1"
    assert out.count("\n") == BASE.count("\n")


def test_set_option_unknown_flag_raises():
    with pytest.raises(KeyError, match="no_such_option"):
        set_option(BASE, "no_such_option", "1")


@pytest.mark.parametrize("name,flag,expected", [
    ("l2_x2", "gpgpu_cache:dl2", "S:256:128:16,L:B:m:L:X,A:192:4,32:0,32"),
    ("l2_half", "gpgpu_cache:dl2", "S:64:128:16,L:B:m:L:X,A:192:4,32:0,32"),
    ("bw_x2", "gpgpu_clock_domains", "1410:1410:1410:3024"),
    ("bw_half", "gpgpu_clock_domains", "1410:1410:1410:756"),
    ("sm_x2", "gpgpu_n_clusters", "216"),
    ("sm_half", "gpgpu_n_clusters", "54"),
])
def test_named_variants_scale_exactly_one_knob(name, flag, expected):
    out = derive_variant(BASE, name)
    assert get_option(out, flag) == expected
    # everything else untouched
    for other in ("gpgpu_cache:dl2", "gpgpu_clock_domains", "gpgpu_n_clusters"):
        if other != flag:
            assert get_option(out, other) == get_option(BASE, other)


def test_baseline_variant_is_identity():
    assert derive_variant(BASE, "base") == BASE


def test_unknown_variant_lists_known_ones():
    with pytest.raises(KeyError, match="l2_x2"):
        derive_variant(BASE, "l3_x9")


def test_variants_registry_covers_the_plan():
    assert {"base", "l2_x2", "l2_half", "bw_x2", "bw_half", "sm_x2", "sm_half"} <= set(VARIANTS)


def test_write_variant_config_creates_file_and_returns_path(tmp_path):
    base = tmp_path / "gpgpusim.config"
    base.write_text(BASE)
    out = write_variant_config(base, tmp_path / "variants", "bw_x2")
    assert out == tmp_path / "variants" / "bw_x2" / "gpgpusim.config"
    assert get_option(out.read_text(), "gpgpu_clock_domains") == "1410:1410:1410:3024"
