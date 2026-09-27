"""gpgpusim.config editing and the named what-if variants.

Each variant scales exactly one knob of the baseline config so a sensitivity is
attributable to one resource:
  l2_*  -gpgpu_cache:dl2  S:<sets>:...   (sets per memory sub-partition)
  bw_*  -gpgpu_clock_domains core:icnt:l2:DRAM   (DRAM clock -> memory bandwidth)
  sm_*  -gpgpu_n_clusters                 (SM count when cores per cluster = 1)
"""
import re
from functools import partial
from pathlib import Path


def _opt_re(flag: str):
    return re.compile(rf"^-{re.escape(flag)}[ \t]+(\S+)", re.M)


def get_option(text: str, flag: str) -> str:
    m = _opt_re(flag).search(text)
    if not m:
        raise KeyError(f"no option -{flag} in config")
    return m.group(1)


def set_option(text: str, flag: str, value: str) -> str:
    rx = _opt_re(flag)
    if not rx.search(text):
        raise KeyError(f"no option -{flag} in config")
    return rx.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + value, text, count=1)


def _scale_l2_sets(v: str, f: float) -> str:
    fields, rest = v.split(",", 1)
    s = fields.split(":")           # S:<sets>:<bsize>:<assoc>
    s[1] = str(int(int(s[1]) * f))
    return ":".join(s) + "," + rest


def _scale_dram_clock(v: str, f: float) -> str:
    d = v.split(":")                # core:icnt:l2:dram
    d[3] = format(float(d[3]) * f, "g")
    return ":".join(d)


def _scale_int(v: str, f: float) -> str:
    return str(int(int(v) * f))


VARIANTS = {
    "base": None,
    "l2_x2": ("gpgpu_cache:dl2", partial(_scale_l2_sets, f=2)),
    "l2_half": ("gpgpu_cache:dl2", partial(_scale_l2_sets, f=0.5)),
    "bw_x2": ("gpgpu_clock_domains", partial(_scale_dram_clock, f=2)),
    "bw_half": ("gpgpu_clock_domains", partial(_scale_dram_clock, f=0.5)),
    "sm_x2": ("gpgpu_n_clusters", partial(_scale_int, f=2)),
    "sm_half": ("gpgpu_n_clusters", partial(_scale_int, f=0.5)),
}


def derive_variant(text: str, name: str) -> str:
    if name not in VARIANTS:
        raise KeyError(f"unknown variant {name!r}; known: {', '.join(VARIANTS)}")
    spec = VARIANTS[name]
    if spec is None:
        return text
    flag, fn = spec
    return set_option(text, flag, fn(get_option(text, flag)))


def write_variant_config(base_path, out_root, name: str) -> Path:
    out = Path(out_root) / name / "gpgpusim.config"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(derive_variant(Path(base_path).read_text(), name))
    return out
