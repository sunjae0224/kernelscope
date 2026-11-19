"""Built-in plugins. Each module is imported lazily-tolerant: a missing optional
dependency (flash_attn, flashinfer, mamba_ssm) drops that module's plugins with a
warning instead of breaking the registry."""
import importlib
import warnings

from kernelscope.plugins.registry import PluginRegistry

REGISTRY = PluginRegistry()

_MODULES = ["sdpa", "flash", "flashinfer", "naive_exec", "triton_tutorial"]

for _m in _MODULES:
    try:
        _mod = importlib.import_module(f"{__name__}.{_m}")
    except ImportError as e:
        warnings.warn(f"builtin plugin module {_m!r} unavailable: {e}")
        continue
    for _cls in _mod.PLUGINS:
        REGISTRY.register(_cls)
