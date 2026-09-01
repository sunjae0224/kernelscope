"""Does a plugin compute the same attention as the reference? Reported, never raised."""
import math

from kernelscope.reference import reference_attention


def check_plugin(plugin, w, atol: float = 1e-2) -> dict:
    try:
        inputs = plugin.build_inputs(w)
        out = plugin.to_dense_output(plugin.run(inputs))
        q, k, v = plugin.to_dense_inputs(inputs)
        ref = reference_attention(q, k, v, w.causal)
        if tuple(out.shape) != tuple(ref.shape):
            return _result(False, math.nan,
                           f"shape mismatch: got {tuple(out.shape)}, expected {tuple(ref.shape)}")
        diff = (out.float() - ref.float()).abs().max().item()
        return _result(diff <= atol, diff, None)
    except Exception as e:  # a broken plugin must not take the sweep down
        return _result(False, math.nan, f"{type(e).__name__}: {e}")


def _result(ok, diff, error):
    return {"ok": ok, "max_abs_diff": diff, "error": error}
