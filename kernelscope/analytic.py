"""Analytic traffic / FLOP model for one attention call — the Nsight-free source of
"achieved bandwidth" and "achieved TFLOPS".

Traffic is the *compulsory* traffic (each operand touched once). For decode this
is exactly what the flash kernels move; for prefill the K/V re-reads across query
blocks mostly hit L2 at these sizes, so DRAM traffic is close to compulsory too.
Cross-check against the simulator's DRAM statistics where it matters.
"""
from kernelscope.workload import Workload

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4, "float8_e4m3fn": 1, "float8_e5m2": 1}


def attended_pairs(w: Workload) -> int:
    """Number of (query, key) pairs that contribute to the output."""
    if not w.causal:
        return w.L_q * w.L_kv
    # bottom-right aligned causal: query i sees L_kv - L_q + i + 1 keys
    return w.L_q * (w.L_kv - w.L_q) + w.L_q * (w.L_q + 1) // 2


def total_attended_pairs(w: Workload) -> int:
    """(query, key) pairs over the whole batch. A ragged batch is decode with L_q=1, so each
    sequence's single query attends to its whole live cache."""
    if w.is_ragged:
        return sum(w.lens())
    return w.B * attended_pairs(w)


def attention_flops(w: Workload) -> int:
    """QK^T and PV: 2 MACs per (pair, head, dim) = 4 FLOPs."""
    return 4 * w.H_q * w.d * total_attended_pairs(w)


def attention_traffic(w: Workload, kv_heads_read: int | None = None) -> dict:
    s = DTYPE_BYTES[w.dtype]
    h_kv = kv_heads_read or w.H_kv
    q = w.B * w.L_q * w.H_q * w.d * s
    kv = 2 * sum(w.lens()) * h_kv * w.d * s
    return {"dtype_bytes": s, "q_bytes": q, "kv_bytes": kv, "o_bytes": q, "total_bytes": q + kv + q}


def arithmetic_intensity(w: Workload, kv_heads_read: int | None = None) -> float:
    return attention_flops(w) / attention_traffic(w, kv_heads_read)["total_bytes"]
