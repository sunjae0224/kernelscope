"""Which SM runs which block: record %smid for a grid that exactly fills every SM."""
from collections import Counter


def smem_for_ctas_per_sm(k: int, smem_per_sm: int, reserved: int) -> int:
    """Dynamic shared memory per CTA such that exactly k CTAs fit on one SM (256-byte aligned)."""
    s = (smem_per_sm // k - reserved) // 256 * 256
    while smem_per_sm // (s + reserved) > k:
        s += 256
    return s


def block_placement(ctas_per_sm: int = 2, device: str = "cuda", threads: int = 128, spin_cycles: int = 2_000_000) -> dict:
    """Launch n_sm * ctas_per_sm blocks that each record %smid and then spin long enough to be co-resident."""
    from kernelscope.backends.realhw.kprofile import props_from_torch
    from kernelscope.bench.cuda_ext import load_ext
    p = props_from_torch(device)
    n = p["num_sms"] * ctas_per_sm
    smem = smem_for_ctas_per_sm(ctas_per_sm, p["smem_per_sm"], p["reserved_smem_per_block"])
    smid = load_ext().smid_probe(n, threads, smem, spin_cycles).cpu().tolist()
    counts = Counter(smid)
    pairs = [smid[i] == smid[i + 1] for i in range(0, n - 1, 2)]
    return {"ctas_per_sm": ctas_per_sm, "n_ctas": n, "distinct_sms": len(counts),
            "max_ctas_on_one_sm": max(counts.values()),
            "adjacent_pair_same_sm": sum(pairs) / len(pairs),
            "first_wave_distinct": len(set(smid[: p["num_sms"]])), "smid": smid}
