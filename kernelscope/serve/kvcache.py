"""A bounded paged KV cache with shared ownership across every decoder layer."""

from __future__ import annotations

import operator

import torch

PAGE = 256  # flash-attn paged cache requires multiples of 256 tokens.
_MAX_LENGTH = 2**31 - 1


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class PagePool:
    def __init__(self, n_layers, n_pages, n_kv_heads, head_dim, dtype=torch.bfloat16, device="cuda"):
        self.n_layers = _integer(n_layers, "n_layers", 1)
        self.n_pages = _integer(n_pages, "n_pages", 1)
        self.n_kv_heads = _integer(n_kv_heads, "n_kv_heads", 1)
        self.head_dim = _integer(head_dim, "head_dim", 1)
        if not dtype.is_floating_point:
            raise ValueError("KV dtype must be floating point")
        self.dtype = dtype
        shape = (self.n_pages, PAGE, self.n_kv_heads, self.head_dim)
        # Contents outside committed lengths are never read; avoid zeroing GiB of storage.
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(self.n_layers)]
        self.v = [torch.empty_like(t) for t in self.k]
        self.device = self.k[0].device
        self._free = list(range(self.n_pages - 1, -1, -1))
        self._pages: dict = {}
        self._len: dict = {}

    @staticmethod
    def bytes_per_page(n_layers, n_kv_heads, head_dim, dtype=torch.bfloat16) -> int:
        return (_integer(n_layers, "n_layers", 1) * 2 * PAGE
                * _integer(n_kv_heads, "n_kv_heads", 1) * _integer(head_dim, "head_dim", 1)
                * torch.empty((), dtype=dtype).element_size())

    @classmethod
    def for_budget(cls, cfg, budget_bytes, device="cuda", dtype=torch.bfloat16):
        budget = _integer(budget_bytes, "budget_bytes", 1)
        per_page = cls.bytes_per_page(cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, dtype)
        if budget < per_page:
            raise ValueError(f"KV budget {budget} bytes cannot hold one page ({per_page} bytes)")
        return cls(cfg.n_layers, budget // per_page, cfg.n_kv_heads, cfg.head_dim, dtype, device)

    @property
    def free_pages(self) -> int:
        return len(self._free)

    def capacity(self, seq_id) -> int:
        return len(self._pages[seq_id]) * PAGE

    def reserve(self, seq_id, total_len: int) -> None:
        self.reserve_many({seq_id: total_len})

    def reserve_many(self, lengths: dict) -> None:
        """Reserve an entire batch atomically, including on allocation exhaustion."""
        needs = {}
        for seq_id, total_len in lengths.items():
            total_len = _integer(total_len, "total_len")
            if total_len > _MAX_LENGTH:
                raise ValueError("sequence length exceeds flash-attn int32 capacity")
            needs[seq_id] = max(0, (total_len + PAGE - 1) // PAGE - len(self._pages.get(seq_id, ())))
        count = sum(needs.values())
        if count > len(self._free):
            raise MemoryError(f"KV pool exhausted: need {count} pages, {len(self._free)} free")
        for seq_id, need in needs.items():
            pages = self._pages.setdefault(seq_id, [])
            self._len.setdefault(seq_id, 0)
            pages.extend(self._free.pop() for _ in range(need))

    def release(self, seq_id) -> None:
        self._free.extend(reversed(self._pages.pop(seq_id, [])))
        self._len.pop(seq_id, None)

    def length(self, seq_id) -> int:
        return self._len[seq_id]

    def set_length(self, seq_id, n: int) -> None:
        n = _integer(n, "length")
        if n > self.capacity(seq_id) or n > _MAX_LENGTH:
            raise ValueError("length exceeds reserved KV capacity")
        self._len[seq_id] = n

    def block_table(self, seq_ids) -> torch.Tensor:
        seq_ids = list(seq_ids)
        width = max((len(self._pages[s]) for s in seq_ids), default=0)
        rows = [self._pages[s] + [0] * (width - len(self._pages[s])) for s in seq_ids]
        return torch.tensor(rows, dtype=torch.int32, device=self.device).reshape(len(seq_ids), width)

    def lengths(self, seq_ids) -> torch.Tensor:
        return torch.tensor([self._len[s] for s in seq_ids], dtype=torch.int32, device=self.device)
