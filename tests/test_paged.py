import torch

from kernelscope.plugins.paged import PAGE, build_block_table, gather_from_pool


def test_block_table_gives_every_sequence_distinct_pages():
    lens = [1000, 256, 1, 513]
    table, n = build_block_table(lens)
    need = [4, 1, 1, 3]
    assert n == sum(need)
    assert table.dtype == torch.int32 and table.shape == (4, 4)
    used = [int(table[b, i]) for b, k in enumerate(need) for i in range(k)]
    assert sorted(used) == list(range(n))            # a permutation of the pool


def test_block_table_is_seeded():
    assert torch.equal(build_block_table([700, 300])[0], build_block_table([700, 300])[0])
    assert not torch.equal(build_block_table([700, 300], seed=0)[0], build_block_table([700, 300], seed=1)[0])


def test_gather_reads_token_t_of_sequence_b_from_its_page():
    lens = [600, 5]
    table, n = build_block_table(lens)
    pool = torch.arange(n * PAGE * 2 * 3, dtype=torch.float32).view(n, PAGE, 2, 3)
    dense = gather_from_pool(pool, lens, table, L=600)
    assert dense.shape == (2, 600, 2, 3)
    for b, t in [(0, 0), (0, 255), (0, 256), (0, 599), (1, 4)]:
        assert torch.equal(dense[b, t], pool[int(table[b, t // PAGE]), t % PAGE])
    assert torch.count_nonzero(dense[1, 5:]) == 0     # past the sequence length
