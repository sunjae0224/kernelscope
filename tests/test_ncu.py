from kernelscope.backends.realhw.ncu import DEFAULT_METRICS, build_ncu_command, parse_ncu_csv

NCU_CSV = '''"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"
"0","4242","python","127.0.0.1","void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t>, false, false, false, false, false, true, true>(Flash_fwd_params)","1","7","(128, 1, 1)","(1, 32, 8)","0","8.0","Command line profiler metrics","gpu__time_duration.sum","usecond","45.12"
"0","4242","python","127.0.0.1","void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t>, false, false, false, false, false, true, true>(Flash_fwd_params)","1","7","(128, 1, 1)","(1, 32, 8)","0","8.0","Command line profiler metrics","launch__grid_size","","256"
"0","4242","python","127.0.0.1","void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t>, false, false, false, false, false, true, true>(Flash_fwd_params)","1","7","(128, 1, 1)","(1, 32, 8)","0","8.0","Command line profiler metrics","smsp__inst_executed.sum","inst","1,234,567"
"1","4242","python","127.0.0.1","void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<128, 64, 128, 4, false, false, cutlass::half_t>, 3, 3>(Flash_fwd_params)","1","7","(128, 1, 1)","(32, 1, 1)","0","8.0","Command line profiler metrics","gpu__time_duration.sum","usecond","3.5"
'''


def test_parse_returns_one_row_per_launch_and_metric():
    rows = parse_ncu_csv(NCU_CSV)
    assert len(rows) == 4
    first = rows[0]
    assert first["launch_idx"] == 0
    assert first["kernel_name"].startswith("void flash_fwd_splitkv_kernel")
    assert first["metric"] == "gpu__time_duration.sum"
    assert first["unit"] == "usecond"
    assert first["value"] == 45.12


def test_parse_strips_thousands_separators():
    rows = parse_ncu_csv(NCU_CSV)
    inst = next(r for r in rows if r["metric"] == "smsp__inst_executed.sum")
    assert inst["value"] == 1234567.0


def test_parse_keeps_launch_index_per_kernel_launch():
    rows = parse_ncu_csv(NCU_CSV)
    assert {r["launch_idx"] for r in rows} == {0, 1}
    combine = [r for r in rows if r["launch_idx"] == 1]
    assert combine[0]["kernel_name"].startswith("void flash_fwd_splitkv_combine_kernel")
    assert combine[0]["grid_size"] == "(32, 1, 1)"


def test_parse_ignores_prof_banner_lines():
    text = "==PROF== Connected to process 4242\n" + NCU_CSV + "==PROF== Disconnected from process 4242\n"
    assert len(parse_ncu_csv(text)) == 4


def test_parse_empty_output_gives_no_rows():
    assert parse_ncu_csv("") == []


def test_build_command_targets_only_the_requested_launches():
    cmd = build_ncu_command(
        target=["/env/bin/python", "run.py", "--kernel", "fa2"],
        kernel_regex=r"flash_fwd",
        num_launches=2,
        launch_skip=3,
        metrics=["gpu__time_duration.sum", "launch__grid_size"],
        ncu_exe="/usr/local/cuda/bin/ncu",
    )
    assert cmd[0] == "/usr/local/cuda/bin/ncu"
    assert cmd[-4:] == ["/env/bin/python", "run.py", "--kernel", "fa2"]
    opts = cmd[1:-4]
    assert "--csv" in opts
    assert opts[opts.index("-k") + 1] == "regex:flash_fwd"
    assert opts[opts.index("--launch-skip") + 1] == "3"
    assert opts[opts.index("--launch-count") + 1] == "2"
    assert opts[opts.index("--metrics") + 1] == "gpu__time_duration.sum,launch__grid_size"


def test_build_command_uses_default_metric_set():
    cmd = build_ncu_command(target=["python", "x.py"], kernel_regex="k", num_launches=1)
    assert cmd[cmd.index("--metrics") + 1] == ",".join(DEFAULT_METRICS)
    assert "gpu__time_duration.sum" in DEFAULT_METRICS
    assert "smsp__inst_executed.sum" in DEFAULT_METRICS
