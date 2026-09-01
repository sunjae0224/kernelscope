"""Measurement hygiene: refuse to silently measure on a GPU someone else is using."""
from kernelscope.backends.realhw.hygiene import gpu_contention, parse_apps, parse_gpu_query, visible_device_index

GPU_QUERY = """\
index, uuid, utilization.gpu [%], memory.used [MiB], power.draw [W]
0, GPU-aaaa, 100 %, 55837 MiB, 224.87 W
1, GPU-bbbb, 100 %, 55753 MiB, 302.58 W
2, GPU-cccc, 3 %, 1831 MiB, 123.12 W
3, GPU-dddd, 0 %, 0 MiB, 60.36 W
"""
APPS = """\
gpu_uuid, pid, process_name, used_gpu_memory [MiB]
GPU-aaaa, 625933, /envs/x/bin/python, 55702 MiB
GPU-bbbb, 436356, /envs/x/bin/python, 55742 MiB
GPU-cccc, 996307, /envs/gradkernel/bin/python, 1692 MiB
"""


def test_parse_gpu_query_gives_one_record_per_gpu():
    g = parse_gpu_query(GPU_QUERY)
    assert [r["index"] for r in g] == [0, 1, 2, 3]
    assert g[1]["uuid"] == "GPU-bbbb" and g[1]["utilization_pct"] == 100 and g[1]["memory_used_mib"] == 55753
    assert g[3]["utilization_pct"] == 0


def test_parse_apps_groups_pids_by_gpu_uuid():
    a = parse_apps(APPS)
    assert a["GPU-aaaa"] == [625933]
    assert a["GPU-cccc"] == [996307]
    assert a.get("GPU-dddd", []) == []


def test_visible_device_index_maps_through_cuda_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert visible_device_index("cuda") == 3
    assert visible_device_index("cuda:1") == 1
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert visible_device_index("cuda") == 0
    assert visible_device_index("cuda:2") == 2
    assert visible_device_index("cpu") is None


def test_gpu_contention_reports_other_processes_and_utilisation():
    c = gpu_contention(1, gpu_query_text=GPU_QUERY, apps_text=APPS, my_pids={999})
    assert c["busy"] is True
    assert c["utilization_pct"] == 100
    assert c["other_pids"] == [436356]
    c3 = gpu_contention(3, gpu_query_text=GPU_QUERY, apps_text=APPS, my_pids=set())
    assert c3["busy"] is False and c3["other_pids"] == []


def test_gpu_contention_ignores_our_own_processes():
    c = gpu_contention(2, gpu_query_text=GPU_QUERY, apps_text=APPS, my_pids={996307})
    assert c["other_pids"] == []
    assert c["busy"] is False          # 3 % utilisation from ourselves is fine


def test_gpu_contention_when_nvidia_smi_is_unavailable():
    c = gpu_contention(0, gpu_query_text=None, apps_text=None, my_pids=set())
    assert c["busy"] is None and "unavailable" in c["note"]
