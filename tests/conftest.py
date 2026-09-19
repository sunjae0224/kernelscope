"""Order the whole suite so a single ``pytest -q`` run is not order-dependent on the torch.profiler
capture-loss issue described in kprofile.ProfilerCaptureLost: on this machine, a CUDA
torch.profiler session can permanently stop recording CUDA kernels for the rest of the process
after a pause following an earlier profiler session (root cause unknown). Running every gpu-marked
test first avoids that pause; tests/test_cuda_ext.py's first-time CUDA-extension build (~35 s) is
the kind of pause that triggers it, so its gpu tests run last among the gpu tests, ahead of the
non-gpu tests.
"""


def pytest_collection_modifyitems(config, items):
    def bucket(item):
        is_gpu = item.get_closest_marker("gpu") is not None
        if not is_gpu:
            return 2
        return 1 if "test_cuda_ext.py" in str(item.fspath) else 0

    items.sort(key=bucket)
