"""Differential coverage of the CPU native simulator against the original algorithm."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from kernelscope.model.simulate import prepare_simulator, simulate_launch, simulate_launch_reference

BASE = dict(n_sm=4, slots_per_sm=2, sm_order=np.array([0, 2, 1, 3]),
            cost_us_per_key=0.05, t0_us=2.0, t_empty_us=0.1,
            gamma=0.5, bytes_per_key=512, bw_bytes_per_us=100000.0)


@pytest.fixture(scope="module")
def native():
    metadata = prepare_simulator()
    if not metadata["native_available"]:
        pytest.skip("optional local C compiler unavailable: " + str(metadata["error"]))
    return metadata


def _compare(keys, parameters):
    reference = simulate_launch_reference(keys, **parameters)
    actual = simulate_launch(keys, **parameters, backend="native")
    # This tolerance is considerably below fitted model measurement uncertainty.
    # Event counts must remain exact, including simultaneous retirements.
    assert actual.events == reference.events
    assert actual.makespan_us == pytest.approx(reference.makespan_us, rel=1e-12, abs=1e-9)
    assert actual.bw_bound_us == pytest.approx(reference.bw_bound_us, rel=1e-12, abs=1e-9)


def test_randomized_differential_against_reference(native):
    rng = np.random.default_rng(20260923)
    for _ in range(220):
        n_sm = int(rng.choice([1, 2, 3, 7, 16, 31, 64, 127, 128, 129]))
        slots = int(rng.integers(1, 5))
        count = int(rng.integers(0, 4 * n_sm * slots + 1))
        keys = rng.integers(0, 32769, count).astype(float)
        keys[rng.random(count) < 0.4] = 0
        parameters = dict(n_sm=n_sm, slots_per_sm=slots, sm_order=rng.permutation(n_sm),
                          cost_us_per_key=float(10 ** rng.uniform(-4, 0)),
                          t0_us=float(rng.uniform(0, 8)), t_empty_us=float(rng.uniform(0, 1)),
                          gamma=float(rng.uniform(0.15, 1.25)), bytes_per_key=int(rng.choice([0, 256, 512, 1024])),
                          bw_bytes_per_us=float(10 ** rng.uniform(3, 9)))
        _compare(keys, parameters)


@pytest.mark.parametrize("n_sm,slots", [(1, 1), (4, 2), (7, 3), (128, 1), (129, 2)])
def test_thresholds_equal_waves_and_bandwidth_transitions(native, n_sm, slots):
    keys = np.tile([0, 1e-7, 1e-6, np.nextafter(1e-6, np.inf), 128, 128, 1024, 32768], 8)
    for startup, empty in ((0, 0), (1e-10, 1e-9), (1e-8, 0.07)):
        _compare(keys, {**BASE, "n_sm": n_sm, "slots_per_sm": slots,
                        "sm_order": np.arange(n_sm)[::-1], "t0_us": startup, "t_empty_us": empty})


def test_sparse_large_grid_and_strided_keys(native):
    keys = np.zeros(65536)
    keys[::257] = 128
    _compare(keys, {**BASE, "n_sm": 128, "slots_per_sm": 1, "sm_order": np.arange(128)})
    _compare(np.arange(1000, dtype=float)[::3], BASE)


@pytest.mark.parametrize("backend", ["python", "native"])
@pytest.mark.parametrize("change", [
    {"n_sm": 0}, {"slots_per_sm": -1}, {"slots_per_sm": 1.5}, {"max_events": 0}, {"max_events": 2**100},
    {"sm_order": [0, 0, 1, 2]}, {"sm_order": [0, 1, 2, 4]}, {"sm_order": [0., 1., 2., 3.]},
    {"cost_us_per_key": 0}, {"cost_us_per_key": float("nan")}, {"cost_us_per_key": 5e-324},
    {"t0_us": -1}, {"t_empty_us": float("inf")}, {"gamma": 0},
    {"bytes_per_key": -512}, {"bw_bytes_per_us": 0}, {"bw_bytes_per_us": float("inf")},
])
def test_invalid_parameters_fail_before_native_call(backend, change):
    with pytest.raises(ValueError):
        simulate_launch([1, 2], **{**BASE, **change}, backend=backend)


@pytest.mark.parametrize("keys", [[-1], [float("nan")], [float("inf")], [[1, 2]], 3.0])
def test_invalid_keys(keys):
    with pytest.raises(ValueError):
        simulate_launch(keys, **BASE, backend="native")


@pytest.mark.parametrize("backend", ["python", "native"])
def test_event_guard_and_nonfinite_progress(backend, native):
    with pytest.raises(RuntimeError, match="exceeded"):
        simulate_launch(np.arange(1, 100), **BASE, max_events=1, backend=backend)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        with pytest.raises(RuntimeError, match="nonfinite|overflow"):
            simulate_launch([1e308], **{**BASE, "cost_us_per_key": 10}, backend=backend)


def test_import_does_not_build_and_missing_compiler_falls_back(tmp_path):
    cache = tmp_path / "cache"
    code = '''
import json
from pathlib import Path
from kernelscope.model.simulate import prepare_simulator, simulate_launch
import os
assert not Path(os.environ["KERNELSCOPE_SIMULATOR_CACHE"]).exists()
meta = prepare_simulator()
assert meta["backend"] == "python_reference" and meta["error"]
result = simulate_launch([10], n_sm=1, slots_per_sm=1, sm_order=[0], cost_us_per_key=.1,
 t0_us=0, t_empty_us=0, gamma=.5, bytes_per_key=0, bw_bytes_per_us=1)
assert result.makespan_us == 1
try:
 prepare_simulator("native")
except RuntimeError:
 pass
else:
 raise AssertionError("explicit native must report build failure")
print(json.dumps(meta))
'''
    env = {**os.environ, "CC": "definitely-missing-kernelscope-compiler",
           "KERNELSCOPE_SIMULATOR": "auto", "KERNELSCOPE_SIMULATOR_CACHE": str(cache)}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["native_available"] is False


def test_known_policy_rankings_unchanged(native, monkeypatch):
    from kernelscope.model.machine import MachineSpec
    from kernelscope.model.params import ModelParams
    from kernelscope.serve.dispatch import ModelPolicy
    project = Path(__file__).resolve().parents[1]
    machine = MachineSpec.from_json(project / "machines/rtx4090.json")
    params = ModelParams.from_json(project / "models/rtx4090.json")
    shapes = [[513] * 32, [32769] + [513] * 31, [16385] + [513] * 15,
              [16393] + [521] * 15 + [513] * 16]
    for state in ("cold", "warm"):
        for lens in shapes:
            outcomes = []
            for backend in ("python", "native"):
                monkeypatch.setenv("KERNELSCOPE_SIMULATOR", backend)
                policy = ModelPolicy(machine, params, cache_state=state)
                outcomes.append((policy.choose(lens, 32, 8), list(policy.last)))
            assert outcomes[0][0] == outcomes[1][0]
            assert [x[0] for x in outcomes[0][1]] == [x[0] for x in outcomes[1][1]]
            np.testing.assert_allclose([x[1] for x in outcomes[0][1]], [x[1] for x in outcomes[1][1]],
                                       rtol=1e-12, atol=1e-9)
