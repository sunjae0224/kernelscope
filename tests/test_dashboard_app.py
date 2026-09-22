import json
from pathlib import Path

import pytest
import pandas as pd

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_app_measured_and_empty_states(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    directory = tmp_path / "hw_4090" / "tiny"
    directory.mkdir(parents=True)
    rows = [dict(status="ok", plugin=p, workload_key="decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal",
                 cache_state="cold", kernel_time_us=t) for p, t in [("fa2", 20.), ("flashdecoding", 30.)]]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert any(metric.value == "30.0 µs" for metric in app.metric)
    assert any("아직 없습니다" in item.value for item in app.info)
    cache = next(radio for radio in app.radio if "Cache" in radio.label)
    cache.set_value("warm")
    app.run()
    assert not app.exception
    assert any("GPU 측정 기록이 없습니다" in item.value for item in app.info)


def test_app_empty_result_root_is_browsable(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert len(app.tabs) == 4
    assert app.metric[0].value == "0"


def test_cpu_serving_never_becomes_gpu_speed_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    scenario = tmp_path / "serve" / "tiny" / "mixed"
    for policy in ("heuristic", "fa2"):
        run = scenario / policy / "repeat_000"
        run.mkdir(parents=True)
        pd.DataFrame({"step": [0, 1], "B": [2, 2], "n_long": [0, 0], "attn_us": [1., 2.],
                      "step_us": [3., 4.], "decode_wall_us": [4., 5.], "policy_us": [.1, .2]}).to_parquet(run / "steps.parquet")
        pd.DataFrame({"rid": [0, 0], "step": [0, 1], "position": [0, 1], "token": [3, 4],
                      "t_us": [0., 100.]}).to_parquet(run / "tokens.parquet")
    (scenario / "manifest.json").write_text(json.dumps({"evidence_kind": "cpu_functional", "performance_claim": False,
                                                         "model": "tiny-random", "repeats": 1, "tokens_equivalent": True}))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("CPU FUNCTIONAL" in item.value for item in app.info)
    assert not any("GPU SERVING" in item.value for item in app.success)
    speed_metric = next(metric for metric in app.metric if metric.label == "관측된 TPOT 가속")
    assert speed_metric.value == "검증 대상 아님"


def test_failed_policy_does_not_invalidate_another_verified_policy(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    scenario = tmp_path / "serve" / "test" / "uniform"
    for policy in ("heuristic", "table", "fixed8"):
        run = scenario / policy / "repeat_000"
        run.mkdir(parents=True)
        (run / "meta.json").write_text(json.dumps({"evidence_kind": "cuda_serving", "performance_claim": True}))
        pd.DataFrame({"step": [0, 1], "B": [1, 1], "n_long": [0, 0], "attn_us": [1., 2.],
                      "step_us": [3., 4.], "decode_wall_us": [4., 5.]}).to_parquet(run / "steps.parquet")
        pd.DataFrame({"rid": [0, 0], "step": [0, 1], "position": [0, 1], "token": [3, 4],
                      "t_us": [0., 100.]}).to_parquet(run / "tokens.parquet")
    (scenario / "manifest.json").write_text(json.dumps({"evidence_kind": "cuda_serving", "performance_claim": False,
                                                         "tokens_equivalent": False, "status": "complete"}))
    pd.DataFrame({"policy": ["table", "fixed8"], "reference": ["heuristic", "heuristic"],
                  "passed": [True, False]}).to_csv(scenario / "equivalence.csv", index=False)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == "관측된 TPOT 가속").value == "1.00×"
    next(s for s in app.selectbox if s.label == "비교할 정책").set_value("fixed8")
    app.run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == "관측된 TPOT 가속").value == "검증 대상 아님"
