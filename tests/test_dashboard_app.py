import json
from pathlib import Path

import pytest
import pandas as pd

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest
from demo_fixtures import make_run

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
LAB = "lab_page.py"
WORKLOAD = "decode_B1_Lq1_Lkv1024_Hq32_Hkv8_d128_float16_causal"


def _hw_rows(tmp_path):
    directory = tmp_path / "hw_4090" / "tiny"
    directory.mkdir(parents=True)
    rows = [dict(status="ok", plugin=p, workload_key=WORKLOAD, cache_state="cold", kernel_time_us=t)
            for p, t in [("fa2", 20.), ("flashdecoding", 30.)]]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def _lab(app):
    app.switch_page(LAB)
    return app.run()


def test_entrypoint_opens_demo_then_lab(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    _hw_rows(tmp_path)
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "같은 답, 더 빠른 토큰."
    assert not app.tabs                       # the Demo page is one flow, no tabs
    app = _lab(app)
    assert not app.exception
    assert len(app.tabs) == 5
    assert any(metric.value == "30.0 µs" for metric in app.metric)
    assert any("아직 없습니다" in item.value for item in app.info)
    cache = next(radio for radio in app.radio if "Cache" in radio.label)
    cache.set_value("warm")
    app.run()
    assert not app.exception
    assert any("GPU 측정 기록이 없습니다" in item.value for item in app.info)


def test_lab_empty_result_root_is_browsable(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    app = _lab(AppTest.from_file(str(APP), default_timeout=60).run())
    assert not app.exception
    assert len(app.tabs) == 5
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
    app = _lab(AppTest.from_file(str(APP), default_timeout=60).run())
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
    app = _lab(AppTest.from_file(str(APP), default_timeout=60).run())
    assert not app.exception
    assert next(m for m in app.metric if m.label == "관측된 TPOT 가속").value == "1.00×"
    next(s for s in app.selectbox if s.label == "비교할 정책").set_value("fixed8")
    app.run()
    assert not app.exception
    assert next(m for m in app.metric if m.label == "관측된 TPOT 가속").value == "검증 대상 아님"


def test_demo_page_renders_hero_and_race(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "같은 답, 더 빠른 토큰."
    assert any("2.00×" in metric.value for metric in app.metric)
    assert any(metric.value == "일치" for metric in app.metric)
    assert any("생성 토큰 일치" in item.value for item in app.success)
    policy = next(radio for radio in app.radio if radio.label == "KernelScope 정책")
    policy.set_value("hybrid")
    app.run()
    assert not app.exception


def test_demo_page_without_pair_is_informative(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic",))
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert any("비교할 정책 쌍이 없습니다" in item.value for item in app.info)


def test_demo_page_empty_root(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    app = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not app.exception
    assert any("서빙 기록이 없습니다" in item.value for item in app.info)


def test_demo_page_family_radio_switches_run(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, B=3)
    make_run(tmp_path, scenario="uniform", B=5)
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception
    assert any("3개 요청" in c.value for c in app.caption)
    family = next(radio for radio in app.radio if radio.label == "배치 종류")
    family.set_value("uniform")
    app.run()
    assert not app.exception
    assert any("5개 요청" in c.value for c in app.caption)
    assert not any("3개 요청" in c.value for c in app.caption)


def test_demo_page_scenes_two_to_four(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELSCOPE_RESULTS", str(tmp_path))
    make_run(tmp_path, policies=("heuristic", "table"))
    make_run(tmp_path, scenario="uniform", policies=("heuristic", "table"))
    directory = tmp_path / "hw_4090" / "worst"
    directory.mkdir(parents=True)
    worst = "decode_B32_Lq1_Lkv32768+512x31_Hq32_Hkv8_d128_float16_causal"
    rows = [dict(status="ok", plugin=p, workload_key=worst, cache_state="cold", kernel_time_us=t)
            for p, t in [("flashdecoding_paged", 1071.8), ("fd_s16_paged", 279.4), ("fa2_paged", 1071.2), ("flashinfer_paged_cudacore", 269.9)]]
    (directory / "summaries.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    app = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not app.exception
    headers = [s.value for s in app.subheader]
    assert any(h.startswith("2 ·") for h in headers) and any(h.startswith("3 ·") for h in headers) and any(h.startswith("4 ·") for h in headers)
    assert any("가장 긴 CTA" in metric.label for metric in app.metric)
    assert any("3.84×" in item.value for item in app.caption)            # 1071.8 / 279.4 from the fixture rows
    assert any("대조군 · 균일 배치의 배율" in item.value for item in app.caption)
    assert len(app.dataframe) >= 1
