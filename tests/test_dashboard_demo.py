import json
from pathlib import Path

import pandas as pd
import pytest

from demo_fixtures import make_run
from kernelscope.dashboard import demo


def test_race_payload_structure_and_clock(tmp_path):
    run = make_run(tmp_path)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert [p["name"] for p in payload["policies"]] == ["heuristic", "table"]
    heuristic, table = payload["policies"]
    assert payload["scenario"]["B"] == 3 and payload["scenario"]["lens"] == [4096, 128, 128]
    assert payload["scenario"]["max_new_tokens"] == 4 and payload["scenario"]["natural_text"] is False
    # clock zero = the first wave's last first-token time: prefill tokens sit at 0, the first decode step at +1000 ms
    assert heuristic["requests"][0]["tokens"][0]["t_ms"] == 0.0
    assert heuristic["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(1000.0)
    assert table["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(500.0)
    assert heuristic["end_ms"] == pytest.approx(3000.0) and table["end_ms"] == pytest.approx(1500.0)
    assert [r["featured"] for r in heuristic["requests"]] == [True, True, False]
    assert heuristic["steps"][0] == {"step": 0, "t_ms": pytest.approx(1000.0), "num_splits": 0, "attn_ms": pytest.approx(0.8)}
    assert table["steps"][0]["num_splits"] == 16
    assert heuristic["tpot_ms"] == pytest.approx(1.0) and heuristic["color"].startswith("#")
    assert payload["reference"] == "heuristic"
    assert payload["agreement"]["table"] == {"identical": True, "requests_equal": 3, "requests_total": 3,
                                             "first_divergence": [], "summary_tokens_equivalent": True}
    json.dumps(payload)  # serializable


def test_race_payload_picks_median_repeat(tmp_path):
    run = make_run(tmp_path, repeats=3)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert [p["repeat_index"] for p in payload["policies"]] == [1, 1]   # end times grow with the repeat; the middle one


def test_race_payload_even_repeats_take_the_smaller_median(tmp_path):
    run = make_run(tmp_path, repeats=2)
    assert demo.race_payload(run, ["heuristic", "table"])["policies"][0]["repeat_index"] == 0


def test_race_payload_detects_divergence(tmp_path):
    run = make_run(tmp_path, diverge=("table", 1, 2))
    agreement = demo.race_payload(run, ["heuristic", "table"])["agreement"]["table"]
    assert agreement["identical"] is False and agreement["first_divergence"] == [{"rid": 1, "position": 2}]
    assert agreement["requests_equal"] == 2 and agreement["summary_tokens_equivalent"] is False


def test_race_payload_decodes_incrementally(tmp_path):
    run = make_run(tmp_path, natural=True)
    payload = demo.race_payload(run, ["heuristic", "table"], decode=lambda ids: " ".join(f"w{t}" for t in ids))
    assert [t["text"] for t in payload["policies"][0]["requests"][0]["tokens"]] == ["w100", " w200", " w210", " w220"]
    assert payload["scenario"]["natural_text"] is True


def test_race_payload_without_decode_has_no_text(tmp_path):
    run = make_run(tmp_path, natural=True)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert payload["scenario"]["natural_text"] is False
    assert all(t["text"] == "" for t in payload["policies"][1]["requests"][2]["tokens"])


def test_race_payload_requires_pair(tmp_path):
    run = make_run(tmp_path, policies=("heuristic",))
    with pytest.raises(ValueError):
        demo.race_payload(run, ["heuristic"])
    with pytest.raises(FileNotFoundError):
        demo.race_payload(run, ["heuristic", "table"])
    with pytest.raises(FileNotFoundError):
        demo.race_payload(make_run(tmp_path, campaign="noref", policies=("table",)), ["table"])


def test_race_payload_arrivals_clock_is_nonnegative(tmp_path):
    run = make_run(tmp_path, arrivals=True, B=3)
    payload = demo.race_payload(run, ["heuristic", "table"])
    times = [t["t_ms"] for p in payload["policies"] for r in p["requests"] for t in r["tokens"]]
    assert min(times) == 0.0 and payload["t0_rule"] == "first_wave_admitted"
    assert payload["policies"][0]["requests"][0]["tokens"][1]["t_ms"] == pytest.approx(1000.0)


def test_race_payload_flat_policy_dir(tmp_path):
    run = make_run(tmp_path, flat=True)
    payload = demo.race_payload(run, ["heuristic", "table"])
    assert payload["policies"][0]["repeat_index"] == 0 and payload["policies"][0]["end_ms"] == pytest.approx(3000.0)
    assert demo.policy_dirs(run) == ["heuristic", "table"]


def test_save_and_load_race(tmp_path):
    run = make_run(tmp_path)
    payload = demo.race_payload(run, ["heuristic", "table"])
    path = demo.save_race(run, payload)
    assert path == run / "race.json" and demo.load_race(run) == json.loads(json.dumps(payload))
    assert demo.load_race(tmp_path) is None
    path.write_text("{not json")
    assert demo.load_race(run) is None


def test_featured_runs_prefers_natural_then_newest(tmp_path):
    make_run(tmp_path, campaign="old", scenario="ragged", created="2026-09-01T00:00:00+00:00")
    make_run(tmp_path, campaign="new", scenario="ragged", created="2026-10-01T00:00:00+00:00")
    make_run(tmp_path, campaign="text", scenario="heldout_ragged", natural=True, created="2026-09-15T00:00:00+00:00",
             scenario_family="ragged")
    make_run(tmp_path, campaign="new", scenario="control_uniform_fixed8", policies=("heuristic", "fixed8"),
             created="2026-10-01T00:00:00+00:00")
    runs = demo.featured_runs(tmp_path)
    assert runs["ragged"].parent.name == "text" and runs["uniform"].name == "control_uniform_fixed8"
    assert "arrivals" not in runs and list(runs) == ["ragged", "uniform"]


def test_featured_runs_empty_root(tmp_path):
    (tmp_path / "serve_4090").mkdir()
    assert demo.featured_runs(tmp_path) == {}


def test_headline_and_missing_policy(tmp_path):
    run = make_run(tmp_path)
    h = demo.headline(run, "table")
    assert h["heuristic_tpot_ms"] == pytest.approx(1.0) and h["policy_tpot_ms"] == pytest.approx(0.5)
    assert h["speedup"] == pytest.approx(2.0) and h["tokens_equivalent"] is True and h["repeats"] == 2
    assert h["model"] == "test/model" and h["scenario_name"] == "ragged" and h["policy"] == "table"
    missing = demo.headline(run, "model")
    assert missing["policy_tpot_ms"] is None and missing["speedup"] is None and missing["tokens_equivalent"] is None
    assert missing["heuristic_tpot_ms"] == pytest.approx(1.0)


def test_campaign_table(tmp_path):
    run = make_run(tmp_path)
    make_run(tmp_path, scenario="uniform")
    table = demo.campaign_table(run.parent)
    assert set(table.scenario) == {"ragged", "uniform"} and len(table) == 4
    assert set(table.family) == {"ragged", "uniform"}
    assert table[table.policy == "table"].speedup.tolist() == [2.0, 2.0]
    assert demo.campaign_table(tmp_path / "nowhere").empty

import yaml

from kernelscope.serve import scenarios
from kernelscope.workload import Workload

ROOT = Path(__file__).resolve().parents[1]
TABLE_CSV = ROOT / "demo_data" / "dispatch_paged_cold.csv"
MACHINE, PARAMS = ROOT / "machines" / "rtx4090.json", ROOT / "models" / "rtx4090.json"
MEASURED = pd.Series({"flashdecoding_paged": 1071.8, "fa2_paged": 1071.2, "fd_s8_paged": 283.5, "fd_s16_paged": 279.4})


def test_cta_work_heuristic_vs_split():
    base = demo.cta_work(demo.WORST_KEY, "flashdecoding_paged", 128)
    split = demo.cta_work(demo.WORST_KEY, "fd_s16_paged", 128)
    assert base["splits"] == 1 and base["ctas"] == 32 * 8 and base["ctas_per_sm"] == 2.0
    assert split["splits"] == 16 and split["ctas"] == 32 * 8 * 16
    assert max(base["keys"]) == 32768 and max(split["keys"]) == 32768 // 16
    assert base["longest_over_mean"] > 10 > split["longest_over_mean"] > 1
    assert all(isinstance(k, int) for k in split["keys"])


def test_decision_card_rows():
    lens = list(Workload.from_key(demo.WORST_KEY).lens())
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, PARAMS, MEASURED)
    assert [r["policy"] for r in rows] == ["heuristic", "table", "hybrid", "model"]
    heuristic, table, hybrid, model = rows
    assert heuristic["num_splits"] == 1 and heuristic["kernel"] == "flashdecoding_paged"
    assert heuristic["kernel_us"] == pytest.approx(1071.8) and heuristic["speedup_vs_heuristic"] == pytest.approx(1.0)
    assert "0.8" in heuristic["note"]
    assert table["num_splits"] == 16 and table["kernel"] == "fd_s16_paged"
    assert table["speedup_vs_heuristic"] == pytest.approx(1071.8 / 279.4) and "거리" in table["note"]
    for row in (hybrid, model):
        assert row["note"]                       # source / ranking text, or the reason it could not run
        assert row["num_splits"] is None or isinstance(row["num_splits"], int)
    assert hybrid["num_splits"] == 16 and hybrid["note"].startswith("출처 table")
    assert model["num_splits"] == 8 and model["note"].startswith("예측 순위")


def test_decision_card_without_params(tmp_path):
    lens = list(Workload.from_key(demo.WORST_KEY).lens())
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, tmp_path / "missing.json", MEASURED)
    assert rows[0]["num_splits"] == 1 and rows[1]["num_splits"] == 16
    assert rows[2]["num_splits"] is None and rows[2]["note"].startswith("선택 불가")
    assert rows[3]["num_splits"] is None and rows[3]["note"].startswith("선택 불가")
    rows = demo.decision_card(lens, 32, 8, TABLE_CSV, MACHINE, PARAMS, None)
    assert rows[1]["kernel_us"] is None and rows[1]["speedup_vs_heuristic"] is None


def test_compose_scenario_roundtrip(tmp_path):
    corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
    assert len(corpus) >= 3 and dataset["id"]
    spec = demo.compose_scenario(16384, 7, 384, 32, "What is measured?\nAnswer:", corpus, dataset)
    path = demo.write_scenario(tmp_path / "live" / "scenario.yaml", spec)
    requests, loaded_dataset = scenarios.load_scenario(path)
    assert len(requests) == 8 and requests[0].prompt_len == 16384 and requests[7].prompt_len == 384
    assert requests[0].max_new_tokens == 32 and requests[0].prompt_suffix == "What is measured?\nAnswer:"
    assert requests[0].prompt_recipe == "corpus_repeat" and loaded_dataset["scenario_family"] == "ragged"
    assert yaml.safe_load(path.read_text())["requests"][1]["count"] == 7


@pytest.mark.parametrize("bad", [dict(long_len=0), dict(n_short=0), dict(short_len=-1), dict(max_new_tokens=0), dict(question="  ")])
def test_compose_scenario_rejects(bad):
    corpus, dataset = demo.load_demo_corpus(ROOT / "scenarios" / "demo_text_ragged.yaml")
    args = dict(long_len=8192, n_short=3, short_len=256, max_new_tokens=8, question="Q:", corpus=corpus, dataset=dataset)
    args.update(bad)
    with pytest.raises(ValueError):
        demo.compose_scenario(**args)


def test_verify_tail():
    assert demo.verify_tail("header\n  row PASS\n{\"PASS\": 111}\n") == {"PASS": 111}
    assert demo.verify_tail("{\"PASS\": 3, \"FAIL\": 1}") == {"PASS": 3, "FAIL": 1}
    assert demo.verify_tail("no json here") == {}


def test_featured_runs_keeps_the_ragged_models_family(tmp_path):
    make_run(tmp_path, campaign="qwen", scenario="ragged", natural=True, created="2026-10-06T00:00:00+00:00", model="qwen")
    make_run(tmp_path, campaign="qwen", scenario="uniform", natural=True, created="2026-09-01T00:00:00+00:00", model="qwen")
    make_run(tmp_path, campaign="llama", scenario="uniform", natural=True, created="2026-10-01T00:00:00+00:00", model="llama")
    runs = demo.featured_runs(tmp_path)
    assert runs["ragged"].parent.name == "qwen" and runs["uniform"].parent.name == "qwen"   # same model beats newer


def test_featured_campaign_prefers_family_coverage(tmp_path):
    make_run(tmp_path, campaign="wide", scenario="ragged", created="2026-09-01T00:00:00+00:00")
    make_run(tmp_path, campaign="wide", scenario="uniform", created="2026-09-01T00:00:00+00:00")
    make_run(tmp_path, campaign="narrow", scenario="ragged", created="2026-10-06T00:00:00+00:00")
    assert demo.featured_campaign(tmp_path).name == "wide"
    assert demo.featured_campaign(tmp_path / "empty") is None


def test_render_race_html_embeds_two_policies(tmp_path):
    run = make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    payload = demo.race_payload(run, ["heuristic", "table", "hybrid"])
    html = demo.render_race_html(payload, "heuristic", "hybrid", "dark")
    assert "__PAYLOAD__" not in html and html.count("<script>") == 1
    start, end = html.index("const P = ") + len("const P = "), html.index(";\n", html.index("const P = "))
    embedded = json.loads(html[start:end])
    assert [p["name"] for p in embedded["policies"]] == ["heuristic", "hybrid"]
    assert embedded["agreement"]["identical"] is True and embedded["theme"] == "dark"
    assert embedded["colors"]["bg"] == "#1a1a19" and embedded["policies"][1]["color"] == "#9085e9"
    assert payload["policies"][2]["color"] == "#4a3aa7"          # the cached payload is not mutated
    with pytest.raises(KeyError):
        demo.render_race_html(payload, "heuristic", "model")


def test_race_template_has_one_payload_slot():
    text = demo.RACE_TEMPLATE.read_text(encoding="utf-8")
    assert text.count("__PAYLOAD__") == 1 and "</script>" in text and 'id="play"' in text


def test_export_race_script_without_text(tmp_path):
    import subprocess, sys
    run = make_run(tmp_path, policies=("heuristic", "table", "hybrid"))
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_race.py"), "--run", str(run), "--no-text"],
                            capture_output=True, text=True, cwd=ROOT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stderr
    payload = demo.load_race(run)
    assert [p["name"] for p in payload["policies"]] == ["heuristic", "hybrid", "table"]
    assert "race.json" in result.stdout and "natural_text False" in result.stdout
