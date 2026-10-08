from pathlib import Path

import pytest

from kernelscope import cli
from kernelscope.verify import CHECKS, Check, run_checks

REPO = Path(__file__).resolve().parents[1]


def _check(value, **kw):
    return Check(id="t", title="test", expected=10.0, tol=0.5, unit="ms", compute=lambda repo, data: value, **kw)


def test_check_passes_within_tolerance_and_fails_outside(tmp_path):
    t = run_checks([_check(10.4), _check(11.0)], repo=tmp_path, data=tmp_path)
    assert list(t.status) == ["PASS", "FAIL"]
    assert list(t.measured) == [10.4, 11.0]


def test_missing_data_is_reported_instead_of_raised(tmp_path):
    def gone(repo, data):
        raise FileNotFoundError(data / "summary.csv")

    t = run_checks([Check(id="m", title="missing", expected=1.0, tol=0.0, unit="", compute=gone)],
                   repo=tmp_path, data=tmp_path)
    assert t.status.item() == "MISSING"
    assert "summary.csv" in t.note.item()


def test_document_that_no_longer_states_the_value_is_flagged(tmp_path):
    (tmp_path / "doc.md").write_text("TPOT 61.06 → 33.80ms\n")
    t = run_checks([_check(10.0, doc="doc.md", doc_text="61.06 → 33.80ms"),
                    _check(10.0, doc="doc.md", doc_text="61.06 → 30.00ms")], repo=tmp_path, data=tmp_path)
    assert list(t.status) == ["PASS", "DOC_DRIFT"]


def test_committed_evidence_reproduces_every_documented_number():
    t = run_checks(CHECKS, repo=REPO, data=REPO / "demo_data")
    bad = t[t.status != "PASS"]
    assert bad.empty, bad[["id", "expected", "measured", "status", "note"]].to_string()
    assert {"kernel", "serve", "followup", "latency", "model", "consistency"} <= {i.split(".")[0] for i in t.id}


def test_verify_command_writes_the_table_and_exits_zero(tmp_path, capsys):
    out = tmp_path / "verify.csv"
    cli.main(["verify", "--only", "kernel,serve", "--out", str(out)])
    text = capsys.readouterr().out
    assert "PASS" in text and out.exists()
    assert "FAIL" not in out.read_text()


def test_verify_command_exits_nonzero_on_a_failed_check(monkeypatch):
    import kernelscope.verify as verify
    monkeypatch.setattr(verify, "CHECKS", [_check(99.0)])
    with pytest.raises(SystemExit) as e:
        cli.main(["verify"])
    assert e.value.code == 1


def test_verify_command_rejects_a_group_that_matches_no_check():
    with pytest.raises(SystemExit, match="valid groups"):
        cli.main(["verify", "--only", "kernal"])


def test_diagnose_checks_recompute_from_the_bundle():
    from kernelscope import verify
    repo = Path(__file__).resolve().parents[1]
    checks = [c for c in verify.CHECKS if c.id.startswith("diagnose.")]
    assert len(checks) == 7
    t = verify.run_checks(checks, repo=repo, data=repo / "demo_data")
    assert set(t.status) == {"PASS"}, t.to_string()


def test_hybrid_campaign_checks_recompute_from_the_bundle():
    from kernelscope import verify
    repo = Path(__file__).resolve().parents[1]
    checks = [c for c in verify.CHECKS if c.id.startswith(("hybrid.", "divergence.", "flashinfer.", "vllm."))]
    assert len(checks) == 64
    t = verify.run_checks(checks, repo=repo, data=repo / "demo_data")
    assert set(t.status) == {"PASS"}, t.to_string()


def test_problem_scope_checks_recompute_from_the_bundle():
    # 2026-10-06 problem-scope note: trace replays, two libraries' static defaults, the 64K long side.
    from kernelscope.verify import CHECKS, run_checks
    checks = [c for c in CHECKS if c.id.startswith(("traffic.", "defaults.", "longctx."))]
    assert len(checks) == 63
    table = run_checks(checks, REPO, REPO / "demo_data")
    assert table.status.tolist() == ["PASS"] * len(checks), table[table.status != "PASS"].to_string()


def test_fiengine_checks_recompute_from_the_bundle():
    # 2026-10-08 FlashInfer-in-the-engine campaign (note 2026-10-06-problem-scope.md §4): needs the packaged
    # demo_data/serve_4090/flashinfer_20261006 (scripts/package_demo.py); MISSING until the bundle holds it.
    from kernelscope.verify import CHECKS, run_checks
    checks = [c for c in CHECKS if c.id.startswith("fiengine.")]
    assert len(checks) == 148
    table = run_checks(checks, REPO, REPO / "demo_data")
    assert table.status.tolist() == ["PASS"] * len(checks), table[table.status != "PASS"].to_string()


def test_anytable_checks_recompute_from_the_bundle():
    # 2026-10-08 library-agnostic table campaign (note 2026-10-06-problem-scope.md §5): needs the packaged
    # demo_data/serve_4090/anytable_20261008 and demo_data/dispatch_paged_cold_any.csv (scripts/package_demo.py).
    from kernelscope.verify import CHECKS, run_checks
    checks = [c for c in CHECKS if c.id.startswith("anytable.")]
    assert len(checks) == 185
    table = run_checks(checks, REPO, REPO / "demo_data")
    assert table.status.tolist() == ["PASS"] * len(checks), table[table.status != "PASS"].to_string()
