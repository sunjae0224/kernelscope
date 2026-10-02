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
