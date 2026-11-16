"""Promotion gates, artifact integrity and rollback, on a throwaway registry."""

import json

import pytest

from churn import registry


@pytest.fixture
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "ROOT", tmp_path)
    monkeypatch.setattr(registry, "INDEX", tmp_path / "registry.json")
    return tmp_path


def metric(v, lo=None, hi=None):
    return {"value": v, "ci": [lo if lo is not None else v, hi if hi is not None else v]}


def evaluation(auc=0.85, ece=0.02, profit=30_000.0, gap=0.02):
    return {
        "metrics": {"roc_auc": metric(auc), "ece": metric(ece), "profit": metric(profit)},
        "subgroups": [{"column": "Contract", "level": "Two year", "n": 300, "calibration_gap": gap}],
        "everyone_profit": -40_000.0,
    }


@pytest.mark.parametrize("bad", [{"auc": 0.7}, {"ece": 0.09}, {"profit": -1.0}, {"gap": 0.2}])
def test_gate_blocks_a_broken_first_model(bad):
    assert not registry.gate(evaluation(**bad), None)["passed"]


def test_gate_blocks_a_challenger_worse_than_the_champion():
    champion = evaluation()
    worse = evaluation() | {"vs_champion": {"log_loss": metric(0.03, 0.02, 0.04), "profit": metric(-6000, -9000, -3000)}}
    report = registry.gate(worse, champion)
    assert not report["passed"]
    assert [c["passed"] for c in report["checks"][-2:]] == [False, False]
    tied = evaluation() | {"vs_champion": {"log_loss": metric(0.0, -0.005, 0.005), "profit": metric(0, -2000, 2000)}}
    assert registry.gate(tied, champion)["passed"]


def fake_version(reg, name, content=b"model-bytes"):
    folder = reg / name
    folder.mkdir()
    (folder / "model.joblib").write_bytes(content)
    (folder / "report.json").write_text(
        json.dumps({"version": name, "artifact_sha256": registry.sha256(folder / "model.joblib")})
    )


def test_tampered_artifact_is_refused(reg, monkeypatch):
    monkeypatch.setattr(registry.joblib, "load", lambda path: "model")
    fake_version(reg, "v1")
    registry.promote("v1", {"passed": True})
    assert registry.load()[0] == "model"
    (reg / "v1" / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="SHA-256"):
        registry.load()


def test_failed_gate_cannot_be_promoted(reg):
    fake_version(reg, "v1")
    with pytest.raises(RuntimeError, match="gate"):
        registry.promote("v1", {"passed": False})


def test_rollback_steps_back_through_champions(reg, monkeypatch):
    monkeypatch.setattr(registry.joblib, "load", lambda path: "model")
    for v in ("v1", "v2", "v3"):
        fake_version(reg, v, v.encode())
        registry.promote(v, {"passed": True})
    assert registry.rollback() == "v2"
    assert registry.rollback() == "v1"
    with pytest.raises(RuntimeError, match="Nothing to roll back"):
        registry.rollback()
    assert registry.index()["champion"] == "v1"
