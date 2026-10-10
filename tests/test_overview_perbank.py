"""Per-bank Overview section loads the resolved active job's artifact.

Isolated: temporary results dir + temporary database. Never touches the
real fedguard.db, results/ artifacts, or saved models.
"""
import json

import pytest


@pytest.fixture
def tdb(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    p = tmp_path / "perbank_test.db"
    monkeypatch.setattr(C, "DB_PATH", p)
    monkeypatch.setattr(db, "DB_PATH", p)
    db.init_db()
    return db


@pytest.fixture
def rdir(tmp_path, monkeypatch):
    import backend.config as C
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "RESULTS_DIR", d)
    return d


def _completed(db, pr_auc=0.7):
    jid = db.create_job("training", {"rounds": 1}, "tester")
    db.set_job(jid, status="completed", finished_at=db.now())
    db.add_round_metric(jid, 1, pr_auc, 0.5, 0.6)
    return jid


def _entry(pr=0.75, rec90=0.9, tn=100, fp=1, fn=2, tp=3):
    return {"pr_auc": pr, "recall_at_1pct_fpr": rec90, "precision": 0.7,
            "recall": 0.6, "f1": 0.65,
            "confusion": [[tn, fp], [fn, tp]]}


def _write(rdir, jid, perbank=None, run=None, raw_perbank=None):
    if raw_perbank is not None:
        (rdir / f"run_{jid}_perbank.json").write_text(raw_perbank)
    elif perbank is not None:
        (rdir / f"run_{jid}_perbank.json").write_text(json.dumps(perbank))
    if run is not None:
        (rdir / f"run_{jid}.json").write_text(json.dumps(run))


def test_loads_active_job_perbank_and_threshold(tdb, rdir):
    from views.overview import load_perbank_metrics
    perbank = {str(b): _entry(pr=0.70 + b * 0.01, tn=100 + b) for b in range(5)}
    _write(rdir, 7, perbank=perbank,
           run={"job_id": 7, "threshold": 0.81, "config": {}})
    rows, thr, note = load_perbank_metrics(7)
    assert note is None
    assert thr == pytest.approx(0.81)
    assert [r["bank"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["pr_auc"] == pytest.approx(0.70)
    assert rows[4]["tn"] == 104


def test_newer_job_not_shown_when_older_active(tdb, rdir):
    from backend import database as db
    from views.overview import resolve_overview_job, load_perbank_metrics
    older = _completed(db, pr_auc=0.61)
    newer = _completed(db, pr_auc=0.82)
    db.set_setting("active_model", older)
    jid, rm, notice = resolve_overview_job()
    assert notice is None and jid == older
    _write(rdir, older,
           perbank={str(b): _entry(pr=0.61) for b in range(5)},
           run={"job_id": older, "threshold": 0.80, "config": {}})
    _write(rdir, newer,
           perbank={str(b): _entry(pr=0.82) for b in range(5)},
           run={"job_id": newer, "threshold": 0.88, "config": {}})
    rows, thr, note = load_perbank_metrics(jid)
    assert note is None and thr == pytest.approx(0.80)
    assert all(r["pr_auc"] == pytest.approx(0.61) for r in rows)
    assert all(r["pr_auc"] != pytest.approx(0.82) for r in rows)


def test_missing_artifact_handled(tdb, rdir):
    from views.overview import load_perbank_metrics
    rows, thr, note = load_perbank_metrics(77)
    assert rows == [] and thr is None
    assert note is not None and "77" in note and "not found" in note


def test_malformed_json_handled(tdb, rdir):
    from views.overview import load_perbank_metrics
    _write(rdir, 8, raw_perbank="{not json,,,",
           run={"job_id": 8, "threshold": 0.5, "config": {}})
    rows, thr, note = load_perbank_metrics(8)
    assert rows == [] and thr == pytest.approx(0.5)
    assert note is not None and "malformed" in note

    _write(rdir, 9, perbank=["not", "a", "dict"],
           run={"job_id": 9, "threshold": 0.5, "config": {}})
    rows, _, note = load_perbank_metrics(9)
    assert rows == [] and "unexpected structure" in note


def test_invalid_entries_skipped_with_caveat(tdb, rdir):
    from views.overview import load_perbank_metrics
    perbank = {str(b): _entry(pr=0.70) for b in range(5)}
    perbank["2"] = {"pr_auc": 0.7}  # missing keys
    perbank["4"] = _entry()
    perbank["4"]["confusion"] = [[1, 2]]  # wrong shape
    _write(rdir, 10, perbank=perbank,
           run={"job_id": 10, "threshold": 0.5, "config": {}})
    rows, _, note = load_perbank_metrics(10)
    assert [r["bank"] for r in rows] == [0, 1, 3]
    assert note is not None and "2" in note and "4" in note

    _write(rdir, 11, perbank={"0": {"pr_auc": "high"}},
           run={"job_id": 11, "threshold": 0.5, "config": {}})
    rows, _, note = load_perbank_metrics(11)
    assert rows == [] and "no usable bank entries" in note


def test_threshold_missing_shows_not_available(tdb, rdir):
    from views.overview import load_perbank_metrics
    perbank = {str(b): _entry() for b in range(5)}
    _write(rdir, 12, perbank=perbank,
           run={"job_id": 12, "config": {}})  # no threshold key
    rows, thr, note = load_perbank_metrics(12)
    assert len(rows) == 5 and thr is None
    assert note is not None and "Not available" in note

    _write(rdir, 13, perbank=perbank)  # no run file at all
    rows, thr, note = load_perbank_metrics(13)
    assert len(rows) == 5 and thr is None
    assert note is not None and "Not available" in note


def test_stale_zero_recall_artifact_flagged(tdb, rdir):
    from views.overview import load_perbank_metrics
    perbank = {str(b): _entry(rec90=0.0) for b in range(5)}
    _write(rdir, 14, perbank=perbank,
           run={"job_id": 14, "threshold": 0.8, "config": {}})
    rows, thr, note = load_perbank_metrics(14)
    assert len(rows) == 5
    assert note is not None and "outdated metric" in note


def test_invalid_job_id_rejected(tdb, rdir):
    from views.overview import load_perbank_metrics
    for bad in ("abc", "-3", None, True):
        rows, thr, note = load_perbank_metrics(bad)
        assert rows == [] and thr is None and "Invalid job id" in note
