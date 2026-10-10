"""Overview shows the active scoring model's metrics, not silently the latest.

Uses an isolated temp DB and existing database abstractions only;
no dataset loading or model training.
"""
import pytest


@pytest.fixture
def tdb(tmp_path, monkeypatch):
    import backend.database as db
    import backend.config as C
    p = tmp_path / "overview_test.db"
    monkeypatch.setattr(C, "DB_PATH", p)
    monkeypatch.setattr(db, "DB_PATH", p)
    db.init_db()
    return db


def _completed_job(db, pr_auc=0.7, kind="training"):
    jid = db.create_job(kind, {"rounds": 1}, "tester")
    db.set_job(jid, status="completed", finished_at=db.now())
    if kind == "training":
        db.add_round_metric(jid, 1, pr_auc, 0.5, 0.6)
    return jid


def test_active_newest_completed_job(tdb):
    from backend import database as db
    from views.overview import resolve_overview_job
    j1 = _completed_job(db, pr_auc=0.60)
    j2 = _completed_job(db, pr_auc=0.80)
    db.set_setting("active_model", j2)
    jid, rm, notice = resolve_overview_job()
    assert notice is None
    assert jid == j2
    assert rm and rm[-1]["pr_auc"] == pytest.approx(0.80)
    assert jid != j1


def test_active_older_completed_job_while_newer_exists(tdb):
    from backend import database as db
    from views.overview import resolve_overview_job
    older = _completed_job(db, pr_auc=0.61)
    newer = _completed_job(db, pr_auc=0.82)
    db.set_setting("active_model", older)
    jid, rm, notice = resolve_overview_job()
    assert notice is None
    assert jid == older
    # Proves we did not silently take the newest job's metrics.
    assert rm[-1]["pr_auc"] == pytest.approx(0.61)
    assert newer != jid


def test_active_missing_invalid_or_without_metrics_labels_fallback(tdb):
    from backend import database as db
    from views.overview import resolve_overview_job
    j1 = _completed_job(db, pr_auc=0.70)
    j2 = _completed_job(db, pr_auc=0.75)
    # Case A: no active_model setting at all.
    jid, rm, notice = resolve_overview_job()
    assert jid == j2 and rm
    assert notice is not None and str(j2) in notice
    assert "do not describe" in notice
    # Case B: invalid (non-integer) setting.
    db.set_setting("active_model", "not-a-job")
    jid, rm, notice = resolve_overview_job()
    assert jid == j2 and notice is not None
    assert "not-a-job" in notice and str(j2) in notice
    # Case C: active points at a job with no completed metrics
    # (queued training job has no round_metrics).
    pending = db.create_job("training", {"rounds": 1}, "tester")
    db.set_setting("active_model", pending)
    jid, rm, notice = resolve_overview_job()
    assert jid == j2 and rm
    assert notice is not None
    assert str(pending) in notice and str(j2) in notice
    # Case D: active points at a completed training job with no metrics.
    empty = db.create_job("training", {"rounds": 1}, "tester")
    db.set_job(empty, status="completed", finished_at=db.now())
    db.set_setting("active_model", empty)
    jid, rm, notice = resolve_overview_job()
    assert jid == j2 and rm
    assert str(empty) in notice and str(j2) in notice
    assert j1 in (j1, j2)  # sanity: older job with metrics still present


def test_no_completed_metrics_reports_explicitly(tdb):
    from backend import database as db
    from views.overview import resolve_overview_job
    # No jobs at all.
    jid, rm, notice = resolve_overview_job()
    assert (jid, rm, notice) == (None, [], "none")
    # Only an experiments job (never valid for model metrics).
    ej = db.create_job("experiments", {}, "tester")
    db.set_job(ej, status="completed", finished_at=db.now())
    db.set_setting("active_model", ej)
    jid, rm, notice = resolve_overview_job()
    assert (jid, rm, notice) == (None, [], "none")
    # Completed training jobs exist but none have metrics: explicit, no fallback.
    empty = db.create_job("training", {"rounds": 1}, "tester")
    db.set_job(empty, status="completed", finished_at=db.now())
    db.set_setting("active_model", empty)
    jid, rm, notice = resolve_overview_job()
    assert jid is None and rm == []
    assert notice not in (None, "none")
    assert "no" in notice.lower() and "metric" in notice.lower()


def test_chart_and_kpi_use_same_selected_job(tdb):
    from backend import database as db
    from views.overview import resolve_overview_job
    older = _completed_job(db, pr_auc=0.61)
    db.add_round_metric(older, 2, 0.63, 0.55, 0.64)
    newer = _completed_job(db, pr_auc=0.82)
    db.set_setting("active_model", older)
    jid, rm, notice = resolve_overview_job()
    assert notice is None and jid == older
    # The KPI row (last) and the chart series come from the same list:
    # two rounds, ascending, both belonging to the active job.
    assert [r["round"] for r in rm] == [1, 2]
    kpi = rm[-1]
    assert (kpi["pr_auc"], kpi["recall_at_1pct_fpr"], kpi["f1"]) == \
        pytest.approx((0.63, 0.55, 0.64))
    assert all(r["job_id"] == older for r in rm)
    assert newer != jid
