"""Classification and display of historical training runs.

Pure-function tests on backend.run_status (no database, no Streamlit, no
training). Synthetic job dicts mirror the real evidence in data/fedguard.db
and commit baf1391; assertions check the evidence chain, not job IDs alone.
"""
import json

from backend import run_status as RS

CLIP_CFG = {"banks": [0, 1, 2, 3, 4], "rounds": 5, "local_epochs": 1,
            "feature_mode": "raw", "clip_norm": 1.0, "noise_multiplier": 0.0,
            "sampling_frac": 1.0, "seed": 42}
NOCLIP_CFG = dict(CLIP_CFG, clip_norm=None)

# Real finished_at values from data/fedguard.db (UTC), kept as literals so
# the provenance ordering is asserted, not assumed from IDs.
JOB1_FINISHED = "2026-10-09T08:51:40.824628+00:00"
JOB2_FINISHED = "2026-10-09T08:55:22.869181+00:00"
JOB3_FINISHED = "2026-10-09T08:59:58.421078+00:00"


def _job(jid, cfg, finished, kind="training", status="completed"):
    return {"id": jid, "kind": kind, "status": status,
            "config_json": json.dumps(cfg) if cfg is not None else None,
            "requested_by": "system", "finished_at": finished}


def _metrics(prs, recs, jid=0, f1=0.7):
    return [{"job_id": jid, "round": i + 1, "pr_auc": pr,
             "recall_at_1pct_fpr": rc, "f1": f1}
            for i, (pr, rc) in enumerate(zip(prs, recs))]


def test_provenance_cutoff_separates_job2_from_job3():
    assert JOB2_FINISHED < RS.RECALL_FIX_CUTOFF_UTC < JOB3_FINISHED


def test_unclipped_job_labelled_from_config_not_scores():
    job = _job(1, NOCLIP_CFG, JOB1_FINISHED)
    m = _metrics([0.009, 0.044, 0.017, 0.016, 0.032], [0.0] * 5, jid=1)
    status, caveat = RS.classify_training_job(job, m, active_id=3)
    assert status == "Superseded — unstable config"
    assert "clip_norm" in caveat and "null" in caveat
    # Must not be mislabelled as the recall-metric bug: config is the cause.
    assert "ascending" not in caveat and "baf1391" not in caveat


def test_pre_fix_zero_recall_labelled_outdated_metric():
    job = _job(2, CLIP_CFG, JOB2_FINISHED)
    m = _metrics([0.747, 0.765, 0.757, 0.760, 0.754], [0.0] * 5, jid=2)
    status, caveat = RS.classify_training_job(job, m, active_id=3)
    assert status == "Superseded — outdated recall metric"
    assert "baf1391" in caveat and JOB2_FINISHED in caveat
    assert "PR-AUC/F1 are unaffected" in caveat


def test_active_model_with_current_results():
    job = _job(3, CLIP_CFG, JOB3_FINISHED)
    m = _metrics([0.747, 0.765, 0.757, 0.760, 0.754],
                 [0.874, 0.903, 0.906, 0.915, 0.906], jid=3)
    status, caveat = RS.classify_training_job(job, m, active_id=3)
    assert status == "Active model"
    assert "Score" in caveat
    # Same job viewed while not active is simply completed.
    status2, _ = RS.classify_training_job(job, m, active_id=99)
    assert status2 == "Completed"


def test_normal_low_performing_run_not_flagged_as_metric_bug():
    job = _job(42, CLIP_CFG, "2026-10-10T00:00:00+00:00")
    m = _metrics([0.03, 0.05, 0.04], [0.0, 0.0, 0.0], jid=42)
    status, caveat = RS.classify_training_job(job, m, active_id=3)
    assert status == "Completed"
    assert "outdated" not in status.lower()
    assert "Superseded" not in status


def test_uncertain_signature_needs_verification_not_invention():
    # Healthy-PR/all-zero-recall shape but no parseable finish time: the old
    # implementation cannot be confirmed, so no conclusion is invented.
    job = _job(43, CLIP_CFG, None)
    m = _metrics([0.70, 0.72], [0.0, 0.0], jid=43)
    status, _ = RS.classify_training_job(job, m, active_id=3)
    assert status == "Needs verification"
    # Completed training job with no metrics at all.
    job2 = _job(44, CLIP_CFG, "2026-10-10T00:00:00+00:00")
    status2, _ = RS.classify_training_job(job2, [], active_id=3)
    assert status2 == "Needs verification"


def test_non_training_and_incomplete_jobs_keep_neutral_labels():
    exp = _job(4, {"seeds": [42]}, "2026-10-09T13:21:48+00:00",
               kind="experiments")
    status, caveat = RS.classify_training_job(exp, [], active_id=3)
    assert status == "Not a training run"
    assert "methodology" in caveat
    queued = _job(50, CLIP_CFG, None, status="queued")
    status_q, _ = RS.classify_training_job(queued, [], active_id=3)
    assert "not completed" in status_q.lower()


def test_active_unstable_config_stays_identified_and_honest():
    job = _job(1, NOCLIP_CFG, JOB1_FINISHED)
    m = _metrics([0.009, 0.044], [0.0, 0.0], jid=1)
    status, caveat = RS.classify_training_job(job, m, active_id=1)
    assert "Active model" in status
    assert "clip_norm" in caveat


def test_annotate_keeps_every_run_with_visible_status():
    jobs = [_job(1, NOCLIP_CFG, JOB1_FINISHED),
            _job(2, CLIP_CFG, JOB2_FINISHED),
            _job(3, CLIP_CFG, JOB3_FINISHED),
            _job(4, {"seeds": [42]}, "2026-10-09T13:21:48+00:00",
                 kind="experiments")]
    by_job = {1: _metrics([0.03], [0.0], jid=1),
              2: _metrics([0.75], [0.0], jid=2),
              3: _metrics([0.75], [0.90], jid=3),
              4: []}
    rows = RS.annotate_jobs(jobs, by_job, active_id=3)
    assert [r["id"] for r in rows] == [1, 2, 3, 4]
    assert all("run_status" in r and "caveat" in r for r in rows)
    by_id = {r["id"]: r for r in rows}
    assert by_id[1]["run_status"] == "Superseded — unstable config"
    assert by_id[2]["run_status"] == "Superseded — outdated recall metric"
    assert by_id[3]["run_status"] == "Active model"
    assert by_id[4]["run_status"] == "Not a training run"
    assert all(by_id[i]["caveat"] for i in (1, 2, 4))
