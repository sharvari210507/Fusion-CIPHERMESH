"""Honest status labels for historical training runs (presentation only).

Never deletes or rewrites records: classification is derived at display time
from each run's saved ``config_json`` and metric provenance, never from a
low score alone. A run that cannot be classified confidently is labelled
"Needs verification" rather than given an invented conclusion.

Evidence baked in (all verifiable in this repository):
- Unclipped default: ``backend/config.py`` ``DEFAULT_RUN`` had
  ``clip_norm=None`` until commit ``baf1391`` ("fix: correct recall-at-FPR
  ranking; default clip 1.0; radar labels", 2026-10-09 14:29:05 +0530), which
  changed it to ``1.0``. A saved training config with ``clip_norm`` null
  therefore ran with unclipped updates.
- Recall ranking bug: the same commit fixed
  ``backend/metrics.py::recall_at_fpr`` from ``np.argsort(s)`` (ascending:
  lowest scores first) to ``np.argsort(-s)`` (descending). Jobs whose
  ``finished_at`` predates the fix and whose stored ``recall_at_1pct_fpr``
  is exactly ``0.0`` in every round despite healthy PR-AUC carry the old
  implementation's output. In UTC the fix landed at
  ``2026-10-09T08:59:05+00:00``; job ``finished_at`` values are stored in
  UTC, so the comparison is like-for-like.
"""

import json
from datetime import datetime, timezone

# Commit baf1391 in UTC (14:29:05 +0530). Training jobs finishing strictly
# before this instant ran the ascending-sort recall implementation.
RECALL_FIX_CUTOFF_UTC = "2026-10-09T08:59:05+00:00"

# Guard so a merely bad model is never labelled a metric bug: the outdated
# recall signature additionally requires a healthy best PR-AUC. The value is
# a coarse sanity floor, not a quality judgement.
PR_HEALTHY_FLOOR = 0.5


def _parse_config(job):
    try:
        cfg = json.loads(job.get("config_json") or "{}")
    except (TypeError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _parse_ts(value):
    if not value or not isinstance(value, str):
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _finished_before_fix(job):
    ts = _parse_ts(job.get("finished_at"))
    if ts is None:
        return None
    return ts < _parse_ts(RECALL_FIX_CUTOFF_UTC)


def _recall_values(metrics):
    try:
        return [float(m["recall_at_1pct_fpr"]) for m in metrics]
    except (KeyError, TypeError, ValueError):
        return None


def _best_pr_auc(metrics):
    try:
        return max(float(m["pr_auc"]) for m in metrics)
    except (KeyError, TypeError, ValueError):
        return None


def _is_active(job, active_id):
    if active_id is None:
        return False
    try:
        return int(job.get("id")) == int(str(active_id).strip())
    except (TypeError, ValueError):
        return False


def classify_training_job(job, metrics, active_id=None):
    """Return ``(status, caveat)`` for one historical job dict.

    ``job`` is a row as returned by ``database.list_jobs``; ``metrics`` is
    its ``database.round_metrics`` list (possibly empty); ``active_id`` is
    the raw ``active_model`` setting value. Pure function: no I/O, no
    retraining, no record changes.
    """
    active = _is_active(job, active_id)
    kind = job.get("kind")
    if kind != "training":
        status = "Not a training run"
        caveat = ("Experiment/suite job: produced under a different "
                  "methodology (see Experiments page); not comparable with "
                  "federated training runs.")
        if active:
            status = "Active model — not a training run"
            caveat += " The active-model setting currently points here, so scoring is unavailable."
        return status, caveat
    if (job.get("status") or "") != "completed":
        state = (job.get("status") or "unknown").strip() or "unknown"
        status = f"{state.capitalize()} — not completed"
        caveat = "Only completed runs carry comparable round metrics."
        if active:
            status = f"Active model — {state}"
            caveat += " The active-model setting currently points here, so scoring is unavailable."
        return status, caveat
    if not metrics:
        status = "Needs verification"
        caveat = "Completed training job but no round metrics were recorded; nothing to compare."
        if active:
            status = "Active model — no metrics"
        return status, caveat
    cfg = _parse_config(job)
    unclipped = cfg is not None and cfg.get("clip_norm") is None
    recs = _recall_values(metrics)
    best_pr = _best_pr_auc(metrics)
    outdated_recall = (
        recs is not None and best_pr is not None
        and len(recs) > 0 and all(r == 0.0 for r in recs)
        and best_pr >= PR_HEALTHY_FLOOR
    )
    if unclipped:
        status = "Superseded — unstable config"
        caveat = ("Saved config has clip_norm null (unclipped updates), the "
                  "pre-fix default; update norms were not bounded, so this "
                  "run is not comparable with clipped runs. Labelled from "
                  "configuration, not from its scores.")
    elif outdated_recall and _finished_before_fix(job) is True:
        status = "Superseded — outdated recall metric"
        caveat = (f"Finished at {job.get('finished_at')}, before the "
                  f"recall-ranking fix (baf1391, cutoff "
                  f"{RECALL_FIX_CUTOFF_UTC}); stored recall@1%FPR is 0.0 in "
                  f"every round from the old ascending-sort implementation. "
                  f"PR-AUC/F1 are unaffected.")
    elif outdated_recall:
        status = "Needs verification"
        caveat = ("recall@1%FPR is 0.0 in every round despite healthy PR-AUC, "
                  "but the run cannot be confidently tied to the old metric "
                  "implementation from available evidence.")
    else:
        status = "Completed"
        caveat = ""
    if active:
        if status == "Completed":
            return "Active model", "Currently scoring transactions: the Score page loads this job's artifact."
        return f"Active model — {status}", caveat
    return status, caveat


def annotate_jobs(jobs, metrics_by_job, active_id):
    """Attach ``run_status``/``caveat`` display columns to every job row.

    Input order and rows are preserved: nothing is filtered or reordered,
    so superseded runs stay available for inspection alongside their
    explanation.
    """
    rows = []
    for j in jobs:
        status, caveat = classify_training_job(
            j, metrics_by_job.get(j.get("id"), []), active_id)
        rows.append({
            "id": j.get("id"),
            "kind": j.get("kind"),
            "job_status": j.get("status"),
            "requested_by": j.get("requested_by"),
            "finished_at": j.get("finished_at"),
            "run_status": status,
            "caveat": caveat,
        })
    return rows
