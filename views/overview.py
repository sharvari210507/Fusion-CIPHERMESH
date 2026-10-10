"""Shared status strip + brand sidebar content (static markup only, data from backend)."""
import json

import streamlit as st
from ui.theme import banner, badge
from backend import data as D, database as db
from backend.jobs import get_manager


def strip():
    """Thin shared status strip shown on every page (data from backend only)."""
    status_strip()


def status_strip():
    mgr = get_manager()
    stt = mgr.status()
    am = db.get_setting("active_model")
    ds = "ready" if D.is_cached() else "downloading"
    cur = stt["current"]
    if cur:
        j = db.get_job(cur)
        prog = len(db.round_metrics(cur))
        st.info(f"Dataset: {ds} | Active model job: {am or 'none'} | "
                f"Running job {cur} ({j['kind']}, round progress: {prog}) | Queued: {len(stt['queued'])}")
    else:
        st.caption(f"Dataset: {ds} | Active model job: {am or 'none'} | "
                   f"No job running | Queued: {len(stt['queued'])}")


def resolve_overview_job(limit=50):
    """Select the training job whose metrics Overview must display.

    Returns (job_id, round_metrics, notice) where notice is None when the
    active model itself supplies metrics, otherwise an explicit message
    identifying the active setting and any fallback job. Never silently
    substitutes an unrelated job: callers must surface a non-None notice.
    Uses only existing database abstractions (list_jobs/get_setting/
    round_metrics); performs no dataset loading or training.
    """
    jobs = [j for j in db.list_jobs(limit)
            if j["kind"] == "training" and j["status"] == "completed"]
    if not jobs:
        return None, [], "none"
    with_metrics = []
    for j in jobs:
        rm = db.round_metrics(j["id"])
        if rm:
            with_metrics.append((j["id"], rm))
    if not with_metrics:
        if jobs:
            return (None, [],
                    f"{len(jobs)} completed training job(s) exist but none "
                    f"have recorded metrics; no model metrics to display.")
        return None, [], "none"
    latest_id = with_metrics[0][0]
    raw = db.get_setting("active_model")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return (latest_id, with_metrics[0][1],
                f"No active model is set; showing latest completed job "
                f"{latest_id} with metrics instead — these metrics do not "
                f"describe a scoring model.")
    try:
        active = int(str(raw).strip())
    except (TypeError, ValueError):
        return (latest_id, with_metrics[0][1],
                f"Active model setting {raw!r} is invalid; showing latest "
                f"completed job {latest_id} with metrics instead — these "
                f"metrics do not describe the scoring model.")
    for jid, rm in with_metrics:
        if jid == active:
            return active, rm, None
    completed_ids = {j["id"] for j in jobs}
    if active not in completed_ids:
        return (latest_id, with_metrics[0][1],
                f"Active model job {active} has no completed training job "
                f"with metrics; showing latest completed job {latest_id} "
                f"with metrics instead — these metrics do not describe the "
                f"scoring model.")
    return (latest_id, with_metrics[0][1],
            f"Active model job {active} is a completed training job but has "
            f"no recorded metrics; showing latest completed job {latest_id} "
            f"with metrics instead — these metrics do not describe the "
            f"scoring model.")


def _valid_job_id(jid):
    try:
        if isinstance(jid, bool):
            return None
        value = int(str(jid).strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return value if value >= 0 else None


def load_perbank_metrics(jid):
    """Load the per-bank test artifact for exactly one resolved job.

    Returns ``(rows, threshold, note)``: ``rows`` is a list of one dict per
    simulated bank (empty when the artifact is missing or unusable),
    ``threshold`` is the persisted operating threshold or None, and ``note``
    is None on success or an explicit caveat naming the job and problem.
    Reads only ``results/run_{jid}_perbank.json`` (+ ``run_{jid}.json`` for
    the threshold); never trains, never substitutes another job's file.
    """
    from backend.config import RESULTS_DIR
    clean = _valid_job_id(jid)
    if clean is None:
        return [], None, f"Invalid job id {jid!r}; no per-bank metrics to display."
    perbank_path = RESULTS_DIR / f"run_{clean}_perbank.json"
    run_path = RESULTS_DIR / f"run_{clean}.json"
    threshold = None
    try:
        run_doc = json.loads(run_path.read_text())
        thr = run_doc.get("threshold") if isinstance(run_doc, dict) else None
        if isinstance(thr, bool) or not isinstance(thr, (int, float)):
            threshold = None
        elif thr != thr or thr in (float("inf"), float("-inf")):
            threshold = None
        else:
            threshold = float(thr)
    except (OSError, ValueError):
        threshold = None
    if not perbank_path.exists():
        return [], threshold, (
            f"Per-bank artifact results/run_{clean}_perbank.json was not found, "
            f"so no per-bank breakdown is shown for job {clean}.")
    try:
        doc = json.loads(perbank_path.read_text())
    except ValueError:
        return [], threshold, (
            f"Per-bank artifact results/run_{clean}_perbank.json is malformed "
            f"and cannot be displayed for job {clean}.")
    if not isinstance(doc, dict):
        return [], threshold, (
            f"Per-bank artifact results/run_{clean}_perbank.json has an "
            f"unexpected structure for job {clean}; no per-bank metrics to display.")
    rows, problems = [], []
    for bank in range(5):
        entry = doc.get(str(bank), doc.get(bank))
        row = {"bank": bank}
        try:
            if not isinstance(entry, dict):
                raise ValueError("entry is not an object")
            for key in ("pr_auc", "recall_at_1pct_fpr", "precision", "recall", "f1"):
                value = entry[key]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"{key} is not numeric")
                row[key] = round(float(value), 4)
            tn, fp, fn, tp = entry["confusion"][0][0], entry["confusion"][0][1], \
                entry["confusion"][1][0], entry["confusion"][1][1]
            for name, count in (("tn", tn), ("fp", fp), ("fn", fn), ("tp", tp)):
                if isinstance(count, bool) or not isinstance(count, int):
                    raise ValueError(f"{name} is not an integer count")
                row[name] = count
            rows.append(row)
        except (KeyError, TypeError, ValueError, IndexError):
            problems.append(str(bank))
    if not rows:
        return [], threshold, (
            f"Per-bank artifact results/run_{clean}_perbank.json contains no "
            f"usable bank entries for job {clean}.")
    note = None
    parts = []
    if problems:
        parts.append(f"bank(s) {', '.join(problems)} are not shown (entry missing or invalid)")
    recs = [r["recall_at_1pct_fpr"] for r in rows]
    if len(recs) == 5 and all(r == 0.0 for r in recs):
        parts.append("recall@1%FPR reads 0.0 for every bank in this saved artifact "
                     "(outdated metric implementation; see the run history status)")
    if threshold is None:
        parts.append("operating threshold is Not available — the run artifact did not "
                     "record one, and precision/recall/F1 depend on the threshold")
    if parts:
        note = f"Per-bank caveat for job {clean}: " + "; ".join(parts) + "."
    return rows, threshold, note


def _perbank_section(jid):
    """Compact per-bank test evaluation for the already-resolved job."""
    st.subheader("Per-bank evaluation (active model)")
    rows, threshold, note = load_perbank_metrics(jid)
    if not rows:
        st.warning(note or f"No per-bank metrics to display for job {jid}.")
        return
    st.dataframe([{"bank": r["bank"], "PR-AUC": r["pr_auc"],
                   "Recall@1%FPR": r["recall_at_1pct_fpr"],
                   "Precision": r["precision"], "Recall": r["recall"],
                   "F1": r["f1"], "TN": r["tn"], "FP": r["fp"],
                   "FN": r["fn"], "TP": r["tp"]} for r in rows],
                 use_container_width=True)
    if threshold is None:
        st.caption("Evaluation threshold: Not available for job "
                   f"{jid} — precision, recall and F1 depend on the threshold.")
    else:
        st.caption(f"Evaluation threshold {threshold:.3f} (persisted with run {jid}; "
                   f"same threshold as the aggregate F1 above).")
    if note:
        st.caption(note)
    st.caption("Simulated PaySim data — held-out test split, per-bank partitions. "
               "Banks differ in transaction mix and fraud rates, so per-bank results "
               "differ; the aggregate above can hide a weak individual bank.")
    st.caption("Prototype results on synthetic data only: not real-world bank "
               "performance and not production readiness.")


def page():
    st.markdown(banner("Overview",
        "Live status of the federation: data, model, experiments and recent activity."),
        unsafe_allow_html=True)
    status_strip()
    s = D.get_stats()
    if "banks" not in s:
        st.warning("Dataset is still downloading in the background (first boot needs internet once). "
                   "No metrics exist yet — this is a loading state, not a zero result. "
                   "An admin-started default run begins automatically once the cache lands.")
        return
    c = st.columns(4)
    c[0].metric("Train rows", f"{s['n_train']:,}")
    c[1].metric("Test rows", f"{s['n_test']:,}")
    c[2].metric("Fraud rate (train)", f"{s['fraud_rate_train']*100:.3f}%")
    c[3].metric("Simulated banks", "5")
    st.dataframe([{**{"bank": b}, **v} for b, v in s["banks"].items()],
                 use_container_width=True)
    # Metrics for the active scoring model (same job the Score page loads).
    jid, rm, notice = resolve_overview_job()
    if jid is not None and rm:
        if notice:
            st.warning(notice)
        last = rm[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("PR-AUC (global test)", f"{last['pr_auc']:.4f}")
        m2.metric("Recall @1% FPR", f"{last['recall_at_1pct_fpr']:.4f}")
        m3.metric("F1 @ operating threshold", f"{last['f1']:.4f}")
        st.line_chart([{"round": r["round"], "PR-AUC": r["pr_auc"],
                        "Recall@1%FPR": r["recall_at_1pct_fpr"]} for r in rm], x="round")
        if notice is None:
            st.caption(f"From completed training job {jid} (active model) on the held-out test split. "
                       "Accuracy is not reported: with ~0.1% fraud it is meaningless.")
        else:
            st.caption(f"From completed training job {jid} (fallback) on the held-out test split. "
                       "Accuracy is not reported: with ~0.1% fraud it is meaningless.")
        _perbank_section(jid)
    elif notice == "none":
        st.info("No completed training run yet. An admin can start one from the Control Room; "
                "a system default run starts automatically once the dataset is cached.")
    else:
        st.warning(notice)
    # Recent audit activity
    st.subheader("Recent activity")
    try:
        rows = db.query("SELECT created_at,user_id,action,detail FROM audit_log ORDER BY id DESC", (), 10)
        if rows:
            st.dataframe(rows, use_container_width=True)
        else:
            st.caption("No audit events recorded yet.")
    except Exception as e:
        st.caption(f"Activity unavailable: {e}")
    st.caption("Use the sidebar Federation section to open the Control Room.")
