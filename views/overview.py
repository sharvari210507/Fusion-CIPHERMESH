"""Shared status strip + brand sidebar content (static markup only, data from backend)."""
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
    # Latest completed run metrics
    jobs = [j for j in db.list_jobs(50) if j["kind"] == "training" and j["status"] == "completed"]
    if jobs:
        jid = jobs[0]["id"]
        rm = db.round_metrics(jid)
        if rm:
            last = rm[-1]
            m1, m2, m3 = st.columns(3)
            m1.metric("PR-AUC (global test)", f"{last['pr_auc']:.4f}")
            m2.metric("Recall @1% FPR", f"{last['recall_at_1pct_fpr']:.4f}")
            m3.metric("F1 @ operating threshold", f"{last['f1']:.4f}")
            st.line_chart([{"round": r["round"], "PR-AUC": r["pr_auc"],
                            "Recall@1%FPR": r["recall_at_1pct_fpr"]} for r in rm], x="round")
            st.caption(f"From completed training job {jid} on the held-out test split. "
                       "Accuracy is not reported: with ~0.1% fraud it is meaningless.")
    else:
        st.info("No completed training run yet. An admin can start one from the Control Room; "
                "a system default run starts automatically once the dataset is cached.")
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
