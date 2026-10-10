import streamlit as st
import plotly.express as px
from backend import data as D, database as db
from backend.jobs import get_manager
from backend.config import DEFAULT_RUN
from views.overview import strip


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    from ui.theme import banner
    st.markdown(banner("Control Room", "Configure, start and follow federated training. Training actions require admin role."), unsafe_allow_html=True)
    st.write("Purpose: configure, start and follow federated training. Admin only for actions.")
    strip()
    admin = user and user.get("role") == "admin"
    with st.form("cfg"):
        banks = st.multiselect("Banks", [0, 1, 2, 3, 4], default=[0, 1, 2, 3, 4], disabled=not admin)
        rounds = st.number_input("Rounds", 1, 50, DEFAULT_RUN["rounds"], disabled=not admin)
        epochs = st.number_input("Local epochs", 1, 20, DEFAULT_RUN["local_epochs"], disabled=not admin)
        mode = st.selectbox("Feature mode", ["raw", "engineered"], disabled=not admin)
        clip = st.number_input("Clip norm (0=none)", 0.0, 100.0, 0.0, disabled=not admin)
        noise = st.number_input("Noise multiplier", 0.0, 5.0, 0.0, disabled=not admin)
        frac = st.number_input("Non-fraud sampling fraction", 0.01, 1.0, 1.0, disabled=not admin)
        seed = st.number_input("Seed", 0, 9999, 42, disabled=not admin)
        go = st.form_submit_button("Start run", disabled=not admin)
    if not admin:
        st.info("Analyst role: form is read-only. An admin starts runs.")
    if go and admin:
        cfg = dict(banks=banks, rounds=int(rounds), local_epochs=int(epochs),
                   feature_mode=mode, clip_norm=(float(clip) or None),
                   noise_multiplier=float(noise), sampling_frac=float(frac), seed=int(seed))
        jid = get_manager().submit("training", cfg, user)
        st.write(f"Started job {jid}.")

    @st.fragment(run_every=3)
    def live():
        from backend import run_status as RS
        jobs = db.list_jobs(10)
        st.write("Run history")
        raw_active = db.get_setting("active_model")
        try:
            active_id = int(str(raw_active).strip())
        except (TypeError, ValueError, AttributeError):
            active_id = None
        metrics_by_job = {j["id"]: db.round_metrics(j["id"]) for j in jobs}
        st.dataframe(RS.annotate_jobs(jobs, metrics_by_job, active_id))
        st.caption("Run status is derived at display time from each run's saved "
                   "config and metric provenance; low scores alone never mark a run. "
                   "Superseded runs remain available for inspection with their reason. "
                   "The active model is the job the Score page loads.")
        if jobs:
            jid = jobs[0]["id"]
            rm = db.round_metrics(jid)
            if rm:
                st.line_chart([{"round": r["round"], "PR-AUC": r["pr_auc"],
                                "Recall@1%FPR": r["recall_at_1pct_fpr"]} for r in rm], x="round")
    live()
    jobs = db.list_jobs(5)
    if jobs and admin:
        jid = st.number_input("Set active model job id", value=jobs[0]["id"], step=1)
        if st.button("Set active"):
            db.set_setting("active_model", int(jid))
            db.audit(user["name"], "active_model", f"job {jid}")
            st.write("Active model updated.")
        if st.button("Cancel current"):
            get_manager().cancel(jobs[0]["id"], user)
            st.write("Cancel requested.")
