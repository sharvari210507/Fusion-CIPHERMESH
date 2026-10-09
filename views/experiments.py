import json
from pathlib import Path
import streamlit as st
from backend import database as db
from backend.jobs import get_manager
from backend.config import RESULTS_DIR, EXP_DEFAULT
from views.overview import strip


def page(user=None):
    import streamlit as st
    user = user if user is not None else st.session_state.get('_u')
    st.title("Experiments")
    st.write("Purpose: E1-E6 scenario results with generated captions. Pooled (E3) is a reference requiring raw-data sharing.")
    strip()
    p = RESULTS_DIR / "experiments.json"
    admin = user and user.get("role") == "admin"
    if st.button("Run experiment suite", disabled=not admin):
        jid = get_manager().submit("experiments", dict(EXP_DEFAULT), user)
        st.write(f"Started experiments job {jid}.")
    if not p.exists():
        st.warning("No experiment results yet."); return
    out = json.loads(p.read_text())
    st.caption(f"Config: {out.get('config')}")
    E = out.get("E", {})
    for k, v in E.items():
        st.subheader(k)
        st.json(v)
        st.caption(_caption(k, v))


def _caption(k, v):
    try:
        if k == "E2_fedsize":
            d = v["5"]["mean_pr_auc"] - v["1"]["mean_pr_auc"]
            return f"5-bank federation differs from 1-bank by {d:+.4f} PR-AUC (mean over seeds)."
        if k == "E4_coldstart":
            d = v["joined_mean"] - v["alone_mean"]
            return f"Target bank {v['target_bank']}: joining changes PR-AUC by {d:+.4f} on its own test rows."
        if k == "E6_noise":
            return f"Noise sweep PR-AUC from {v['0']:.4f} (no noise) to {v['1.0']:.4f} (multiplier 1.0)."
        return "Values are means over seeds where applicable."
    except Exception:
        return "See table."
