import json
from pathlib import Path
import streamlit as st
from backend import database as db
from backend.jobs import get_manager
from backend.config import RESULTS_DIR, EXP_DEFAULT
from views.overview import strip


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    from ui.theme import banner
    st.markdown(banner("Experiments", "E1-E6 scenario results with generated captions. Pooled (E3) is a reference requiring raw-data sharing."), unsafe_allow_html=True)
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
    NAMES = {"E1_local": "E1 — Local-only baselines", "E2_fedsize": "E2 — Federation size (1-5 banks)",
             "E3_pooled_reference": "E3 — Pooled reference (requires raw-data sharing)",
             "E4_coldstart": "E4 — Cold-start bank", "E5_noniid": "E5 — Bank Divergence Radar (local vs federated per bank)",
             "E6_noise": "E6 — Update protection: privacy vs accuracy"}
    for k, v in E.items():
        st.subheader(NAMES.get(k, k))
        st.json(v)
        st.caption(_caption(k, v))


def _caption(k, v):
    try:
        if k == "E2_fedsize":
            d = v["5"]["mean_pr_auc"] - v["1"]["mean_pr_auc"]
            return f"5-bank federation differs from 1-bank by {d:+.4f} PR-AUC (mean over seeds)."
        if k == "E4_coldstart":
            d = v["joined_own_mean"] - v["alone_own_mean"]
            return (f"Target bank {v['target_bank']} ({v['limited_rows']} rows, "
                    f"{v['limited_fraud_rows']} fraud): joining changes own-partition PR-AUC "
                    f"by {d:+.4f}.")
        if k == "E6_noise":
            sw = v["sweep"] if isinstance(v, dict) and "sweep" in v else v
            clip = v.get("clip_norm_measured", "?") if isinstance(v, dict) else "?"
            return (f"Clip norm measured at {clip}; PR-AUC from {sw['0']:.4f} (no noise) "
                    f"to {sw['1.0']:.4f} (multiplier 1.0). Noise is not a formal guarantee.")
        if k == "E5_noniid":
            ds = [x["delta"] for x in v.values()]
            return (f"Federated minus local PR-AUC per bank (same partition): "
                    + ", ".join(f"bank {b}: {x['delta']:+.4f}" for b, x in v.items())
                    + ". Negative values are reported honestly.")
        return "Values are means over seeds where applicable."
    except Exception:
        return "See table."
