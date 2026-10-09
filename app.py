"""FedGuard entry point: theme, backend start, auth gate, navigation."""
import streamlit as st

st.set_page_config(page_title="FedGuard", layout="wide", initial_sidebar_state="expanded")
from ui.theme import CSS
st.markdown(CSS, unsafe_allow_html=True)
from backend.jobs import startup, get_manager
from backend.auth import current_user, logout


@st.cache_resource
def _backend():
    startup()
    return get_manager()


_backend()
user = current_user()
if user is None:
    # Signed-out shell: brand-only sidebar, no navigation, no user content.
    # The auth gate below (st.stop) is what protects every page; this explicit
    # sidebar render additionally replaces any prior navigation entries.
    with st.sidebar:
        st.markdown("## FedGuard")
        st.caption("Privacy-preserving fraud signal sharing. Please sign in.")
    from views import login
    login.page()
    st.caption("Dataset: flwrlabs/fed-fraud-paysim-banks (CC-BY-4.0). Synthetic data. Prototype only.")
    st.stop()

st.session_state["_u"] = user
with st.sidebar:
    st.markdown("## FedGuard")
    st.caption("Privacy-preserving fraud signal sharing. Synthetic data prototype.")
    st.write(f"Signed in: {user['name']} ({user['role']})")
    if st.button("Sign out"):
        logout()
        st.rerun()

from views import (overview, data_explorer, control_room, experiments,
                   privacy_audit, score_transaction, threat_model, admin,
                   cross_bank_alerts)

pages = {
    "Monitor": [st.Page(overview.page, title="Overview", url_path="overview"),
                st.Page(data_explorer.page, title="Data Explorer", url_path="data-explorer")],
    "Federation": [st.Page(control_room.page, title="Control Room", url_path="control-room"),
                   st.Page(experiments.page, title="Experiments", url_path="experiments")],
    "Security": [st.Page(privacy_audit.page, title="Privacy Audit", url_path="privacy-audit"),
                 st.Page(threat_model.page, title="Threat Model", url_path="threat-model")],
    "Tools": [st.Page(score_transaction.page, title="Score Transaction", url_path="score-transaction"),
              st.Page(cross_bank_alerts.page, title="Cross-Bank Alerts", url_path="cross-bank-alerts")],
}
if user.get("role") == "admin":
    pages["Administration"] = [st.Page(admin.page, title="Administration", url_path="administration")]
st.caption("FedGuard prototype. Data: flwrlabs/fed-fraud-paysim-banks (CC-BY-4.0), synthetic. Not production-ready.")
st.navigation(pages).run()
