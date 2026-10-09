import streamlit as st
from backend import database as db
from views.overview import strip


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    from ui.theme import banner
    st.markdown(banner("Privacy Audit", "Evidence of what was exchanged. Only numeric updates cross the trust boundary."), unsafe_allow_html=True)
    st.write("Purpose: evidence of what was exchanged. Only numeric updates cross the trust boundary.")
    strip()
    jobs = db.list_jobs(20)
    if not jobs:
        st.warning("No runs yet."); return
    jid = st.selectbox("Run", [j["id"] for j in jobs])
    msgs = db.get_messages(jid)
    st.dataframe(msgs)
    if msgs:
        tot = sum(m["update_bytes"] for m in msgs)
        st.metric("Total bytes exchanged", f"{tot:,}")
        st.metric("Rows transmitted (sum)", sum(m["rows_transmitted"] for m in msgs))
        m0 = msgs[0]
        st.write("Message inspector (one real stored message): fields bank_id, round, "
                 f"n_samples={m0['n_samples']}, update_bytes={m0['update_bytes']}, "
                 f"payload_type={m0.get('payload_type', 'unknown')}, "
                 f"norm_before={m0['norm_before_clip']:.4f}, norm_after={m0['norm_after_clip']:.4f}, "
                 f"noise_std={m0['noise_std']}. rows_transmitted is 0 only together with "
                 "payload_type evidence that the stored payload was a validated numeric "
                 "update array (which has no row field); unknown payloads would stay NULL, "
                 "never silent zero. Raw DataFrames are rejected before insert.")
    st.warning("Noise protection is not a formal differential-privacy guarantee: no privacy accounting is implemented. "
               "Federated averaging alone does not prevent model-update leakage; secure aggregation is future work.")
