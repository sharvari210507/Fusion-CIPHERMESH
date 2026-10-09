import streamlit as st
from backend import database as db
from views.overview import strip


def page(user=None):
    import streamlit as st
    user = user if user is not None else st.session_state.get('_u')
    st.title("Privacy Audit")
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
                 f"norm_before={m0['norm_before_clip']:.4f}, norm_after={m0['norm_after_clip']:.4f}, "
                 f"noise_std={m0['noise_std']}. No DataFrame payload is accepted by the aggregator.")
    st.warning("Noise protection is not a formal differential-privacy guarantee: no privacy accounting is implemented. "
               "Federated averaging alone does not prevent model-update leakage; secure aggregation is future work.")
