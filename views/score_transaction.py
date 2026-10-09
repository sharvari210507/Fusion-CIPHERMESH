import streamlit as st
from backend import scoring as S
from backend import database as db
from backend.config import TYPES
from views.overview import strip


def page(user=None):
    import streamlit as st
    user = user if user is not None else st.session_state.get('_u')
    st.title("Score Transaction")
    st.write("Purpose: score one transaction with the active federated model.")
    strip()
    jid = db.get_setting("active_model")
    if not jid:
        st.warning("No active model yet. Train a run first."); return
    st.caption(f"Active model: job {jid}")
    with st.form("score"):
        t = st.selectbox("Type", TYPES)
        amount = st.number_input("Amount", 0.0, 1e9, 1000.0)
        obo = st.number_input("Old balance origin", 0.0, 1e9, 1000.0)
        nbo = st.number_input("New balance origin", 0.0, 1e9, 0.0)
        obd = st.number_input("Old balance destination", 0.0, 1e9, 0.0)
        nbd = st.number_input("New balance destination", 0.0, 1e9, 1000.0)
        hour = st.number_input("Hour (0-23)", 0, 23, 12)
        bank = st.selectbox("Bank", [0, 1, 2, 3, 4])
        go = st.form_submit_button("Score")
    c1, c2 = st.columns(2)
    if c1.button("Load real fraud example"):
        st.session_state["ex"] = S.example_row(True)
    if c2.button("Load real non-fraud example"):
        st.session_state["ex"] = S.example_row(False)
    ex = st.session_state.get("ex")
    if ex:
        st.caption(f"Example row from test split: {ex}")
    if go:
        try:
            r = S.score({"type": t, "amount": amount, "oldbalanceOrg": obo,
                         "newbalanceOrig": nbo, "oldbalanceDest": obd,
                         "newbalanceDest": nbd, "hour": hour},
                        int(jid), int(bank), user)
            st.metric("P(fraud) global", f"{r['p_global']:.4f}")
            st.write(f"Decision at threshold {r['threshold']:.3f}: {r['decision']}")
            st.write("Top feature contributions (coefficient x value):")
            st.dataframe([{"feature": k, "contribution": round(v, 4)} for k, v in r["contributions"]])
        except Exception as e:
            st.error(str(e))
