import streamlit as st
from backend import data as D, database as db

STATUS = "status"


def strip():
    ds = "ready" if D.is_cached() else "downloading/missing"
    am = db.get_setting("active_model")
    st.caption(f"Dataset: {ds} | Active model job: {am or 'none'} | Synthetic PaySim data (CC-BY-4.0, flwrlabs/fed-fraud-paysim-banks)")


def page():
    st.title("Overview")
    st.write("Purpose: monitor federated fraud training across five simulated banks.")
    strip()
    s = D.get_stats()
    if "banks" in s:
        c = st.columns(3)
        c[0].metric("Train rows", f"{s['n_train']:,}")
        c[1].metric("Test rows", f"{s['n_test']:,}")
        c[2].metric("Fraud rate (train)", f"{s['fraud_rate_train']*100:.3f}%")
        st.dataframe([{**{"bank": b}, **v} for b, v in s["banks"].items()])
    else:
        st.warning("Dataset not cached yet. The backend is downloading it (needs internet once).")
    st.write("Flow: (1) banks train locally, (2) only weight updates go to the aggregator, "
             "(3) FedAvg builds a global model, (4) evaluation on the held-out test split.")
