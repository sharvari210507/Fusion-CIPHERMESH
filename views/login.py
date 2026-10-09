import streamlit as st
from backend import auth as A


def page():
    st.title("FedGuard")
    st.write("Five simulated banks improve one shared fraud-detection model without sharing customer transactions. Only numeric model updates leave each bank.")
    if A.google_configured():
        try:
            if st.button("Sign in with Google"):
                st.login("google")
        except Exception as e:
            st.info("Google sign-in is not configured on this deployment.")
    else:
        st.info("Google sign-in is not configured on this deployment. Use a local account below.")
    t1, t2 = st.tabs(["Sign in", "Create account"])
    with t1:
        u = st.text_input("Username", key="li_u")
        p = st.text_input("Password", type="password", key="li_p")
        if st.button("Sign in"):
            ok, msg = A.login_local(u, p)
            st.write(msg)
            if ok:
                st.rerun()
    with t2:
        u = st.text_input("Username", key="rg_u")
        p = st.text_input("Password", type="password", key="rg_p")
        if st.button("Create account"):
            ok, msg = A.register(u, p)
            st.write(msg)
