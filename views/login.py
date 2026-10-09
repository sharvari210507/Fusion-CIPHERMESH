import streamlit as st
from backend import auth as A
from ui.theme import banner


def _password_field(label, key):
    """Password input with a real, supported show/hide toggle.

    The toggle is a Streamlit checkbox that switches the input type; the
    plaintext value is never logged or persisted (it lives only in the
    widget state for the current run). The native eye button is hidden by
    theme CSS because its glyph may not render where icon assets are blocked.
    """
    show = st.checkbox("Show password", key=f"{key}_show")
    return st.text_input(label, type="default" if show else "password", key=key)


def page():
    st.markdown(banner("FedGuard",
        "Five simulated banks improve one shared fraud-detection model "
        "without sharing customer transactions. Only numeric model updates "
        "leave each bank."), unsafe_allow_html=True)
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        if A.google_configured():
            try:
                if st.button("Sign in with Google", use_container_width=True):
                    st.login("google")
            except Exception:
                st.caption("Google sign-in is not configured on this deployment.")
        else:
            st.caption("Google sign-in is not configured on this deployment. "
                       "Use a local account below.")
        t1, t2 = st.tabs(["Sign in", "Create account"])
        with t1:
            u = st.text_input("Username", key="li_u")
            p = _password_field("Password", key="li_p")
            if st.button("Sign in", use_container_width=True):
                ok, msg = A.login_local(u, p)
                if ok:
                    st.rerun()
                else:
                    st.error(msg)
        with t2:
            u = st.text_input("Username", key="rg_u",
                              help="3-32 characters: letters, digits, dot, underscore, hyphen.")
            p = _password_field("Password (minimum 10 characters)", key="rg_p")
            if st.button("Create account", use_container_width=True):
                ok, msg = A.register(u, p)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
