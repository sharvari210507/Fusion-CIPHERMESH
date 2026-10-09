import streamlit as st
from views.overview import strip

BODY = """
Assets: each bank's raw transactions, model updates, user accounts.
Adversaries considered: an honest-but-curious aggregator; a malicious participating bank
sending poisoned updates; an external attacker against the web app.
Implemented protections: data stays in each bank's client; only numeric updates are exchanged;
optional update clipping and Gaussian noise; role-based access; password hashing and lockout;
input validation; parameterized queries; audit log.
Not implemented (future work): secure aggregation, TLS between real bank servers,
formal differential-privacy accounting, robust aggregation against poisoned updates.
Limitations: synthetic PaySim data; all banks simulated on one machine; label leakage in the
balance columns; plain FedAvg improves a shared model but does not by itself link one fraud
ring across banks (that would need private set intersection or similar); model updates can
leak information without secure aggregation.
"""


def page(user=None):
    import streamlit as st
    user = user if user is not None else st.session_state.get('_u')
    st.title("Threat Model")
    st.write("Purpose: state what is and is not protected.")
    strip()
    st.write(BODY)
