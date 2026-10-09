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

IMPACT = """
Why this matters (external context, cited figures, not computed by this app): Indian banks and
financial institutions reported Rs 48,021 crore fraud across 10,114 cases in FY26, up 46% in
value versus FY25 while case counts fell 57% (RBI Annual Report 2025-26, via PTI/Outlook
Business; FY26 includes Rs 30,199 crore of reclassified earlier-year cases, mostly loan fraud,
while CIPHERMESH targets transaction-level fraud). One 2026 study reported federated F1 0.90
vs local-only 0.64 with centralized at 0.925 (arXiv 2603.13617); another found 35 percent
data-reconstruction success from plain FedAvg updates (Plymouth FedFraud study), so privacy
must be inspected, not assumed.
"""


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    from ui.theme import banner
    st.markdown(banner("Threat Model", "What is and is not protected."), unsafe_allow_html=True)
    st.write("Purpose: state what is and is not protected.")
    strip()
    st.write(BODY)
    st.subheader("Impact context")
    st.write(IMPACT)
