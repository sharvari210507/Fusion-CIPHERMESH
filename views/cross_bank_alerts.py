import streamlit as st
from views.overview import strip


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    from ui.theme import banner
    st.markdown(banner("Cross-Bank Alerts",
        "Real-time risk signals between banks. Token-only sharing; investigation warnings, never verdicts."),
        unsafe_allow_html=True)
    strip()
    from backend import alerts as A
    admin = user and user.get("role") == "admin"
    st.subheader("Demo replay: Bank 0 flags, Bank 1 warned two hours later")
    st.caption("Uses real test-split records linked by a shared recipient; the shared token is a "
               "salted HMAC pseudonym (raw identifiers never leave the bank). Simulated clock: "
               "no real two-hour wait. Detector: active federated model score vs validation threshold.")
    if st.button("Run two-hour replay", disabled=not admin):
        try:
            with st.spinner("Replaying events..."):
                trail = A.replay_two_hour_demo(user, delay_hours=2.0)
            st.session_state["replay"] = trail
        except Exception as e:
            st.error(f"Replay failed: {e}")
    if not admin:
        st.info("Replay control requires an admin account; results below are readable by analysts.")
    trail = st.session_state.get("replay")
    if trail:
        st.subheader("Event trail")
        st.json(trail)
        if trail.get("flagged") and trail.get("match_hits"):
            st.success(f"MATCH: Bank 1 warned {trail['simulated_elapsed_hours']}h (simulated) after "
                       f"Bank 0's event. Real processing latency: "
                       f"{trail['real_processing_latency_ms']} ms. {trail['decision']}")
        elif not trail.get("flagged"):
            st.warning("Detector did not flag the source event; no signal was published. "
                       "Honest negative — adjust the demo seed or model.")
    st.subheader("Active signals from other banks")
    try:
        rows = A.active_signals()
        if rows:
            st.dataframe([{"id": r["id"], "source_bank": r["source_bank"],
                           "category": r["category"], "risk": r["risk_score"],
                           "model": r["model_version"], "expires": r["expires_at"],
                           "status": r["status"]} for r in rows],
                         use_container_width=True)
        else:
            st.caption("No active signals. Run the replay (admin) to publish the demo signal.")
    except Exception as e:
        st.error(f"Alert service unavailable: {e}")
    st.caption("Trust assumptions: tokens are HMAC pseudonyms under a server-side salt; anyone "
               "knowing an identifier and the salt could recompute its token, so tokens expire, "
               "are never logged with raw identifiers, and carry no formal unlinkability claim.")
