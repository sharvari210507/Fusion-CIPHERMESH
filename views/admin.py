import streamlit as st
from backend import database as db, data as D
from backend.auth import require_admin
from views.overview import strip


def page():
    from backend.auth import get_page_user
    user = get_page_user()
    try:
        require_admin(user)
    except PermissionError:
        st.error("Admin role required."); return
    from ui.theme import banner
    st.markdown(banner("Administration", "Users, roles, audit records and job history. Admin role required."), unsafe_allow_html=True)
    strip()
    st.subheader("Users")
    st.dataframe(db.query("SELECT id,username,provider,role,failed_attempts,locked_until FROM users"))
    u = st.text_input("Username to modify")
    r = st.selectbox("New role", ["analyst", "admin"])
    if st.button("Set role"):
        import sqlite3
        from backend.config import DB_PATH
        c = sqlite3.connect(str(DB_PATH))
        c.execute("UPDATE users SET role=? WHERE username=?", (r, u))
        c.commit(); c.close()
        db.audit(user["name"], "role_change", f"{u}->{r}")
        st.write("Updated.")
    if st.button("Unlock"):
        import sqlite3
        from backend.config import DB_PATH
        c = sqlite3.connect(str(DB_PATH))
        c.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE username=?", (u,))
        c.commit(); c.close()
        st.write("Unlocked.")
    st.subheader("Audit log")
    st.dataframe(db.query("SELECT * FROM audit_log ORDER BY id DESC", (), 200))
    st.subheader("Jobs")
    st.dataframe(db.list_jobs(50))
    st.subheader("Dataset cache")
    st.write({"cached": D.is_cached(), "stats": D.get_stats().get("n_train", "missing")})
