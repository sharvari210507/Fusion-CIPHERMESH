"""Local accounts (offline) + optional Google OIDC via st.login. Roles backend-enforced."""
import re
import bcrypt
import sqlite3
import streamlit as st
from datetime import datetime, timezone, timedelta
from . import database as db

URE = re.compile(r"^[A-Za-z0-9._-]{3,32}$")


def _c():
    from .config import DB_PATH
    return sqlite3.connect(str(DB_PATH), timeout=30)


def valid_username(u):
    return bool(URE.match(u or ""))


def register(username, password):
    if not valid_username(username):
        return False, "Invalid username (3-32 chars: letters, digits, dot, underscore, hyphen)."
    if not password or len(password) < 10:
        return False, "Password must be at least 10 characters."
    h = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    try:
        c = _c()
        c.execute("INSERT INTO users(username,provider,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                  (username, "local", h, "analyst", db.now()))
        c.commit(); c.close()
    except sqlite3.IntegrityError:
        return False, "Invalid username or password"
    db.audit(username, "register", "")
    return True, "Account created. Please sign in."


def login_local(username, password):
    c = _c(); c.row_factory = sqlite3.Row
    r = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    c.close()
    generic = "Invalid username or password"
    if r is None:
        return False, generic
    if r["locked_until"]:
        try:
            if datetime.fromisoformat(r["locked_until"]) > datetime.now(timezone.utc):
                db.audit(username, "login_locked", "")
                return False, generic
        except Exception:
            pass
    ok = r["password_hash"] and bcrypt.checkpw(password.encode(), r["password_hash"].encode())
    c = _c()
    if not ok:
        fails = (r["failed_attempts"] or 0) + 1
        lock = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat() if fails >= 5 else None
        c.execute("UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?", (fails, lock, r["id"]))
        c.commit(); c.close()
        db.audit(username, "login_failure", f"attempt {fails}")
        if lock:
            db.audit(username, "lockout", "")
        return False, generic
    c.execute("UPDATE users SET failed_attempts=0, locked_until=NULL, last_login_at=? WHERE id=?",
              (db.now(), r["id"]))
    c.commit(); c.close()
    st.session_state["user"] = {"name": r["username"], "role": admin_checked_role(r["username"], r["role"])}
    db.audit(username, "login_success", "local")
    return True, "Signed in."


def admin_users():
    try:
        return st.secrets["app"]["admin_users"]
    except Exception:
        return []


def admin_checked_role(username, role):
    if username in admin_users():
        return "admin"
    return role or "analyst"


def current_user():
    if "user" in st.session_state:
        return st.session_state["user"]
    try:
        u = st.user
        if u and getattr(u, "is_logged_in", False):
            name = getattr(u, "email", None) or getattr(u, "name", "google-user")
            role = "admin" if name in admin_users() else "analyst"
            return {"name": name, "role": role, "google": True}
    except Exception:
        pass
    return None


def require_admin(user):
    if not user or user.get("role") != "admin":
        raise PermissionError("Admin role required.")


def google_configured():
    try:
        s = st.secrets["auth"]["google"]
        return bool(s.get("client_id"))
    except Exception:
        return False


def logout():
    st.session_state.pop("user", None)
    try:
        st.logout()
    except Exception:
        pass
