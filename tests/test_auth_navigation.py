"""Regression: protected navigation must not persist after logout.

Uses AppTest against the real app.py entry point and only Streamlit's
stable contracts (page titles, sidebar text, widget labels) — never
generated class names.
"""
import pytest

at = pytest.importorskip("streamlit.testing.v1").AppTest

PROTECTED = {"Overview", "Data Explorer", "Control Room", "Experiments",
             "Privacy Audit", "Threat Model", "Score Transaction",
             "Cross-Bank Alerts"}


def _fresh_app():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = at.from_file(str(root / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    return app


def _page_names(app):
    return {v.get("page_name") for v in app._registered_pages.values()}


def _sidebar_text(app):
    parts = [str(m.value) for m in app.sidebar.markdown]
    parts += [str(c.value) for c in app.sidebar.caption]
    return " ".join(parts)


def test_signed_out_sidebar_has_no_protected_navigation():
    app = _fresh_app()
    names = _page_names(app)
    assert "Sign in" in names
    assert not (names & PROTECTED), f"signed-out nav leaks: {names & PROTECTED}"
    texts = _sidebar_text(app)
    assert "FedGuard" in texts
    for forbidden in tuple(PROTECTED) + ("Administration", "Signed in:"):
        assert forbidden not in texts, f"signed-out sidebar leaks: {forbidden}"
    assert app.tabs, "expected Sign in / Create account tabs"


def test_sign_in_restores_permitted_navigation():
    app = _fresh_app()
    app.session_state["user"] = {"name": "analyst1", "role": "analyst"}
    app.session_state["_u"] = {"name": "analyst1", "role": "analyst"}
    app.run()
    assert not app.exception
    names = _page_names(app)
    assert PROTECTED <= names, f"missing pages: {PROTECTED - names}"
    assert "Administration" not in names, "analyst must not see admin page"
    assert "Signed in: analyst1 (analyst)" in _sidebar_text(app)


def test_admin_restores_admin_page():
    app = _fresh_app()
    app.session_state["user"] = {"name": "boss", "role": "admin"}
    app.session_state["_u"] = {"name": "boss", "role": "admin"}
    app.run()
    assert not app.exception
    names = _page_names(app)
    assert "Administration" in names
    assert PROTECTED <= names


def test_sign_out_removes_protected_navigation():
    app = _fresh_app()
    app.session_state["user"] = {"name": "analyst1", "role": "analyst"}
    app.session_state["_u"] = {"name": "analyst1", "role": "analyst"}
    app.run()
    assert PROTECTED <= _page_names(app)
    clicked = False
    for b in app.button:
        if str(b.label) == "Sign out":
            b.click().run()
            clicked = True
            break
    assert clicked, "Sign out button not found"
    assert not app.exception
    names = _page_names(app)
    assert names == {"Sign in"}, f"nav not cleared after logout: {names}"
    texts = _sidebar_text(app)
    assert "FedGuard" in texts
    for forbidden in tuple(PROTECTED) + ("Administration", "Signed in:"):
        assert forbidden not in texts, f"post-logout sidebar leaks: {forbidden}"
    assert app.tabs, "login tabs must return after logout"


def test_unauthenticated_cannot_access_protected_page_directly():
    app = _fresh_app()
    names = _page_names(app)
    assert not (names & PROTECTED)
    # Protected content is never dispatched when signed out: the default
    # signed-in landing page (Overview metrics) must be absent and the
    # login form must be shown instead.
    assert not app.metric, "protected metrics must not render when signed out"
    assert app.tabs, "login form must render instead of protected content"
    from backend.auth import get_page_user
    import streamlit as st
    st.session_state.pop("_u", None)
    assert get_page_user() is None
