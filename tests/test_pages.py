"""Phase 1 regression: every registered view page must be callable with zero args
(Streamlit invokes page callables without arguments) and must obtain the user
via backend.auth.get_page_user, never via a required parameter."""

import inspect

import views.overview as overview
import views.data_explorer as data_explorer
import views.control_room as control_room
import views.experiments as experiments
import views.privacy_audit as privacy_audit
import views.score_transaction as score_transaction
import views.threat_model as threat_model
import views.admin as admin
import views.cross_bank_alerts as cross_bank_alerts

PAGES = [overview, data_explorer, control_room, experiments,
         privacy_audit, score_transaction, threat_model, admin, cross_bank_alerts]


def test_all_pages_zero_arg_callable():
    for m in PAGES:
        sig = inspect.signature(m.page)
        missing = [p for p in sig.parameters.values()
                   if p.default is inspect.Parameter.empty
                   and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                  inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        assert not missing, f"{m.__name__}.page requires args: {missing}"


def test_pages_use_session_user_not_params():
    import pathlib
    for m in PAGES:
        if m.__name__ in ("views.overview", "views.login"):
            continue
        src = pathlib.Path(m.__file__).read_text()
        assert "get_page_user" in src, f"{m.__name__} does not use get_page_user"


def test_get_page_user_returns_none_unauthenticated():
    from backend.auth import get_page_user
    import streamlit as st
    st.session_state.pop("_u", None)
    # outside a logged-in session this must be None, never a fabricated user
    assert get_page_user() is None
