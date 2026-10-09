"""UI regression tests: signed-out shell renders login with brand-only sidebar.

Slow (boots the real backend once); run selectively with `-k login_ui`.
"""
import pytest

at = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def signed_out():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = at.from_file(str(root / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, f"app raised: {app.exception}"
    return app


def test_login_renders_no_exception(signed_out):
    assert not signed_out.exception


def test_login_heading_and_form(signed_out):
    heads = " ".join(str(m.value) for m in signed_out.markdown)
    assert "FedGuard" in heads
    assert signed_out.tabs, "expected Sign in / Create account tabs"
    assert any(str(b.label) == "Sign in" for b in signed_out.button), \
        "expected a Sign in button"


def test_password_show_toggle_present(signed_out):
    labels = [str(c.label) for c in signed_out.checkbox]
    assert any("Show password" in lb for lb in labels), \
        "expected a supported show/hide password toggle"


def test_sidebar_brand_only_when_signed_out(signed_out):
    texts = " ".join(
        [str(m.value) for m in signed_out.sidebar.markdown]
        + [str(c.value) for c in signed_out.sidebar.caption])
    assert "FedGuard" in texts
    for forbidden in ("Overview", "Control Room", "Score Transaction",
                      "Administration", "Signed in:"):
        assert forbidden not in texts, f"signed-out sidebar leaks: {forbidden}"
