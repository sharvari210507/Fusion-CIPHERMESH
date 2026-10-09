PAGE_PATHS = ["overview", "data-explorer", "control-room", "experiments",
               "privacy-audit", "threat-model", "score-transaction", "cross-bank-alerts",
               "administration"]


def test_nav_url_paths_unique():
    """Regression: every st.navigation page must have a unique stable url_path."""
    assert len(PAGE_PATHS) == len(set(PAGE_PATHS)), "duplicate page url_path"
    assert "page" not in PAGE_PATHS


def test_nav_pages_match_views():
    import pathlib
    views = pathlib.Path(__file__).resolve().parents[1] / "views"
    modules = {"overview", "data_explorer", "control_room", "experiments",
               "privacy_audit", "score_transaction", "threat_model", "admin",
               "cross_bank_alerts"}
    existing = {p.stem for p in views.glob("*.py") if p.stem != "__init__"}
    assert modules <= existing, f"missing view modules: {modules - existing}"
    app = (pathlib.Path(__file__).resolve().parents[1] / "app.py").read_text()
    for slug in PAGE_PATHS:
        assert slug in app, f"url_path {slug} not registered in app.py"
