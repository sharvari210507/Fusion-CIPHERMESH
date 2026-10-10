"""Focused regression: sidebar collapse icon must keep its icon font.

Uses only Streamlit's stable `data-testid` contract, never generated class names.
"""
from ui.theme import CSS


def test_icon_spans_keep_material_font():
    assert "stIconMaterial" in CSS
    assert "Material Symbols Rounded" in CSS


def test_no_pseudo_overlay_on_chrome_buttons():
    for testid in ("stExpandSidebarButton", "stSidebarCollapseButton"):
        assert f'[data-testid="{testid}"]::after' not in CSS, \
            f"pseudo-element overlay on {testid}"


def test_collapse_button_not_hidden_or_disabled():
    assert "stSidebarCollapseButton" not in CSS or "display:none" not in CSS.split(
        "stSidebarCollapseButton")[1][:120]
