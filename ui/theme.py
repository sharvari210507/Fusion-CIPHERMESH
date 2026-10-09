"""CIPHERMESH design system (ui-ux-pro-max methodology, fintech/SOC dashboard).

Generated design-system summary (skill reasoning: product=Cybersecurity Platform /
Fintech fraud-ops dashboard, Streamlit stack):
  PATTERN : ops-dashboard (status bar + KPI row + charts + tables + actions)
  STYLE   : Dark Mode operations console, flat panels, 4px radius, no gradients/neon
  COLORS  : Primary #38BDF8 (sky) / Secondary #0E7490 (teal) / CTA #0891B2 /
            Background #0B1220-#16213A / Text #E8EEF6 / Muted #A9B7CC /
            Border #334155 / Success #4ADE80 / Warning #FBBF24 / Danger #F87171
  TYPE    : Times New Roman stack everywhere (hackathon spec mandate; the skill's
            modern-font recommendation is intentionally overridden here).
  EFFECTS : progress/spinner motion only; hover feedback on buttons/tabs;
            visible :focus-visible outlines; prefers-reduced-motion respected.
  AVOID   : purple/pink gradients, neon, glow/shadow, emoji icons, centered heroes,
            pill shapes, animations beyond loading indicators.
  CHARTS  : line for rounds/progress, bar for per-bank comparison; titled axes with
            units, horizontal legend, grid #334155, no 3D.

Contrast rules: every surface declares background AND foreground. Light text
(#E8EEF6, 4.5:1+ on dark surfaces); muted #A9B7CC for secondary text only.
"""

TOKENS = {
    "bg_app": "#0F172A", "bg_sidebar": "#0B1220", "bg_canvas": "#16213A",
    "bg_panel": "#1E293B", "bg_input": "#0B1220", "bg_header": "#1E3A5F",
    "text": "#E8EEF6", "text_bright": "#F1F5F9", "muted": "#A9B7CC",
    "faint": "#94A3B8", "border": "#334155", "border_strong": "#1E3A5F",
    "primary": "#38BDF8", "secondary": "#0E7490", "cta": "#0891B2",
    "accent": "#22D3EE", "accent2": "#7DD3FC", "accent3": "#5EEAD4",
    "success": "#4ADE80", "success_bg": "#14532D", "success_tx": "#BBF7D0",
    "warning": "#FBBF24", "warning_bg": "#78350F", "warning_tx": "#FDE68A",
    "danger": "#F87171", "danger_bg": "#7F1D1D", "danger_tx": "#FECACA",
    "info_bg": "#164E63", "info_tx": "#A5F3FC",
    "radius": "4px", "max_width": "1280px",
}
T = TOKENS

CSS = f"""
<style>
/* ---- type (spec-mandated stack) ---- */
html,body,[class*="st-"],h1,h2,h3,h4,p,div,span,label,button,input,select,table,
.stMarkdown,.stMetric,.stDataFrame,.stTable,.stCaption,.stAlert {{
  font-family:'Times New Roman',Times,'Liberation Serif',Tinos,serif !important;}}
/* ---- shell ---- */
.stApp{{background-color:{T['bg_app']};color:{T['text']};}}
.block-container{{max-width:{T['max_width']};padding:1rem 2rem 3rem;}}
header[data-testid="stHeader"]{{background-color:{T['bg_app']};}}
section[data-testid="stSidebar"]{{background-color:{T['bg_sidebar']};
  border-right:1px solid {T['border_strong']};}}
section[data-testid="stSidebar"] *{{color:{T['text']} !important;}}
section[data-testid="stSidebar"] .stCaption{{color:{T['faint']} !important;}}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2{{
  color:{T['text_bright']} !important;border-bottom:1px solid {T['border_strong']};
  padding-bottom:8px;}}
/* ---- ops canvas ---- */
section.main{{background-color:{T['bg_app']};}}
section.main .block-container{{background-color:{T['bg_canvas']};
  border:1px solid {T['border_strong']};border-radius:8px;color:{T['text']};}}
/* ---- hierarchy: eyebrow / title / subtitle ---- */
.fg-eyebrow{{color:{T['accent']};font-size:12px;font-weight:bold;
  letter-spacing:2px;text-transform:uppercase;margin-bottom:0;}}
h1,h2,h3,h4{{color:{T['text_bright']} !important;line-height:1.25;}}
h1{{border-bottom:3px solid {T['accent']};padding-bottom:8px;font-size:30px;
  margin-bottom:4px;}}
h2{{font-size:22px;color:{T['accent2']} !important;margin-top:24px;}}
h3{{font-size:18px;color:{T['accent3']} !important;margin-top:16px;}}
p,div,span,label{{color:{T['text']};}}
.stCaption,.fg-sub{{color:{T['muted']} !important;font-size:14px;max-width:90ch;}}
hr{{border-color:{T['border_strong']};margin:16px 0;}}
a{{color:{T['accent']} !important;}}
/* ---- KPI + status cards ---- */
.fg-card{{background:{T['bg_panel']};border:1px solid {T['border']};
  border-left:4px solid {T['accent']};border-radius:{T['radius']};
  padding:14px 16px;margin:8px 0;color:{T['text']};}}
.fg-card.navy{{border-left-color:{T['primary']};}}
.fg-card.green{{border-left-color:{T['success']};}}
.fg-card.amber{{border-left-color:{T['warning']};}}
.fg-card.red{{border-left-color:{T['danger']};}}
.fg-card h4{{margin:0 0 4px;color:{T['text_bright']};font-size:16px;}}
.fg-kpi{{font-size:28px;font-weight:bold;color:#FFFFFF;
  font-variant-numeric:tabular-nums;}}
.fg-badge{{display:inline-block;padding:2px 10px;border-radius:{T['radius']};
  font-size:13px;font-weight:bold;white-space:nowrap;}}
.fg-ok{{background:{T['success_bg']};color:{T['success_tx']};}}
.fg-warn{{background:{T['warning_bg']};color:{T['warning_tx']};}}
.fg-bad{{background:{T['danger_bg']};color:{T['danger_tx']};}}
.fg-info{{background:{T['info_bg']};color:{T['info_tx']};}}
/* ---- metrics: label / value / delta ---- */
div[data-testid="stMetric"]{{background:{T['bg_panel']};border:1px solid {T['border']};
  border-top:3px solid {T['accent']};border-radius:6px;padding:10px 12px;}}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] div{{
  color:{T['text']} !important;}}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{{
  font-variant-numeric:tabular-nums;}}
/* ---- ops status strip ---- */
.fg-opsbar{{background:{T['bg_sidebar']};border:1px solid {T['border_strong']};
  border-left:4px solid {T['secondary']};border-radius:{T['radius']};
  padding:8px 14px;margin:8px 0;font-size:14px;color:{T['text']};}}
/* ---- inputs: one consistent control system ---- */
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] div,
.stDateInput input,.stTextArea textarea,.stMultiSelect div[data-baseweb="select"] div{{
  background-color:{T['bg_input']} !important;color:{T['text_bright']} !important;
  border:1px solid {T['primary']} !important;border-radius:{T['radius']} !important;}}
div[data-baseweb="select"] svg{{fill:{T['accent2']} !important;}}
.stNumberInput button{{background-color:{T['bg_panel']} !important;}}
.stNumberInput button svg{{fill:{T['accent2']} !important;}}
.stSlider [data-baseweb="slider"] div{{background-color:{T['accent']} !important;}}
/* ---- buttons: primary / disabled / danger-secondary ---- */
.stButton>button{{border-radius:{T['radius']} !important;
  background-color:{T['secondary']} !important;color:#FFFFFF !important;
  border:1px solid {T['accent']} !important;font-weight:bold;cursor:pointer;}}
.stButton>button:disabled{{background-color:#334155 !important;
  border-color:#475569 !important;color:{T['faint']} !important;cursor:not-allowed;}}
.stButton>button:hover:not(:disabled){{background-color:{T['cta']} !important;}}
.stButton>button:active:not(:disabled){{background-color:#0E7490 !important;}}
/* ---- keyboard focus: visible on every interactive element ---- */
button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,
textarea:focus-visible,[tabindex]:focus-visible{{
  outline:2px solid {T['accent']} !important;outline-offset:2px;}}
/* ---- tables: dense ops readout ---- */
.stDataFrame,.stTable{{background-color:{T['bg_input']};}}
td,th{{color:{T['text']} !important;border-bottom:1px solid {T['border']} !important;
  font-variant-numeric:tabular-nums;}}
th{{background-color:{T['bg_header']} !important;color:#FFFFFF !important;
  font-size:13px;letter-spacing:0.5px;}}
tbody tr:hover td{{background-color:{T['bg_panel']} !important;}}
/* ---- tabs / expanders / options ---- */
.stTabs [data-baseweb="tab"]{{color:{T['muted']} !important;}}
.stTabs [data-baseweb="tab"]:hover{{color:{T['text_bright']} !important;}}
.stTabs [aria-selected="true"]{{color:{T['accent']} !important;
  border-bottom:2px solid {T['accent']} !important;}}
.stExpander{{background-color:{T['bg_panel']};border:1px solid {T['border']};
  border-radius:6px;}}
.stExpander summary,.stExpander p{{color:{T['text']} !important;}}
.stCheckbox label,.stRadio label{{color:{T['text']} !important;}}
.stAlert{{background-color:{T['bg_panel']} !important;color:{T['text']} !important;
  border-radius:{T['radius']};}}
.stProgress div div{{background-color:{T['accent']};}}
.stSpinner div{{border-top-color:{T['accent']} !important;}}
code{{color:{T['accent3']} !important;background-color:{T['bg_input']} !important;
  overflow-wrap:anywhere;}}
/* ---- motion: loading indicators only (skill UX rule) ---- */
@media (prefers-reduced-motion: reduce){{
  *,*::before,*::after{{animation-duration:0.01ms !important;
    transition-duration:0.01ms !important;}}
}}
/* ---- laptop-responsive: tighten gutters under 1100px ---- */
@media (max-width: 1100px){{
  .block-container{{padding:0.75rem 1rem 2rem;}}
  h1{{font-size:24px;}}
  .fg-kpi{{font-size:22px;}}
  .fg-sub{{max-width:100%;}}
}}
</style>
"""

TEMPLATE = dict(layout=dict(paper_bgcolor="#16213A", plot_bgcolor="#16213A",
  font=dict(family="'Times New Roman',Times,serif", color="#E8EEF6"),
  xaxis=dict(gridcolor="#334155", title_standoff=12, tickfont=dict(color="#E8EEF6")),
  yaxis=dict(gridcolor="#334155", title_standoff=12, tickfont=dict(color="#E8EEF6")),
  legend=dict(orientation="h", y=-0.25, font=dict(color="#E8EEF6")),
  margin=dict(l=56, r=24, t=48, b=64)))
COLORS = {"fed": "#38BDF8", "local": "#94A3B8", "pooled": "#4ADE80", "noise": "#FBBF24"}


def banner(title, subtitle, eyebrow="CIPHERMESH / FEDGUARD"):
  """Page header: eyebrow + title + one-line purpose. Static markup only."""
  return (f"<p class='fg-eyebrow'>{eyebrow}</p><h1>{title}</h1>"
          f"<p class='fg-sub'>{subtitle}</p>")


def card(title, value, sub="", tone="navy"):
  return (f"<div class='fg-card {tone}'><h4>{title}</h4>"
          f"<div class='fg-kpi'>{value}</div><div class='fg-sub'>{sub}</div></div>")


def badge(text, kind="info"):
  return f"<span class='fg-badge fg-{kind}'>{text}</span>"


def opsbar(text):
  """Thin ops status strip. Static markup only; caller supplies backend text."""
  return f"<div class='fg-opsbar'>{text}</div>"
