"""Shared FedGuard theme: dark navy/slate SOC shell, Times New Roman, card styling."""

CSS = """
<style>
html,body,[class*="st-"],h1,h2,h3,h4,p,div,span,label,button,input,select,table,
.stMarkdown,.stMetric,.stDataFrame,.stTable,.stCaption {
  font-family:'Times New Roman',Times,'Liberation Serif',Tinos,serif !important;}
/* App shell */
.stApp{background-color:#0F172A;}
.block-container{max-width:1280px;padding:1rem 2rem 3rem;}
section[data-testid="stSidebar"]{background-color:#0B1220;border-right:1px solid #1E3A5F;}
section[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
section[data-testid="stSidebar"] .stButton>button{background-color:#0E7490;border:1px solid #0E7490;}
/* Main content canvas */
section.main > div{background-color:#F1F5F9;border-radius:6px;}
h1{color:#0B3D7A !important;border-bottom:3px solid #0E7490;padding-bottom:8px;font-size:30px;}
h2{color:#0B3D7A !important;font-size:22px;}
h3{color:#134E4A !important;font-size:18px;}
hr{border-color:#0E7490;}
/* Cards */
.fg-card{background:#FFFFFF;border:1px solid #CBD5E0;border-left:4px solid #0E7490;
  border-radius:4px;padding:14px 16px;margin:8px 0;}
.fg-card.navy{border-left-color:#0B3D7A;}
.fg-card.green{border-left-color:#2F855A;}
.fg-card.amber{border-left-color:#B7791F;}
.fg-card.red{border-left-color:#C53030;}
.fg-card h4{margin:0 0 4px;color:#0B3D7A;font-size:16px;}
.fg-kpi{font-size:26px;font-weight:bold;color:#0B3D7A;}
.fg-sub{color:#475569;font-size:14px;}
.fg-badge{display:inline-block;padding:2px 10px;border-radius:4px;font-size:13px;font-weight:bold;}
.fg-ok{background:#DCFCE7;color:#166534;}
.fg-warn{background:#FEF3C7;color:#92400E;}
.fg-bad{background:#FEE2E2;color:#991B1B;}
.fg-info{background:#CFFAFE;color:#155E75;}
/* Buttons / inputs */
.stButton>button{border-radius:4px !important;background-color:#0B3D7A !important;
  color:#FFFFFF !important;border:1px solid #0B3D7A !important;}
.stButton>button:disabled{background-color:#64748B !important;border-color:#64748B !important;}
div[data-testid="stMetric"]{background:#FFFFFF;border:1px solid #CBD5E0;
  border-top:3px solid #0E7490;border-radius:4px;padding:10px;}
table{border-collapse:collapse;}td,th{border-bottom:1px solid #CBD5E0 !important;}
.stAlert{border-radius:4px;}
</style>
"""

TEMPLATE = dict(layout=dict(paper_bgcolor="white", plot_bgcolor="white",
  font=dict(family="'Times New Roman',Times,serif", color="#1A202C"),
  xaxis=dict(gridcolor="#E2E8F0", title_standoff=10),
  yaxis=dict(gridcolor="#E2E8F0", title_standoff=10),
  legend=dict(orientation="h", y=-0.25)))
COLORS = {"fed": "#0B3D7A", "local": "#64748B", "pooled": "#2F855A", "noise": "#B7791F"}


def banner(title, subtitle):
  """Page header: title + one-line purpose + rule. Static markup only."""
  return f"<h1>{title}</h1><p class='fg-sub'>{subtitle}</p>"


def card(title, value, sub="", tone="navy"):
  return (f"<div class='fg-card {tone}'><h4>{title}</h4>"
          f"<div class='fg-kpi'>{value}</div><div class='fg-sub'>{sub}</div></div>")


def badge(text, kind="info"):
  return f"<span class='fg-badge fg-{kind}'>{text}</span>"
