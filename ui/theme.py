"""Shared FedGuard/CIPHERMESH theme: dark navy SOC shell, light readable text.

Contrast rules: every surface declares both background AND foreground color.
Light text (#E8EEF6) on dark surfaces; dark text (#1A202C) only inside white cards.
"""

CSS = """
<style>
html,body,[class*="st-"],h1,h2,h3,h4,p,div,span,label,button,input,select,table,
.stMarkdown,.stMetric,.stDataFrame,.stTable,.stCaption,.stAlert {
  font-family:'Times New Roman',Times,'Liberation Serif',Tinos,serif !important;}
/* App shell: dark navy */
.stApp{background-color:#0F172A;color:#E8EEF6;}
.block-container{max-width:1280px;padding:1rem 2rem 3rem;}
header[data-testid="stHeader"]{background-color:#0F172A;}
section[data-testid="stSidebar"]{background-color:#0B1220;border-right:1px solid #1E3A5F;}
section[data-testid="stSidebar"] *{color:#E8EEF6 !important;}
section[data-testid="stSidebar"] .stCaption{color:#94A3B8 !important;}
/* Main content: slate canvas, light text */
section.main{background-color:#0F172A;}
section.main .block-container{background-color:#16213A;border:1px solid #1E3A5F;
  border-radius:8px;color:#E8EEF6;}
h1,h2,h3,h4{color:#F1F5F9 !important;}
h1{border-bottom:3px solid #22D3EE;padding-bottom:8px;font-size:30px;}
h2{font-size:22px;color:#7DD3FC !important;}
h3{font-size:18px;color:#5EEAD4 !important;}
p,div,span,label{color:#E8EEF6;}
.stCaption,.fg-sub{color:#A9B7CC !important;font-size:14px;}
hr{border-color:#1E3A5F;}
a{color:#22D3EE !important;}
/* Cards: lighter slate panels with light text */
.fg-card{background:#1E293B;border:1px solid #334155;border-left:4px solid #22D3EE;
  border-radius:6px;padding:14px 16px;margin:8px 0;color:#E8EEF6;}
.fg-card.navy{border-left-color:#38BDF8;}
.fg-card.green{border-left-color:#4ADE80;}
.fg-card.amber{border-left-color:#FBBF24;}
.fg-card.red{border-left-color:#F87171;}
.fg-card h4{margin:0 0 4px;color:#F1F5F9;font-size:16px;}
.fg-kpi{font-size:26px;font-weight:bold;color:#FFFFFF;}
.fg-badge{display:inline-block;padding:2px 10px;border-radius:4px;font-size:13px;font-weight:bold;}
.fg-ok{background:#14532D;color:#BBF7D0;}
.fg-warn{background:#78350F;color:#FDE68A;}
.fg-bad{background:#7F1D1D;color:#FECACA;}
.fg-info{background:#164E63;color:#A5F3FC;}
/* Metrics */
div[data-testid="stMetric"]{background:#1E293B;border:1px solid #334155;
  border-top:3px solid #22D3EE;border-radius:6px;padding:10px;}
div[data-testid="stMetric"] label,div[data-testid="stMetric"] div{color:#E8EEF6 !important;}
/* Inputs: dark fields, light text, visible borders */
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb="select"] div,
.stDateInput input,.stTextArea textarea{background-color:#0B1220 !important;
  color:#F1F5F9 !important;border:1px solid #38BDF8 !important;border-radius:4px !important;}
div[data-baseweb="select"] svg{fill:#7DD3FC !important;}
.stNumberInput button{background-color:#1E293B !important;}
.stNumberInput button svg{fill:#7DD3FC !important;}
/* Buttons */
.stButton>button{border-radius:4px !important;background-color:#0E7490 !important;
  color:#FFFFFF !important;border:1px solid #22D3EE !important;font-weight:bold;}
.stButton>button:disabled{background-color:#334155 !important;border-color:#475569 !important;
  color:#94A3B8 !important;}
.stButton>button:hover:not(:disabled){background-color:#0891B2 !important;}
/* Tables */
.stDataFrame,.stTable{background-color:#0B1220;}
td,th{color:#E8EEF6 !important;border-bottom:1px solid #334155 !important;}
th{background-color:#1E3A5F !important;color:#FFFFFF !important;}
/* Tabs, expanders, checkboxes, radio */
.stTabs [data-baseweb="tab"]{color:#A9B7CC !important;}
.stTabs [aria-selected="true"]{color:#22D3EE !important;}
.stExpander{background-color:#1E293B;border:1px solid #334155;border-radius:6px;}
.stExpander summary,.stExpander p{color:#E8EEF6 !important;}
.stCheckbox label,.stRadio label{color:#E8EEF6 !important;}
.stAlert{background-color:#1E293B !important;color:#E8EEF6 !important;}
.stProgress div div{background-color:#22D3EE;}
code{color:#5EEAD4 !important;background-color:#0B1220 !important;}
</style>
"""

TEMPLATE = dict(layout=dict(paper_bgcolor="#16213A", plot_bgcolor="#16213A",
  font=dict(family="'Times New Roman',Times,serif", color="#E8EEF6"),
  xaxis=dict(gridcolor="#334155", title_standoff=10, tickfont=dict(color="#E8EEF6")),
  yaxis=dict(gridcolor="#334155", title_standoff=10, tickfont=dict(color="#E8EEF6")),
  legend=dict(orientation="h", y=-0.25, font=dict(color="#E8EEF6"))))
COLORS = {"fed": "#38BDF8", "local": "#94A3B8", "pooled": "#4ADE80", "noise": "#FBBF24"}


def banner(title, subtitle):
  """Page header: title + one-line purpose. Static markup only."""
  return f"<h1>{title}</h1><p class='fg-sub'>{subtitle}</p>"


def card(title, value, sub="", tone="navy"):
  return (f"<div class='fg-card {tone}'><h4>{title}</h4>"
          f"<div class='fg-kpi'>{value}</div><div class='fg-sub'>{sub}</div></div>")


def badge(text, kind="info"):
  return f"<span class='fg-badge fg-{kind}'>{text}</span>"
