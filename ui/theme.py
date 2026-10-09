CSS = """
<style>
html,body,[class*="st-"],h1,h2,h3,p,div,span,label,button,input,select,table {
 font-family:'Times New Roman',Times,'Liberation Serif',Tinos,serif !important;}
.block-container{max-width:1200px;padding-top:1rem;}
section[data-testid="stSidebar"]{background-color:#F7F8FA;border-right:1px solid #CBD5E0;}
h1{border-bottom:2px solid #0B3D7A;padding-bottom:8px;font-size:28px;}
.stButton>button{border-radius:4px !important;background-color:#0B3D7A;color:white;border:1px solid #0B3D7A;}
div[data-testid="stMetric"]{border:1px solid #CBD5E0;border-radius:4px;background:#F7F8FA;padding:8px;}
table{border-collapse:collapse;}td,th{border-bottom:1px solid #CBD5E0 !important;text-align:right;}
</style>
"""
TEMPLATE = dict(layout=dict(paper_bgcolor="white", plot_bgcolor="white",
  font=dict(family="'Times New Roman',Times,serif", color="#1A202C"),
  xaxis=dict(gridcolor="#E2E8F0", title_standoff=10),
  yaxis=dict(gridcolor="#E2E8F0", title_standoff=10),
  legend=dict(orientation="h", y=-0.25)))
COLORS = {"fed": "#0B3D7A", "local": "#718096", "pooled": "#2F855A", "noise": "#B7791F"}
