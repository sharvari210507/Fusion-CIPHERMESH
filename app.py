"""CIPHERMESH — Federated Fraud Intelligence Network.

Member banks collaboratively train a shared fraud detector via Federated
Averaging. Raw transaction records never leave a member's partition; only
validated model parameters are exchanged.

Run:  streamlit run app.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from src.data_loader import SCHEMA_INFO
from src.pipeline import run_experiment
from src.risk_scoring import score_transaction

# ---------------------------------------------------------------- theme ---
st.set_page_config(page_title="CIPHERMESH | Federated Fraud Intelligence",
                   page_icon="◈", layout="wide")

TEAL, NAVY, SLATE, AMBER, RED, BG = (
    "#0E9384", "#0B1F3A", "#475467", "#B54708", "#B42318", "#F6F8FA")
PALETTE = ["#0E9384", "#0B1F3A", "#4E80FF", "#B54708", "#7A5AF8"]
CHART_TEMPLATE = "plotly_white"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
  .block-container {{ padding-top: 1.2rem; max-width: 1400px; }}
  /* top brand bar */
  .brand-bar {{ background: linear-gradient(90deg, {NAVY} 0%, #12325e 100%);
    border-radius: 14px; padding: 22px 28px; color: white; margin-bottom: 18px; }}
  .brand-bar h1 {{ margin: 0; font-size: 1.9rem; letter-spacing: .12em; font-weight: 800; }}
  .brand-bar p {{ margin: 4px 0 0 0; opacity: .82; font-size: .95rem; }}
  .badge {{ display: inline-block; font-size: .68rem; font-weight: 700; letter-spacing: .08em;
    padding: 3px 10px; border-radius: 999px; margin-right: 6px; vertical-align: middle; }}
  .b-live {{ background: #067647; color: #D1FADF; }}
  .b-proto {{ background: rgba(255,255,255,.16); color: #fff; border: 1px solid rgba(255,255,255,.35); }}
  /* KPI cards */
  .kpi {{ background: #fff; border: 1px solid #E4E7EC; border-radius: 12px;
    padding: 14px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06); }}
  .kpi .lbl {{ font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: {SLATE};
    text-transform: uppercase; }}
  .kpi .val {{ font-size: 1.65rem; font-weight: 800; color: {NAVY}; margin: 2px 0; }}
  .kpi .sub {{ font-size: .8rem; color: {SLATE}; }}
  /* section titles */
  .sec-title {{ font-size: 1.25rem; font-weight: 800; color: {NAVY}; margin: 6px 0 2px 0; }}
  .sec-sub {{ color: {SLATE}; font-size: .9rem; margin-bottom: 12px; }}
  .panel {{ background: #fff; border: 1px solid #E4E7EC; border-radius: 12px;
    padding: 18px 20px; box-shadow: 0 1px 3px rgba(16,24,40,.06); margin-bottom: 14px; }}
  .pill {{ display:inline-block; font-size:.72rem; font-weight:700; border-radius:999px;
    padding:2px 10px; }}
  .pill-up {{ background:#D1FADF; color:#067647; }} .pill-dn {{ background:#FEE4E2; color:#B42318; }}
  .pill-flat {{ background:#E4E7EC; color:#344054; }}
  /* sidebar */
  section[data-testid="stSidebar"] {{ background: {NAVY}; }}
  section[data-testid="stSidebar"] * {{ color: #E4E7EC !important; }}
  section[data-testid="stSidebar"] .stRadio label {{ font-weight: 600; }}
  footer {{ visibility: hidden; }}
</style>""", unsafe_allow_html=True)

BANK_LABEL = {0: "Bank A", 1: "Bank B", 2: "Bank C", 3: "Bank D", 4: "Bank E"}
BANK_COLOR = {0: PALETTE[0], 1: PALETTE[1], 2: PALETTE[2], 3: PALETTE[3], 4: PALETTE[4]}

PAGES = ["Overview", "Data & Member Banks", "Training Lab", "Divergence Radar",
         "Privacy & Data Flow", "Trust Ledger", "Risk Scoring"]

# ---------------------------------------------------------------- state ---
if "result" not in st.session_state:
    st.session_state.result, st.session_state.coord, st.session_state.pre = None, None, None


def _fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def _load_saved():
    files = sorted(glob.glob(os.path.join(C.RESULTS_DIR, "experiment_*.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _ensure():
    if st.session_state.result is None:
        st.session_state.result = _load_saved()
    return st.session_state.result


def kpi_row(cards):
    cols = st.columns(len(cards))
    for col, (lbl, val, sub) in zip(cols, cards):
        col.markdown(f'<div class="kpi"><div class="lbl">{lbl}</div>'
                     f'<div class="val">{val}</div><div class="sub">{sub}</div></div>',
                     unsafe_allow_html=True)


def styled(fig, title):
    fig.update_layout(template=CHART_TEMPLATE, title=dict(text=title, font=dict(size=15, color=NAVY)),
                      font=dict(family="Inter", color="#344054"),
                      margin=dict(l=40, r=20, t=60, b=40), height=420,
                      legend=dict(orientation="h", y=-0.18))
    return fig


# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown("### ◈ CIPHERMESH")
    st.caption("Federated Fraud Intelligence Network")
    page = st.radio("Console", PAGES)
    st.divider()
    st.markdown("**Experiment configuration**")
    val_subset = st.number_input("Sample size (HF subset)", 2000, 200000, 30000, 1000)
    val_rounds = st.slider("Federated rounds", 1, 10, C.N_ROUNDS_DEFAULT)
    val_iters = st.slider("Local epochs / round", 5, 200, C.LOCAL_MAX_ITER)
    c1, c2 = st.columns(2)
    with c1:
        val_clip = st.selectbox("Clip norm", [None, 0.5, 1.0, 2.0], index=2)
    with c2:
        val_noise = st.number_input("Noise σ", 0.0, 1.0, 0.0, 0.05)
    val_banks = st.multiselect("Member banks", [0, 1, 2, 3, 4], default=[0, 1, 2, 3, 4],
                               format_func=lambda i: BANK_LABEL[i])
    val_thr = st.slider("Review threshold", 0.10, 0.90, 0.50, 0.05)
    use_synth = st.checkbox("Offline mode (synthetic data)", False)
    st.caption("Prototype console — member nodes are simulated on this host.")

# -------------------------------------------------------------- brand bar ---
st.markdown(f"""<div class="brand-bar">
  <span class="badge b-live">● NETWORK ACTIVE</span>
  <span class="badge b-proto">PROTOTYPE · SIMULATED NODES</span>
  <h1>CIPHERMESH</h1>
  <p>Privacy-preserving fraud signal sharing across member banks · Federated Averaging (FedAvg) ·
  5 members · model <b>SGD log-loss</b></p></div>""", unsafe_allow_html=True)

# ============================================================== 1 OVERVIEW ==
if page == "Overview":
    st.markdown('<div class="sec-title">Network overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Live state of the federation, shared model and latest evaluation.</div>',
                unsafe_allow_html=True)
    r = _ensure()
    if r and r.get("history"):
        h, last = r["history"], r["history"][-1]
        f1s = [(last["per_bank"][b] or {}).get("f1") for b in last["per_bank"]]
        prs = [(last["per_bank"][b] or {}).get("pr_auc") for b in last["per_bank"]]
        f1s = [x for x in f1s if x is not None] or [0]
        prs = [x for x in prs if x is not None] or [0]
        kpi_row([("Mean F1 · global model", f"{np.mean(f1s):.3f}", f"across {len(last['banks'])} members · held-out test"),
                 ("Mean PR-AUC · global", f"{np.mean(prs):.3f}", "rare-class metric · fraud ≈0.1%"),
                 ("Model updates aggregated", str(r["audit"]["updates_submitted"]),
                  f"{r['audit']['updates_accepted']} accepted · {r['audit']['updates_rejected']} rejected"),
                 ("Raw records shared", str(r["audit"]["raw_rows_sent"]),
                  "parameters only cross the boundary")])
        st.markdown('<div class="panel"><b>How the network operates</b><br>'
                    '<span style="color:#475467">Each member trains the shared architecture on its private '
                    'ledger partition → submits clipped parameters → coordinator validates and applies '
                    'sample-weighted FedAvg → redistributes the global model. Cycle repeats per round.</span>'
                    '</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            rounds = [x["round"] for x in h]
            fig = go.Figure()
            for b in last["banks"]:
                fig.add_trace(go.Scatter(
                    x=rounds, y=[(x["per_bank"][str(b)] or {}).get("f1") for x in h],
                    mode="lines+markers", name=BANK_LABEL[b],
                    line=dict(color=BANK_COLOR[b], width=2.5)))
            st.plotly_chart(styled(fig, "Global-model F1 per member, by round"), width='stretch')
        with c2:
            w = last.get("weights", [])
            if w:
                wf = pd.DataFrame([{"Member": BANK_LABEL[last["banks"][x["idx"]]],
                                    "Aggregation weight": x["weight"],
                                    "Training rows": x["n"]} for x in w])
                fig2 = px.bar(wf, x="Member", y="Aggregation weight",
                              color="Member", color_discrete_map={BANK_LABEL[b]: BANK_COLOR[b] for b in last["banks"]},
                              hover_data=["Training rows"])
                st.plotly_chart(styled(fig2, f"FedAvg aggregation weights · round {last['round']}"),
                                width='stretch')
            else:
                st.info("No aggregation data in this run.")
    else:
        kpi_row([("Member banks", "5", "Bank A – Bank E · simulated nodes"),
                 ("Dataset", "PaySim · HF", C.DATASET_ID),
                 ("Shared model", "Not trained", "open Training Lab to run"),
                 ("Raw records shared", "0", "by design")])
        st.warning("No trained federation found. Open **Training Lab** and start a run.")

# ==================================================== 2 DATA & BANKS =======
elif page == "Data & Member Banks":
    st.markdown('<div class="sec-title">Data & member banks</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sec-sub">{SCHEMA_INFO}</div>', unsafe_allow_html=True)
    r = _ensure()
    if not r:
        st.warning("No experiment data. Run training first.")
    else:
        cc = r["class_counts"]
        df = pd.DataFrame([{"Member": BANK_LABEL[b], "Partition": s.title(),
                             "Transactions": cc[f"bank_{b}_{s}"]["n"],
                             "Fraud cases": cc[f"bank_{b}_{s}"]["fraud"]}
                           for b in r["config"]["banks"] for s in ("train", "test")])
        kpi_row([("Transactions (sample)", f"{df['Transactions'].sum():,}", f"seed {r['config']['seed']} · stratified"),
                 ("Fraud cases (sample)", str(int(df["Fraud cases"].sum())),
                  f"{df['Fraud cases'].sum()/df['Transactions'].sum():.3%} blended rate"),
                 ("Hold-out test rows", f"{int(df[df.Partition=='Test']['Transactions'].sum()):,}",
                  "never used in training or aggregation"),
                 ("Feature vector", f"{len(r['config']['features'])} dims",
                  "target + IDs excluded from inputs")])
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(styled(px.bar(df, x="Member", y="Transactions", color="Partition",
                                          barmode="group", color_discrete_sequence=[NAVY, TEAL]),
                                   "Ledger volume per member"), width='stretch')
        with c2:
            st.plotly_chart(styled(px.bar(df, x="Member", y="Fraud cases", color="Partition",
                                          barmode="group", color_discrete_sequence=[AMBER, RED]),
                                   "Fraud cases per member (rare class)"), width='stretch')
        st.markdown('<div class="panel"><b>Partition ledger</b></div>', unsafe_allow_html=True)
        st.dataframe(df, width='stretch', hide_index=True)
        warns = [w for ws in r["warnings"].values() for w in ws]
        if warns:
            for w in warns:
                st.warning(w)
        with st.expander("Sampling, features & leakage controls"):
            st.json({"provenance": r["config"]["dataset"], "seed": r["config"]["seed"],
                     "model_inputs": r["config"]["features"],
                     "excluded": ["isFraud (target)", "BankID (routing key)",
                                  "nameOrig/nameDest (identifiers)"],
                     "scaling": "StandardScaler fit on TRAIN partitions only",
                     "note": "isFlaggedFraud is a near-constant legacy rule output; retained but untrusted."})

# ====================================================== 3 TRAINING LAB =====
elif page == "Training Lab":
    st.markdown('<div class="sec-title">Training Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Configure and execute a federated training program across member nodes.</div>',
                unsafe_allow_html=True)
    st.markdown(f"""<div class="panel"><b>Program</b> &nbsp;·&nbsp; members:
      <b>{', '.join(BANK_LABEL[b] for b in val_banks) or '—'}</b> &nbsp;·&nbsp;
      rounds: <b>{val_rounds}</b> &nbsp;·&nbsp; local epochs: <b>{val_iters}</b> &nbsp;·&nbsp;
      clip: <b>{val_clip}</b> &nbsp;·&nbsp; noise σ: <b>{val_noise}</b> &nbsp;·&nbsp;
      sample: <b>{val_subset:,}</b></div>""", unsafe_allow_html=True)
    run = st.button("Start federated training", type="primary")
    if run:
        if not val_banks:
            st.error("Select at least one member bank.")
        else:
            prog = st.progress(0, text="Initialising federation…")
            try:
                with st.spinner("Sampling ledgers → local training → FedAvg aggregation…"):
                    prog.progress(25, text="Loading member partitions…")
                    out, coord, pre = run_experiment(
                        subset_total=int(val_subset), n_rounds=int(val_rounds),
                        local_iters=int(val_iters), bank_ids=sorted(val_banks),
                        clip_norm=val_clip, noise_sigma=float(val_noise),
                        threshold=float(val_thr), force_synthetic=use_synth, save=True)
                    prog.progress(100, text="Program complete.")
                st.session_state.result, st.session_state.coord, st.session_state.pre = out, coord, pre
                st.success(f"Program complete — {len(out['history'])} rounds · "
                           f"{out['audit']['updates_accepted']}/{out['audit']['updates_submitted']} updates accepted · "
                           f"raw records shared: {out['audit']['raw_rows_sent']}")
            except Exception as e:
                st.error(f"Training program failed: {e}")
    r = _ensure()
    if r and r.get("history"):
        h = r["history"]
        rounds = [x["round"] for x in h]
        fig = go.Figure()
        for b in h[0]["banks"]:
            fig.add_trace(go.Scatter(
                x=rounds, y=[(x["per_bank"][str(b)] or {}).get("f1") for x in h],
                mode="lines+markers", name=f"{BANK_LABEL[b]} · F1",
                line=dict(color=BANK_COLOR[b], width=2.5)))
            fig.add_trace(go.Scatter(
                x=rounds, y=[(x["per_bank"][str(b)] or {}).get("pr_auc") for x in h],
                mode="lines", name=f"{BANK_LABEL[b]} · PR-AUC",
                line=dict(color=BANK_COLOR[b], width=1, dash="dot")))
        st.plotly_chart(styled(fig, "Convergence — F1 (solid) and PR-AUC (dotted) on held-out partitions"),
                        width='stretch')

# ================================================= 4 DIVERGENCE RADAR ======
elif page == "Divergence Radar":
    st.markdown('<div class="sec-title">Divergence Radar</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Local-only model vs federated global model, measured independently on each '
                "member's held-out partition. Scores are observed outcomes — regressions are reported, never hidden. "
                "Elevated score ⇒ <b>flag for review</b>, not confirmed fraud.</div>", unsafe_allow_html=True)
    r = _ensure()
    if not r:
        st.warning("No evaluation available. Run training first.")
    else:
        div = r["divergence"]

        def pill(d):
            if d is None:
                return '<span class="pill pill-flat">n/a</span>'
            cls = "pill-up" if d > 0.005 else ("pill-dn" if d < -0.005 else "pill-flat")
            return f'<span class="pill {cls}">{"+" if d > 0 else ""}{d:+.3f}</span>'

        cards = "".join(
            f"""<div class="kpi" style="margin-bottom:10px"><div class="lbl">{BANK_LABEL[int(b)]}</div>
            <div style="display:flex;gap:18px;margin-top:6px;flex-wrap:wrap">
            <div><span class="sub">Local F1</span><br><b>{_fmt(div[b]['local']['f1'])}</b></div>
            <div><span class="sub">Federated F1</span><br><b>{_fmt((div[b]['federated'] or {}).get('f1'))}</b></div>
            <div><span class="sub">Δ F1</span><br>{pill(div[b]['delta_f1'])}</div>
            <div><span class="sub">Δ PR-AUC</span><br>{pill(div[b]['delta_pr_auc'])}</div>
            <div><span class="sub">Fraud in test</span><br><b>{div[b]['local']['n_fraud']}</b></div>
            </div></div>""" for b in div)
        st.markdown(cards, unsafe_allow_html=True)
        for b in div:
            if (div[b]["local"]["n_fraud"] or 0) < 5:
                st.warning(f"{BANK_LABEL[int(b)]}: only {div[b]['local']['n_fraud']} fraud cases in test — "
                           "treat this member's metrics as indicative, not conclusive.")
        m = pd.DataFrame(
            [{"Member": BANK_LABEL[int(b)], "Model": "Local-only",
              "F1": div[b]["local"]["f1"] or 0, "PR-AUC": div[b]["local"]["pr_auc"] or 0,
              "Recall": div[b]["local"]["recall"] or 0} for b in div] +
            [{"Member": BANK_LABEL[int(b)], "Model": "Federated",
              "F1": (div[b]["federated"] or {}).get("f1") or 0,
              "PR-AUC": (div[b]["federated"] or {}).get("pr_auc") or 0,
              "Recall": (div[b]["federated"] or {}).get("recall") or 0} for b in div])
        c1, c2, c3 = st.columns(3)
        for col, metric in zip((c1, c2, c3), ("F1", "PR-AUC", "Recall")):
            with col:
                st.plotly_chart(styled(px.bar(m, x="Member", y=metric, color="Model", barmode="group",
                                              color_discrete_sequence=[SLATE, TEAL]), metric),
                                width='stretch')
        sel = st.selectbox("Confusion matrix — federated global model",
                           sorted(div, key=int), format_func=lambda b: BANK_LABEL[int(b)])
        cm = (div[str(sel)]["federated"] or {}).get("confusion_matrix")
        if cm:
            st.plotly_chart(styled(px.imshow(
                cm, text_auto=True, x=["Predicted legitimate", "Predicted fraud"],
                y=["Actual legitimate", "Actual fraud"],
                color_continuous_scale=["#E6FFF9", "#0E9384"],
                title=f"{BANK_LABEL[int(sel)]} — held-out confusion matrix"),
                ""), width='stretch')

# ================================================= 5 PRIVACY ===============
elif page == "Privacy & Data Flow":
    st.markdown('<div class="sec-title">Privacy & data flow</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Audited record of everything that crossed the member → coordinator boundary '
                "during the latest program.</div>", unsafe_allow_html=True)
    r = _ensure()
    a = r["audit"] if r else {"raw_rows_sent": 0, "updates_submitted": 0, "updates_accepted": 0,
                              "updates_rejected": 0, "rounds_completed": 0, "participants": [],
                              "last_training_time": None}
    kpi_row([("Raw records to coordinator", str(a["raw_rows_sent"]), "target: zero, always"),
             ("Parameter updates", str(a["updates_submitted"]), "weight vectors + sample counts"),
             ("Accepted / rejected", f"{a['updates_accepted']} / {a['updates_rejected']}",
              "malformed updates are quarantined"),
             ("Rounds completed", str(a["rounds_completed"]),
              str(a.get("last_training_time") or "no program yet"))])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="panel"><b>Stays inside each member</b><br>'
                    '<span style="color:#475467">Individual transaction records · account identifiers · '
                    'balances · customer PII. None of these cross the boundary at any point.</span></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="panel"><b>Transmitted to coordinator</b><br>'
                    '<span style="color:#475467">Model coefficient vectors · intercept · training-row count · '
                    'round metadata. Sufficient for FedAvg — and nothing more.</span></div>',
                    unsafe_allow_html=True)
    st.markdown(f"""<div class="panel" style="border-left:4px solid {AMBER}"><b>Disclosure — read carefully</b><br>
      <span style="color:#475467">Zero raw records does <b>not</b> mean zero privacy risk: parameter updates can
      still leak information about training rows. Clipping and noise in this console are configurable
      safeguards, <b>not</b> a certified differential-privacy guarantee (no formal budget accounting is
      implemented).</span></div>""", unsafe_allow_html=True)
    if r and r.get("history"):
        st.markdown('<div class="sec-title" style="font-size:1rem">Update-protection diagnostics · last round</div>',
                    unsafe_allow_html=True)
        ci = r["history"][-1].get("clip_info", [])
        if ci:
            cdf = pd.DataFrame([{"Member": BANK_LABEL[int(x.get("bank_id", i))],
                                 "Drift before clip": round(x.get("norm_before", 0), 4),
                                 "Drift after clip": round(x.get("norm_after", 0), 4),
                                 "Clipped": "Yes" if x.get("clipped") else "No",
                                 "Noise σ": x.get("noise_sigma", 0.0)}
                                for i, x in enumerate(ci)])
            st.dataframe(cdf, width='stretch', hide_index=True)

# ================================================= 6 LEDGER ================
elif page == "Trust Ledger":
    st.markdown('<div class="sec-title">Trust Ledger</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Append-only, SHA-256 hash-chained audit log of every federation event. '
                "The chain detects retrospective edits to the log; it does not certify that submitted updates "
                "were honest or that every real-world event was captured.</div>", unsafe_allow_html=True)
    r = _ensure()
    evs = r["ledger"] if r else []
    if not evs:
        st.info("Ledger is empty — run a training program first.")
    else:
        ok, msg = r["ledger_verify"]
        kpi_row([("Chain integrity", "VERIFIED" if ok else "BROKEN", msg),
                 ("Events recorded", str(len(evs)), f"{len(set(e['event'] for e in evs))} event types"),
                 ("Rounds covered", str(len(r["history"])), "genesis → latest aggregation")])
        ldf = pd.DataFrame([{"#": e["seq"], "Event": e["event"], "Actor": e["actor"],
                             "Round": e["round"], "Hash": e["hash"][:12] + "…",
                             "Detail": json.dumps(e["meta"])[:90]} for e in evs])
        st.dataframe(ldf, width='stretch', hide_index=True, height=420)
        if st.button("Run tamper-evidence check"):
            from src.trust_ledger import TrustLedger
            tl = TrustLedger()
            tl.events = [dict(e) for e in evs]
            tl.events[0]["meta"]["INJECTED"] = "unauthorised edit"
            ok2, msg2 = tl.verify()
            if not ok2:
                st.error(f"Tamper detected — verification failed as designed ({msg2}). "
                         "The log cannot be silently rewritten.")
            else:
                st.error("Unexpected: modified log verified. Escalate to engineering.")

# ================================================= 7 RISK ==================
else:
    st.markdown('<div class="sec-title">Risk scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Score a single transaction against the current shared model. '
                "Output is a triage signal — <b>flag for review</b> or <b>below review threshold</b> — "
                "never a verdict of fraud. Demonstration model, not a live monitoring feed.</div>",
                unsafe_allow_html=True)
    pre, params = st.session_state.pre, None
    if st.session_state.coord is not None:
        params = st.session_state.coord.global_params
    if (pre is None or params is None) and _ensure() is not None:
        try:
            from src.preprocessing import Preprocessor
            pre_cands = sorted(glob.glob(os.path.join(C.MODELS_DIR, "preprocessor_*.pkl")), reverse=True)
            mod_cands = sorted(
                [f for f in glob.glob(os.path.join(C.MODELS_DIR, "global_*.npz"))
                 if not os.path.basename(f).startswith("global_model_")], reverse=True)
            for pf in pre_cands:
                pre_try = Preprocessor.load(pf)
                need = len(pre_try.schema()["feature_order"])
                for mf in mod_cands:
                    dd = np.load(mf)
                    if dd["coef"].shape[1] == need:
                        pre, params = pre_try, {"coef": dd["coef"], "intercept": dd["intercept"]}
                        break
                if params is not None:
                    break
            if params is None:
                raise RuntimeError("no compatible model found")
        except Exception as e:
            st.warning(f"No deployable model available ({e}). Complete a training program first.")
            pre, params = None, None
    if pre is None or params is None:
        st.info("Complete a training program in the Training Lab to activate scoring.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="panel"><b>Transaction under review</b></div>', unsafe_allow_html=True)
            typ = st.selectbox("Transaction type", C.TYPE_CATEGORIES, index=3)
            amt = st.number_input("Amount", 0.0, value=2500.0, step=100.0)
            c1a, c1b = st.columns(2)
            with c1a:
                obo = st.number_input("Origin balance (before)", 0.0, value=5000.0)
                odb = st.number_input("Destination balance (before)", 0.0, value=0.0)
            with c1b:
                nbo = st.number_input("Origin balance (after)", 0.0, value=2500.0)
                ndb = st.number_input("Destination balance (after)", 0.0, value=2500.0)
            step = st.slider("Time step", 1, 744, 200)
        with c2:
            st.markdown('<div class="panel"><b>Model assessment</b></div>', unsafe_allow_html=True)
            if st.button("Score transaction", type="primary"):
                out = score_transaction({"amount": amt, "type": typ, "oldbalanceOrg": obo,
                                         "newbalanceOrig": nbo, "oldbalanceDest": odb,
                                         "newbalanceDest": ndb, "isFlaggedFraud": 0,
                                         "step": step}, pre, params, float(val_thr))
                kpi_row([("Fraud-risk score", f"{out['risk_score']:.3f}",
                          f"review threshold {out['threshold']:.2f}"),
                         ("Disposition", out["verdict"], out["note"])])
                st.progress(min(1.0, out["risk_score"]),
                            text=f"Risk {out['risk_score']:.1%} of threshold {out['threshold']:.0%}")
                if out["flag"]:
                    st.error("Disposition: FLAG FOR REVIEW — route to investigations. "
                             "This is a statistical signal, not proof of fraud.")
                else:
                    st.success("Disposition: BELOW REVIEW THRESHOLD — no action required by this model.")

st.divider()
st.caption("CIPHERMESH prototype console · member nodes simulated on this host · synthetic/sampled data · "
           "for pilot evaluation only — production deployment requires security review, authenticated channels, "
           "secure aggregation, privacy analysis and legal approval.")
