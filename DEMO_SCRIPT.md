# CIPHERMESH — 3-minute judge demo script

> Setup before judges arrive: `streamlit run app.py`, open section C, defaults loaded
> (30k HF subset cached, 5-round baseline saved). Keep this script open in a second tab.

## 0:00–0:30 — A. Overview: the one-sentence pitch
“Banks see fraud independently but can't share customer rows. CIPHERMESH trains **one
shared fraud detector across 5 simulated banks with zero raw rows leaving any bank** —
only model weights move, averaged with FedAvg.”
Point at: *Simulated banks = 5 · Raw rows sent = 0 · Rounds completed = 5*.

## 0:30–1:00 — B. Data & Banks: real schema, rare fraud
“Real Hugging Face PaySim data, 5.7M rows sampled to 30k with the fraud class preserved.
Fraud is ~0.1% — that's why we report PR-AUC and recall, never accuracy alone.”
Point at the fraud-per-bank bars + any warnings.

## 1:00–1:45 — C. Training Lab: run it live
Press **🚀 Run federated training** (3 rounds for speed). Narrate while it runs:
“Each bank installs the global weights, trains locally, clips its update, sends back
only numbers — the coordinator validates shapes, rejects garbage, and averages.”
Point at the F1-over-rounds chart when it lands.

## 1:45–2:15 — D. Divergence Radar: honest numbers
“This is our differentiator: local-only vs federated **per bank**. Bank 0 gains, Bank 3
*loses* — we show the regression instead of hiding it. With 6 frauds per test split,
treat small gaps as noisy.”
Point at ΔF1 column + one confusion matrix.

## 2:15–2:45 — E+F. Privacy + Ledger: no fairy tales
“Raw rows = 0, verified by tests — but updates can still leak, and our clipping is **not**
formal differential privacy.” Then in F: press **Verify** (PASS), press **Tamper demo**
(FAIL). “Tamper-evident log — it catches edits, it doesn't prove honesty.”

## 2:45–3:00 — G. Risk demo: close with a story
Score a TRANSFER of $8,000 with drained origin balance → **Flag for review**.
“A high score means *flag for review*, never *confirmed fraud*. And this is a demo model,
not a live bank feed.”

## Likely questions
- *“Is this differentially private?”* — “No. Clipping + noise are the mechanism shape,
  but we implement no budget accounting, so we claim nothing formal.”
- *“Why does a bank get worse?”* — “Non-IID data + tiny fraud support; federation helps
  on average, not everywhere — that's why per-bank reporting matters.”
- *“Production-ready?”* — “No: needs TLS, auth, secure aggregation, legal review.
  See README constraints.”
