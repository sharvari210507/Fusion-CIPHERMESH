# 🔐 CIPHERMESH — Privacy-Preserving Fraud Signal Sharing Across Banks

**Problem (CSB-04):** banks detect fraud independently but can't share raw customer
transactions. Fraud rings exploit the gaps by hopping between banks. CIPHERMESH shows how
5 simulated banks can **collaboratively train one fraud detector without sending a single
raw transaction row** to the coordinator — using Federated Averaging (FedAvg).

> **Simulation, not deployment.** All five banks run on this one computer. The dataset is
> synthetic. See [Honest constraints](#honest-constraints) before believing anything else.

---

## ⚡ Quick-start (beginner friendly)

Run every command **from the project folder** (`Fusion-CIPHERMESH/`).

```bash
# 1. Create + activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the checks (13 tests, ~seconds, synthetic data)
python -m pytest tests/ -q

# 4. Start the dashboard
streamlit run app.py
# → opens http://localhost:8501
```

**First training run:** open section **C. Training Lab** → press **🚀 Run federated training**.
The first run downloads the Hugging Face dataset (`flwrlabs/fed-fraud-paysim-banks`, ~5.7M rows),
takes a reproducible stratified sample (default 30 000 rows, seed 42, ≥30 frauds/bank guaranteed),
and caches it in `data/` — later runs take ~1 min.

**No internet?** Tick *“Force synthetic fallback”* in the sidebar — the run uses a small
clearly-labelled synthetic dataset instead. It is never presented as the real data.

### Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | You forgot `source venv/bin/activate`, or reinstall: `pip install -r requirements.txt` |
| HF download fails / slow | Tick *Force synthetic fallback*; or retry (anonymous HF rate limits apply) |
| `KeyError: 'BankID'` / shape errors | Delete `data/subset_*.parquet` and re-run (stale cache) |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| Python < 3.10 | Install Python 3.11+; check with `python3 --version` |

---

## 🧭 How it works (5-minute version)

1. **Split** — HF transactions are partitioned by `BankID` → Bank A…E. Each bank keeps its
   own train/test split (stratified, no leakage between them).
2. **Preprocess once** — one `Preprocessor` (one-hot `type` in fixed order + `StandardScaler`
   fit on **train only**) transforms every bank identically, so all models have the same
   13 features in the same order. Target `isFraud`, `BankID`, `nameOrig/Dest` are never inputs.
3. **Local train** — each bank installs the current global weights into its own
   `SGDClassifier(loss="log_loss")` and runs local SGD steps (`partial_fit`) on its rows.
4. **Protect** — each update's drift from global is clipped to a max norm; optional Gaussian
   noise can be added. Settings + norms are logged (this is **not** formal differential privacy).
5. **FedAvg** — the coordinator validates every update (shape, finite values) and averages:
   `w_global = Σ (n_k/Σn_j) · w_k`. Bad updates are rejected + logged, never averaged.
6. **Evaluate honestly** — the new global is scored on each bank's **held-out test** rows:
   precision, recall, F1, PR-AUC (average precision — the right metric at ~0.1% fraud),
   confusion matrix, fraud support. Local-only vs federated is shown side by side;
   regressions are displayed, never hidden.
7. **Audit** — every step lands in a SHA-256 hash-chained ledger (tamper-evident log)
   and in privacy counters proving `Raw Rows Sent = 0`.

**Jargon buster:** *FedAvg* = weighted average of model weights. *PR-AUC* = area under the
precision-recall curve; unlike accuracy it stays honest when fraud is 1 in 1000.
*Clipping* = capping how far one bank can drag the shared model per round.

---

## 🗂️ Project structure

```
app.py                 # Streamlit dashboard (sections A–G) — start here
config.py              # All defaults: dataset, seed, model, features
src/
  data_loader.py       # HF load + stratified subset + per-bank splits (+fallback)
  preprocessing.py     # Shared encoder/scaler, schema validation, save/load
  bank_simulator.py    # Bank A–E objects (own data + local training)
  local_training.py    # SGD log-loss, init-from-global, single-class fallback
  fedavg.py            # From-scratch FedAvg + update validation
  coordinator.py       # Rounds: distribute → train → protect → aggregate → evaluate
  evaluation.py        # Precision/recall/F1/PR-AUC/confusion/support (rare-label safe)
  privacy_audit.py     # Verified transfer counters + event timeline
  update_protection.py # Clipping + Gaussian-noise experiment
  trust_ledger.py      # SHA-256 hash-chained audit log + tamper demo
  risk_scoring.py      # Single-transaction scoring demo
  pipeline.py          # run_experiment(): full run + saved JSON/CSV/model artifacts
tests/                 # pytest suite (FedAvg, clipping, ledger, privacy, end-to-end…)
data/ results/ models/ # git-ignored artifacts (cache, JSON/CSV, .npz weights)
```

Legacy files kept working: `dashboard.py` (v1 UI) and `src/federated_learning.py`,
`src/dataset.py` (v1 modules) still import fine.

## 📊 Dashboard tour (sections A–G)

- **A. Overview** — purpose, banks, dataset, global-model status, latest scores.
- **B. Data & Banks** — schema, rows/bank, fraud distribution, warnings, sampling docs.
- **C. Training Lab** — run federated training (rounds, iters, banks, clip, noise), watch F1/round.
- **D. Bank Divergence Radar** — per-bank local-vs-federated charts, confusion matrices, support counts.
- **E. Privacy Transparency** — raw-row counter (=0), update counts, data-flow, clip diagnostics.
- **F. Trust Ledger** — hash-chain table, PASS/FAIL verification, one-click tamper demo.
- **G. Risk Scoring Demo** — enter a transaction → risk score + *Flag for review / Below threshold*.

## 🧪 Tests

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Covers: FedAvg math + sample weighting, shape/NaN/inf rejection, clipping bound, ledger
pass-then-fail-after-tamper, coordinator-takes-no-raw-rows, rare-label metrics, cross-bank
feature compatibility, preprocessor+model save/load, full synthetic end-to-end run.

## 🗣️ Judge demo script (3 min)

1. **A**: “5 simulated banks, one shared model, zero raw rows move.”
2. **B**: “Real HF PaySim schema; fraud is ~0.1% — note the per-bank fraud counts.”
3. **C**: Run 3 rounds live → point at rising (or honest flat) F1 curves.
4. **D**: “Bank C gains +0.06 F1 from federation; Bank E drops — we show that too.”
5. **E**: “Raw rows = 0, updates = N; but updates can still leak — no absolute privacy claim.”
6. **F**: Verify PASS → tamper demo → FAIL. “Tamper-evident log, not proof of honesty.”
7. **G**: Score a big TRANSFER → *Flag for review* (not ‘confirmed fraud’).

## Honest constraints

- Banks are simulated on one computer: no TLS, auth, or secure aggregation.
- Dataset is synthetic (PaySim-style); results don't transfer to real fraud.
- Model updates can leak training-row info; clipping/noise here are **not** formal DP
  (no privacy budget/accounting).
- Hash ledger detects log edits; it doesn't prove updates were honest or complete.
- No real bank-system integration. Production would need security review, authenticated
  channels, access control, secure aggregation, privacy analysis, legal review, pilots.
- CIPHERMESH does not prevent all fraud or guarantee better detection.

## Future work

Secure aggregation, TLS + auth, formal DP accounting (e.g. Opacus-style), per-bank
threshold tuning, non-IID robustness (FedProx), model interpretability (SHAP), drift monitoring.
