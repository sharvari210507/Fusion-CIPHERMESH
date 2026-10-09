"""Risk scoring demo: score ONE transaction with the trained global model.

Input = the same model fields the pipeline uses. Output = fraud-risk score in
[0,1] + flag/below-threshold at a configurable threshold. A high score means
'flag for review', never 'confirmed fraud'. This is a demo on a simulated
model — not a live bank-monitoring integration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation import score_probs


def score_transaction(tx: dict, preprocessor, global_params: dict,
                      threshold: float = 0.5) -> dict:
    df = pd.DataFrame([{
        "step": float(tx.get("step", 100)),
        "type": str(tx.get("type", "PAYMENT")),
        "amount": float(tx.get("amount", 100.0)),
        "oldbalanceOrg": float(tx.get("oldbalanceOrg", 500.0)),
        "newbalanceOrig": float(tx.get("newbalanceOrig", 400.0)),
        "oldbalanceDest": float(tx.get("oldbalanceDest", 300.0)),
        "newbalanceDest": float(tx.get("newbalanceDest", 400.0)),
        "isFlaggedFraud": int(tx.get("isFlaggedFraud", 0)),
    }])
    X = preprocessor.transform(df)
    p = float(score_probs(global_params["coef"], global_params["intercept"], X)[0])
    return {"risk_score": p, "threshold": float(threshold),
            "flag": bool(p >= threshold),
            "verdict": "Flag for review" if p >= threshold else "Below review threshold",
            "note": "Model score only — not proof of fraud."}
