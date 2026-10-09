import json
import streamlit as st
import pandas as pd
from backend import data as D, database as db
from backend import features as F
from backend import metrics as M
from backend.jobs import get_manager
from views.overview import strip


def page(user):
    st.title("Data Explorer")
    st.write("Purpose: inspect the synthetic dataset distribution per bank.")
    strip()
    s = D.get_stats()
    if "banks" not in s:
        st.warning("Dataset not ready."); return
    st.dataframe(pd.DataFrame.from_dict(s["banks"], orient="index"))
    st.bar_chart(pd.DataFrame(
        [{"bank": b, "fraud_rate": v["fraud_rate"]} for b, v in s["banks"].items()]).set_index("bank"))
    st.write("Type mix (train):", s.get("type_mix"))
    st.subheader("Feature separability (stand-alone PR-AUC, train sample)")
    df = D.load_split("train", columns=["step", "type", "amount", "oldbalanceOrg",
                                        "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]).sample(50000, random_state=42)
    X = F.transform(df, "engineered"); y = df["isFraud"].values
    rows = [{"feature": c, "pr_auc": round(M.pr_auc(y, X[c].values), 4)} for c in X.columns]
    st.dataframe(sorted(rows, key=lambda r: -r["pr_auc"]))
    st.info("Label-leakage note: in PaySim, balance columns almost determine fraud, so "
            "engineered error-balance features make the task close to trivial. Raw mode is the default.")
