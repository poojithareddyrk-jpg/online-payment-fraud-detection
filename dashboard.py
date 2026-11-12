import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Online Payment Fraud Detection", layout="wide")
st.title("Online Payment Fraud Detection: Model Dashboard")
st.caption("Comparison of the models trained on the transaction dataset "
           "(about 6.36 million transactions, 0.13% fraud).")

df = pd.read_csv("model_comparison.csv")

# ---------- Top summary numbers ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models compared", len(df))
c2.metric("Transactions", "6,362,620")
c3.metric("Fraud cases", "8,213")
best = df.loc[df["F1"].idxmax()]
c4.metric("Best F1 score", f"{best['F1']:.4f}", best["model"], delta_color="off")

tab1, tab2, tab3 = st.tabs(
    ["Model comparison", "Missed fraud vs false alarms", "Data insights"])

# ---------- Tab 1: comparison ----------
with tab1:
    st.subheader("Results table")
    score_cols = [c for c in ["precision", "recall", "F1", "ROC_AUC", "PR_AUC"]
                  if c in df.columns]
    styled = df.style.format(precision=6).highlight_max(
        subset=score_cols, color="#c6efce")
    st.dataframe(styled)
    st.caption("Green = best value in each column. All scores use the same "
               "test set (1,272,524 transactions, 1,643 of them fraud).")

    st.subheader("Compare metrics")
    chosen = st.multiselect("Metrics to show", score_cols,
                            default=["precision", "recall", "F1"])
    zoom = st.checkbox("Zoom the y-axis to 0.9 - 1.0 (scores are very close)",
                       value=True)
    if chosen:
        long = df.melt(id_vars="model", value_vars=chosen,
                       var_name="metric", value_name="score")
        fig = px.bar(long, x="model", y="score", color="metric",
                     barmode="group")
        if zoom:
            fig.update_yaxes(range=[0.9, 1.0])
        st.plotly_chart(fig)

    st.subheader("Training time")
    st.plotly_chart(px.bar(df, x="model", y="train_time_s",
                           labels={"train_time_s": "seconds"}))

# ---------- Tab 2: errors ----------
with tab2:
    st.subheader("What each model gets wrong")
    err = df.melt(id_vars="model",
                  value_vars=["missed_fraud", "false_alarms"],
                  var_name="error type", value_name="count")
    st.plotly_chart(px.bar(err, x="model", y="count", color="error type",
                           barmode="group"))
    st.write("**Missed fraud** = real fraud the model let through (costly). "
             "**False alarms** = genuine transactions wrongly flagged "
             "(annoying for customers). Lower is better for both.")

# ---------- Tab 3: EDA ----------
with tab3:
    st.subheader("What the data looks like")
    for img in ["eda_charts_1.png", "eda_charts_2.png"]:
        if Path(img).exists():
            st.image(img)
    st.write("Fraud only occurs in TRANSFER and CASH_OUT transactions, and "
             "only about 0.13% of all transactions are fraud.")