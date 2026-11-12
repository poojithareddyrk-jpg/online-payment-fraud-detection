"""Builds the 5-framework comparison table (markdown + json) and bar chart.
XGBoost / Keras / fastai rows come from the teammates' model_comparison.csv in the repo root (NOT re-run here)."""
import json,pandas as pd,matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
base=pd.read_csv("../model_comparison.csv")[["model","RMSE","R2","MAE","train_time_s"]]
tn=json.load(open("tabnet_results.json"));fl=json.load(open("flash_results.json"))
new=pd.DataFrame([tn,fl])[["model","RMSE","R2","MAE","train_time_s"]]
new["model"]=["PyTorch TabNet","Flash (750k sample)"]
t=pd.concat([base,new],ignore_index=True)
open("comparison_table.md","w").write(t.to_markdown(index=False,floatfmt=".6f") if hasattr(t,"to_markdown") else t.to_string(index=False))
t.to_json("comparison_table.json",orient="records",indent=1);print(t.to_string())
fig,ax=plt.subplots(1,3,figsize=(15,4.8))
for a,m,c in zip(ax,["RMSE","R2","MAE"],["#c44e52","#4c72b0","#dd8452"]):
    a.bar(t.model,t[m],color=c);a.set_title(m+(" (higher is better)" if m=="R2" else " (lower is better)"));a.tick_params(axis="x",rotation=35)
fig.text(0.5,0.005,"XGBoost, Keras, fastai: teammates' runs (classifiers scored by probability, fastai on 1M rows). TabNet fitted on 4.89M rows, Flash on ~675k. Not strictly like for like.",ha="center",fontsize=9,color="#585254");plt.tight_layout(rect=(0,0.04,1,1));plt.savefig("comparison_chart.png",dpi=150)
