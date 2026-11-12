"""Rebuild the train/test split exactly as eda.ipynb does (80/20, stratified, random_state=42), in float32 parquet.
Usage: python prep.py [--csv new_data.csv] [--out .]   -> train32.parquet, test32.parquet (expect 5,090,096 / 1,272,524 rows)"""
import argparse,os,numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
ap=argparse.ArgumentParser();ap.add_argument("--csv",default="new_data.csv");ap.add_argument("--out",default=".");a=ap.parse_args()
types=["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]
use=["step","type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud"]
parts=[]
for ch in pd.read_csv(a.csv,chunksize=1_000_000,usecols=use):
    d=ch.copy()
    d["errorBalanceOrig"]=d.newbalanceOrig+d.amount-d.oldbalanceOrg
    d["errorBalanceDest"]=d.oldbalanceDest+d.amount-d.newbalanceDest
    for t in types: d["type_"+t]=(d.type==t).astype("int8")
    d=d.drop(columns="type")
    parts.append(d.astype({c:"float32" for c in d.columns if d[c].dtype=="float64"}))
df=pd.concat(parts,ignore_index=True);del parts
print(df.shape,"frauds:",int(df.isFraud.sum()))
tr,te=train_test_split(np.arange(len(df)),test_size=0.2,random_state=42,stratify=df.isFraud.values)
os.makedirs(a.out,exist_ok=True)
df.iloc[tr].reset_index(drop=True).to_parquet(os.path.join(a.out,"train32.parquet"))
df.iloc[te].reset_index(drop=True).to_parquet(os.path.join(a.out,"test32.parquet"))
print("train",len(tr),"test",len(te),"test frauds",int(df.isFraud.values[te].sum()))
