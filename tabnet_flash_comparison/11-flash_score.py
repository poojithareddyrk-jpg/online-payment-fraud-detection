"""Usage: python flash_score.py [--ckpt flash_checkpoint_run2.ckpt --info flash_train_info_run2.json --data . --out .] (--max_rows N for a smoke test)
Stage B: score the FULL test set (1,272,524 rows) with the saved Flash checkpoint, in 100k-row chunks."""
import json,gc,time,numpy as np,pandas as pd,torch,pyarrow.parquet as pq
from flash import Trainer
from flash.tabular import TabularRegressionData,TabularRegressor
from common import MONEY,metrics
import argparse
ap=argparse.ArgumentParser();ap.add_argument("--ckpt",default="flash_checkpoint_run2.ckpt");ap.add_argument("--info",default="flash_train_info_run2.json");ap.add_argument("--data",default=".");ap.add_argument("--out",default=".");ap.add_argument("--max_rows",type=int,default=0);a=ap.parse_args()
ck=a.ckpt
m=TabularRegressor.load_from_checkpoint(ck);params=torch.load(ck,map_location="cpu",weights_only=False)["hyper_parameters"]["parameters"]
t=pq.read_table(__import__("os").path.join(a.data,"test32.parquet")).to_pandas();t=t.iloc[:a.max_rows] if a.max_rows else t;yte=t.isFraud.values;cols=[c for c in t.columns if c!="isFraud"]
for c in MONEY: t[c]=np.sign(t[c])*np.log1p(np.abs(t[c]))
tr=Trainer(accelerator="cpu",enable_progress_bar=False,enable_checkpointing=False,logger=False)
P=[];s=time.time()
for i in range(0,len(t),100000):
    dm=TabularRegressionData.from_data_frame(predict_data_frame=t[cols].iloc[i:i+100000].reset_index(drop=True),parameters=params,batch_size=8192,num_workers=0)
    P.append(np.concatenate([np.asarray(b).ravel() for b in tr.predict(m,datamodule=dm)]))
p=np.concatenate(P);assert len(p)==len(yte)
info=json.load(open(a.info))
r=metrics("Lightning Flash TabularRegressor (750k-row stratified train sample)",yte,p,info["train_time_s"]);r.update(info);r["score_time_s"]=time.time()-s
json.dump(r,open(__import__("os").path.join(a.out,"flash_results.json"),"w"),indent=1);print(r)
