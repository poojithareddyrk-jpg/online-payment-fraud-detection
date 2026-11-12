"""Usage: python flash_train.py [--rows 750000 --epochs 8 --batch 1024 --data . --out out]
Stage A: Lightning Flash TabularRegressor on a proportionally stratified sample of the train rows.
Sample: N_SAMPLE rows drawn with sklearn train_test_split(stratify=isFraud, random_state=42) from the 5,090,096 train rows,
so the fraud rate equals the full train rate (0.129%). Validation: 10% of the sample (Flash val_split)."""
import sys,time,json,gc,numpy as np,pandas as pd,torch,pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
from flash import Trainer
from flash.tabular import TabularRegressionData,TabularRegressor
from common import MONEY
import argparse,os
ap=argparse.ArgumentParser();ap.add_argument("--rows",type=int,default=750000);ap.add_argument("--epochs",type=int,default=8);ap.add_argument("--batch",type=int,default=1024);ap.add_argument("--data",default=".");ap.add_argument("--out",default="out");a=ap.parse_args()
N,EP,BS=a.rows,a.epochs,a.batch;os.makedirs(a.out,exist_ok=True)
torch.manual_seed(42);np.random.seed(42)
y=pq.read_table(os.path.join(a.data,"train32.parquet"),columns=["isFraud"]).column("isFraud").to_numpy()
idx,_=train_test_split(np.arange(len(y)),train_size=N,random_state=42,stratify=y);idx.sort()
tr=pq.read_table(os.path.join(a.data,"train32.parquet")).take(idx).to_pandas();del y;gc.collect()
cols=[c for c in tr.columns if c!="isFraud"]
for c in MONEY: tr[c]=np.sign(tr[c])*np.log1p(np.abs(tr[c]))
info=dict(n_sample=len(tr),n_fraud=int(tr.isFraud.sum()),fraud_rate=float(tr.isFraud.mean()),epochs=EP,batch_size=BS)
print(info,flush=True)
dm=TabularRegressionData.from_data_frame(numerical_fields=cols,target_field="isFraud",train_data_frame=tr,val_split=0.1,batch_size=BS,num_workers=0)
del tr;gc.collect()
m=TabularRegressor.from_data(dm);t=Trainer(max_epochs=EP,accelerator="cpu",enable_progress_bar=False,enable_checkpointing=False,default_root_dir=os.path.join(a.out,"flash"))
s=time.time();t.fit(m,datamodule=dm);info["train_time_s"]=time.time()-s;info["final_val_metrics"]={k:float(v) for k,v in t.callback_metrics.items()}
t.save_checkpoint(os.path.join(a.out,"flash_final.ckpt"))
json.dump(info,open(os.path.join(a.out,"flash_train_info.json"),"w"));print(info)
