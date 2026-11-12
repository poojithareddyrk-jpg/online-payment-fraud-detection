"""Score a saved TabNet checkpoint. Usage: python infer_ckpt.py CHECKPOINT.zip val|test
Reproduces the reported numbers from tabnet_epoch4_best.zip (val RMSE 0.00356186 on the first 200k train rows; test results in tabnet_results.json).
Recomputes the feature mean/std from train32.parquet exactly as the training script did."""
import sys,json,gc,numpy as np,pyarrow.parquet as pq
from pytorch_tabnet.tab_model import TabNetRegressor
from common import metrics,MONEY
ck=sys.argv[1];mode=sys.argv[2]  # mode: val or test
def tf(c,v): return np.sign(v)*np.log1p(np.abs(v)) if c in MONEY else v
tr=pq.ParquetFile("train32.parquet");cols=[c for c in tr.schema_arrow.names if c!="isFraud"]
N=tr.metadata.num_rows
Xtr=np.empty((N,len(cols)),dtype="float32")
for j,c in enumerate(cols): Xtr[:,j]=tf(c,tr.read(columns=[c]).column(c).to_numpy().astype("float32"))
mu=Xtr.mean(0);sd=Xtr.std(0)+1e-6   # same axis-0 float32 reductions as the training script
yv=tr.read(columns=["isFraud"]).column("isFraud").to_numpy()[:200000].astype("float32")
Xv=((Xtr[:200000]-mu)/sd).astype("float32");del Xtr;gc.collect()
b=TabNetRegressor();b.load_model(ck)
p=b.predict(Xv).ravel();print("val_rmse_first200k",float(np.sqrt(np.mean((p-yv)**2))),flush=True)
if mode=="test":
    te=pq.ParquetFile("test32.parquet");yte=te.read(columns=["isFraud"]).column("isFraud").to_numpy().astype("float32")
    X=np.empty((len(yte),len(cols)),dtype="float32")
    for j,c in enumerate(cols): X[:,j]=(tf(c,te.read(columns=[c]).column(c).to_numpy().astype("float32"))-mu[j])/sd[j]
    P=np.concatenate([b.predict(X[i:i+100000]).ravel() for i in range(0,len(X),100000)])
    r=metrics("PyTorch TabNet",yte,P,0.0);print(r);json.dump(r,open("tabnet_ckpt_eval.json","w"),indent=1)
