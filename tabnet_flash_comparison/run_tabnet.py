"""Train TabNetRegressor and score the full test set.
Usage: python run_tabnet.py [--data .] [--out out] [--epochs 8] [--rows 0]
 --rows N  uses only the first N training rows (smoke test); 0 = all.
Fits on the train rows minus a 200,000-row validation slice (first 200,000 rows of the shuffled train file), keeps the checkpoint with the best
validation RMSE (out/tabnet_best.zip), then scores the test set with that checkpoint (out/tabnet_results.json).
NOTE: the reported results in this folder came from an earlier version of this script that saved the last epoch after every epoch and was cut
short by a machine restart; tabnet_epoch4_best.zip is the best-validation checkpoint from that run. infer_ckpt.py reproduces its scores."""
import argparse,gc,json,os,time,numpy as np,torch,pyarrow.parquet as pq
from pytorch_tabnet.tab_model import TabNetRegressor
from common import metrics,MONEY
ap=argparse.ArgumentParser();ap.add_argument("--data",default=".");ap.add_argument("--out",default="out");ap.add_argument("--epochs",type=int,default=8);ap.add_argument("--rows",type=int,default=0);a=ap.parse_args()
os.makedirs(a.out,exist_ok=True);torch.manual_seed(42);np.random.seed(42)
def arr(path,rows=0):
    t=pq.read_table(path);t=t.slice(0,rows) if rows else t
    cols=[c for c in t.column_names if c!="isFraud"];y=t.column("isFraud").to_numpy().astype("float32")
    X=np.empty((t.num_rows,len(cols)),dtype="float32")
    for j,c in enumerate(cols): X[:,j]=t.column(c).to_numpy()
    del t;gc.collect();return X,y,cols
Xtr,ytr,cols=arr(os.path.join(a.data,"train32.parquet"),a.rows);Xte,yte,_=arr(os.path.join(a.data,"test32.parquet"),a.rows//4 if a.rows else 0)
mi=[cols.index(c) for c in MONEY]
for X in (Xtr,Xte):
    for j in mi: X[:,j]=np.sign(X[:,j])*np.log1p(np.abs(X[:,j]))
mu=Xtr.mean(0);sd=Xtr.std(0)+1e-6
for X in (Xtr,Xte): X-=mu;X/=sd
nv=min(200000,len(Xtr)//5);Xv,yv=Xtr[:nv],ytr[:nv].reshape(-1,1)
m=TabNetRegressor(seed=42,verbose=1);t0=time.time()
m.fit(Xtr[nv:],ytr[nv:].reshape(-1,1),eval_set=[(Xv,yv)],eval_metric=["rmse"],max_epochs=a.epochs,patience=3,batch_size=8192,virtual_batch_size=512)
tt=time.time()-t0
m.save_model(os.path.join(a.out,"tabnet_best"))  # TabNet restores the best-validation weights at the end of fit when early stopping is on
p=np.concatenate([m.predict(Xte[i:i+100000]).ravel() for i in range(0,len(Xte),100000)])
r=metrics("PyTorch TabNet",yte,p,tt);r.update(fit_rows=len(Xtr)-nv,validation_rows=nv,test_rows=len(Xte))
json.dump(r,open(os.path.join(a.out,"tabnet_results.json"),"w"),indent=1);print(r)
