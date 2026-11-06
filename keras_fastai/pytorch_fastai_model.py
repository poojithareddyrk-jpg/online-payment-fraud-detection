
import time
import json
import numpy as np
import pandas as pd
import torch

from fastai.tabular.all import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

np.random.seed(42)
torch.manual_seed(42)

print("Loading training and testing data...")

train = pd.read_pickle("../train.pkl")
test = pd.read_pickle("../test.pkl")

feature_cols = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

# Create training and validation splits
print("Preparing fastai TabularPandas...")

splits = RandomSplitter(valid_pct=0.1, seed=42)(range(len(train)))

to = TabularPandas(
    train,
    procs=[Normalize],
    cat_names=[],
    cont_names=feature_cols,
    y_names="isFraud",
    y_block=RegressionBlock(),
    splits=splits
)

dls = to.dataloaders(bs=4096, num_workers=0)

# Build TabularLearner
learn = tabular_learner(
    dls,
    layers=[64, 32],
    loss_func=MSELossFlat()
)

print("\nStarting PyTorch + fastai training...")
start_time = time.time()

learn.fit_one_cycle(5, 1e-2)

training_time = time.time() - start_time

# Evaluate on the original test dataset
print("\nEvaluating model...")

X_test = test[feature_cols]
y_test = test["isFraud"].to_numpy(dtype=np.float32)

test_dl = learn.dls.test_dl(X_test)
predictions, _ = learn.get_preds(dl=test_dl)

predictions = predictions.numpy().flatten()

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

results = {
    "Model": "PyTorch + fastai TabularLearner",
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R2": float(r2),
    "Training_Time_Seconds": float(training_time)
}

print("\n========== PYTORCH + FASTAI RESULTS ==========")

for key, value in results.items():
    print(f"{key}: {value}")

learn.export("pytorch_fastai_model.pkl")

with open("pytorch_fastai_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nPyTorch + fastai model and results saved successfully!")
