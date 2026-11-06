
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

print("Loading training data...")

train = pd.read_pickle("../train.pkl")

feature_cols = [
    "step", "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "errorBalanceOrig", "errorBalanceDest",
    "type_CASH_IN", "type_CASH_OUT", "type_DEBIT",
    "type_PAYMENT", "type_TRANSFER"
]

print("Fitting StandardScaler...")

scaler = StandardScaler()
scaler.fit(train[feature_cols].to_numpy(dtype="float32"))

joblib.dump(scaler, "keras_scaler.pkl")

print("Scaler saved successfully!")
