"""Shared data loading for the whole team.

Everyone imports load_split() so all frameworks use the SAME split.
Edit CSV_PATH / TARGET below once the dataset columns are confirmed.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "fraud.csv"   # change to your downloaded file name
TARGET = "isFraud"       # change if the label column has a different name
SEED = 42


def load_raw(path=CSV_PATH, sample=None):
    df = pd.read_csv(path)
    if sample:  # optional: use a subset if the file is huge
        df = df.sample(n=sample, random_state=SEED)
    return df


def add_features(df):
    """Feature engineering. Keep only what exists in your dataset."""
    cols = set(df.columns)
    if {"newbalanceOrig", "amount", "oldbalanceOrg"} <= cols:
        df["errorOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    if {"oldbalanceDest", "amount", "newbalanceDest"} <= cols:
        df["errorDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    return df


def prepare(df):
    df = add_features(df.copy())
    # drop identifier-like text columns (e.g. nameOrig, nameDest)
    id_cols = [c for c in df.columns if c.lower().startswith("name")]
    df = df.drop(columns=id_cols + ["isFlaggedFraud"], errors="ignore")
    # one-hot encode remaining text columns (e.g. type)
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df


def load_split(path=CSV_PATH, sample=None, test_size=0.2, scale=False):
    """Returns X_train, X_test, y_train, y_test (DataFrames/Series).

    scale=True standardises features (use this for Keras / PyTorch models).
    Tree models like XGBoost do not need scaling.
    """
    df = prepare(load_raw(path, sample))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr = pd.DataFrame(sc.transform(X_tr), columns=X.columns, index=X_tr.index)
        X_te = pd.DataFrame(sc.transform(X_te), columns=X.columns, index=X_te.index)
    return X_tr, X_te, y_tr, y_te
