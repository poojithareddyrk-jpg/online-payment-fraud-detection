"""Shared evaluation. Every framework calls evaluate() so results are comparable.

Usage:
    from metrics import evaluate
    evaluate("XGBoost", y_test, preds, train_seconds=12.3, lines_of_code=25)
Results are appended to results.csv, which the comparison table is built from.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             precision_score, recall_score, f1_score)

RESULTS_FILE = "results.csv"


def evaluate(name, y_true, y_pred, train_seconds=None, lines_of_code=None, notes=""):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_label = (y_pred >= 0.5).astype(int)  # for the extra classification view

    row = {
        "framework": name,
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "precision": precision_score(y_true, y_label, zero_division=0),
        "recall": recall_score(y_true, y_label, zero_division=0),
        "f1": f1_score(y_true, y_label, zero_division=0),
        "train_seconds": train_seconds,
        "lines_of_code": lines_of_code,
        "notes": notes,
    }
    print(pd.Series(row))

    df = pd.DataFrame([row])
    df.to_csv(RESULTS_FILE, mode="a", header=not os.path.exists(RESULTS_FILE), index=False)
    return row
