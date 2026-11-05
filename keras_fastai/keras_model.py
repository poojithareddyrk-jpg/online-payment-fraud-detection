import os
import time
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Loading training and testing data...")

# Load Person 1's exact datasets
train = pd.read_pickle("../train.pkl")
test = pd.read_pickle("../test.pkl")

# Separate features and target
X_train = train.drop(columns=["isFraud"])
y_train = train["isFraud"].to_numpy(dtype=np.float32)

X_test = test.drop(columns=["isFraud"])
y_test = test["isFraud"].to_numpy(dtype=np.float32)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Scale features using training data only
print("Scaling features...")

scaler = StandardScaler()

X_train = X_train.to_numpy(dtype=np.float32)
X_test = X_test.to_numpy(dtype=np.float32)

scaler.fit(X_train)
scaler.transform(X_train, copy=False)
scaler.transform(X_test, copy=False)

# Build Keras regression model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="linear")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="mse"
)

model.summary()

# Train model
print("\nStarting Keras training...")

start_time = time.time()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=4096,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
training_time = time.time() - start_time

# Predictions
print("\nEvaluating model...")

predictions = model.predict(X_test, batch_size=8192).flatten()

# Calculate regression metrics
rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Display results
results = {
    "Model": "Keras Neural Network",
    "RMSE": float(rmse),
    "MAE": float(mae),
    "R2": float(r2),
    "Training_Time_Seconds": float(training_time)
}

print("\n========== KERAS RESULTS ==========")
for key, value in results.items():
    print(f"{key}: {value}")

# Save model and results
model.save("keras_fraud_model.keras")

with open("keras_results.json", "w") as f:
    json.dump(results, f, indent=4)

print("\nKeras model and results saved successfully!")