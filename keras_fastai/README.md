\# Person 2 - Keras and PyTorch + fastai



\## Overview

This module implements and compares Keras and PyTorch + fastai

neural network models for online payment fraud detection.



\## Dataset

Both models use the same preprocessed training and testing

datasets provided by Person 1.



Target variable: isFraud



\## Models

1\. Keras Neural Network

&#x20;  - Dense layers: 64, 32, 1

&#x20;  - ReLU activation

&#x20;  - Adam optimizer

&#x20;  - Mean Squared Error loss

&#x20;  - EarlyStopping callback



2\. PyTorch + fastai

&#x20;  - TabularLearner

&#x20;  - Layers: 64, 32

&#x20;  - MSE loss

&#x20;  - fit\_one\_cycle training



\## Keras Results

\- RMSE: 0.02247

\- MAE: 0.00363

\- R2: 0.60828

\- Training time: 22.83 seconds



\## PyTorch + fastai Results

\- RMSE: 0.02217

\- MAE: 0.00541

\- R2: 0.61879

\- Training time: 127.98 seconds



\## Conclusion

Keras achieved lower MAE and faster training time.

PyTorch + fastai achieved slightly better RMSE and R2.



Both models were evaluated using the same test dataset.



Note: The target is treated as a numerical regression target

in this implementation.

