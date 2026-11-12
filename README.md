# online-payment-fraud-detection

## Project Overview

This project focuses on detecting fraudulent online payment transactions using machine learning and deep learning frameworks. It compares different approaches based on their predictive performance, simplicity, transparency, and efficacy.

The project explores the following frameworks:

1. XGBoost
2. Keras
3. PyTorch + fastai
4. PyTorch + TabNet
5. PyTorch + Lightning Flash

## Project Objectives

- Analyze the online payment fraud detection dataset.
- Perform Exploratory Data Analysis (EDA).
- Clean and preprocess the dataset.
- Split the dataset into training and testing sets.
- Implement different machine learning and deep learning frameworks.
- Evaluate models using RMSE, R² and MAE.
- Compare the frameworks based on simplicity, transparency and efficacy.

## Dataset

The project uses an online payment fraud detection dataset.

- Dataset file: `new_data.csv`
- Dataset size: Approximately 493 MB.
- TabNet experiment: 500,000 sampled records.
- Target variable: `isFraud`.

  Download it from: https://drive.google.com/file/d/127JqP3WGjBVihR-ZcUR86T3wwy3_g63v/view


## Project Structure

```text
online-payment-fraud-detection/
│
├── eda.ipynb
├── xgboost_model.ipynb
├── comparison.ipynb
├── dashboard.py
│
├── keras_fastai/
│   ├── Keras implementation
│   └── fastai implementation
│
├── data.py
├── metrics.py
├── run_tabnet.py
├── run_flash.py
├── new_data.csv
├── results.csv
│
├── model_comparison.csv
├── keras_results.csv
├── fastai_results.csv
├── xgboost_results.csv
│
└── README.md
```

*Note: Some Person 3 files and results currently exist only in the local `fraudproj` folder and have not yet been uploaded to GitHub.*

## Work Completed

### Person 1: Dataset and XGBoost

- Dataset exploration.
- Exploratory Data Analysis (EDA).
- Data cleaning and preprocessing.
- Train/test splitting.
- XGBoost implementation.

### Person 2: Keras and fastai

- Keras implementation.
- PyTorch + fastai implementation.
- Framework-specific methodology and initial results work.

### Person 3: TabNet and Lightning Flash

#### PyTorch + TabNet

- Prepared the TabNet implementation.
- Used the shared data preprocessing module.
- Used the shared evaluation metrics module.
- Trained the model using 500,000 sampled records.
- Used CPU for training.
- Applied early stopping.
- Generated feature importance scores.


