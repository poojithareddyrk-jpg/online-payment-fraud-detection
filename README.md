# Online Payment Fraud Detection

An end-to-end machine-learning project that explores online payment transaction data, trains multiple models, and compares their ability to identify fraudulent transactions. The repository includes exploratory data analysis (EDA), model-training notebooks and scripts, saved evaluation results, and a Streamlit dashboard.

> **Important:** The comparison contains five model entries, but the experiments are not perfectly like-for-like. Read the evaluation notes below before drawing conclusions.

## Project overview

Online payment systems process large transaction volumes, and fraudulent transactions are rare compared with legitimate ones. This class imbalance makes fraud detection challenging: a model can appear accurate by predicting almost every transaction as legitimate while still missing fraud.

This project investigates the use of machine-learning and neural-network approaches to estimate the `isFraud` target and compares their performance using classification and regression metrics.

### Objectives

- Explore transaction patterns and the distribution of fraudulent transactions.
- Prepare transaction features for machine-learning models.
- Train and evaluate multiple modelling approaches.
- Compare precision, recall, F1-score, ROC-AUC, PR-AUC and regression metrics.
- Examine missed fraud cases, false alarms and training time.
- Present the available results through a Streamlit dashboard.

## Dataset

The project uses an online payment transaction dataset containing approximately **6,362,620 transactions**, including **8,213 fraud cases** (about **0.13%** of all rows).

The dataset file `new_data.csv` is approximately 471 MB and is intentionally not committed to this repository.

**Dataset download:** [Google Drive – new_data.csv](https://drive.google.com/file/d/127JqP3WGjBVihR-ZcUR86T3wwy3_g63v/view)

After downloading, place `new_data.csv` in the project root directory.

### Target

- `isFraud`: target variable indicating whether a transaction is fraudulent.

### Main input features

- `step`
- `type` (one-hot encoded into transaction-type columns)
- `amount`
- `oldbalanceOrg`
- `newbalanceOrig`
- `oldbalanceDest`
- `newbalanceDest`
- `errorBalanceOrig`
- `errorBalanceDest`

The Person 3 preprocessing script derives the two balance-error features and one-hot encodes the five transaction types: `CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, and `TRANSFER`.

## Models included

1. **XGBoost** – gradient-boosted decision trees.
2. **Keras MLP** – a feed-forward neural network.
3. **fastai Tabular** – a tabular deep-learning model built with fastai.
4. **PyTorch TabNet** – an attention-based neural architecture for tabular data.
5. **Lightning Flash TabularRegressor** – a tabular regression workflow using Lightning Flash.

The repository also contains earlier Keras and fastai result JSON files. The five-model comparison below uses the consolidated `model_comparison.csv` values for the first three models and the final JSON result files in `tabnet_flash_comparison/` for TabNet and Flash.

## Evaluation results

The comparison table reports the values stored in the project files.

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | RMSE | R² | MAE | Train time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.9425 | 0.9982 | 0.9696 | 0.9997 | 0.9981 | 0.002080 | 0.9966 | 0.000022 | 65.5 |
| Keras MLP | 0.9903 | 0.9976 | 0.9939 | 0.9999 | 0.9986 | 0.004365 | 0.9852 | 0.000317 | 60.9 |
| fastai Tabular | 0.9709 | 0.9945 | 0.9826 | 0.9990 | 0.9977 | 0.011738 | 0.8931 | 0.009665 | 11.1 |
| PyTorch TabNet | 0.9964 | 0.9976 | 0.9970 | 0.9996 | 0.9946 | 0.002998 | 0.9930 | 0.000605 | 366.0 |
| Lightning Flash | 0.9150 | 0.9300 | 0.9224 | 0.9751 | 0.9426 | 0.015972 | 0.8022 | 0.002595 | 246.5 |

### Interpretation

- **XGBoost** has the lowest reported RMSE (0.002080) and highest reported R² (0.996643) in this table.
- **Keras MLP** has the highest reported F1-score (0.993936), precision (0.990332), and PR-AUC (0.998615) among the five entries.
- **PyTorch TabNet** reports an F1-score of 0.996959, recall of 0.997565 and ROC-AUC of 0.999625.
- **Lightning Flash** reports lower classification scores than the other entries in this run and was trained on a smaller sample.
- The fastest reported training time is fastai Tabular at 11.1 seconds, but training times were measured in different environments/settings and should not be treated as a controlled speed benchmark.

### Important experimental limitations

The results should be interpreted carefully:

- The original train/test files shared by the other team members were not included in the repository. Person 3 reconstructed an 80/20 stratified split with `random_state=42`; the row counts match the XGBoost notebook, but byte-for-byte identity of the split was not verified.
- XGBoost, Keras MLP and fastai results were copied from the existing team comparison file and were not re-run by Person 3.
- The first three models were trained as classifiers, while TabNet and Flash were trained as MSE regressors. The regression scores and classification scores therefore come from different modelling setups.
- fastai used a 1,000,000-row training sample in the team experiment.
- TabNet was fitted on 4,890,096 rows after holding out 200,000 training rows for validation. Its saved checkpoint corresponds to the best validation result at epoch 4; the run was interrupted before the planned training schedule completed.
- Lightning Flash used a stratified 750,000-row sample (about 675,000 fitting rows and 75,000 validation rows), rather than the full training set. Its second run's settings were selected after the first run's test score had been observed, so its reported test result is not fully blind.
- Training times came from different runs and hardware/settings. They are descriptive, not a fair speed comparison.
- The 0.5 threshold is used to calculate the reported precision, recall and F1 for TabNet and Flash.
- Since fraud is rare, accuracy alone would be misleading. Precision, recall, F1 and PR-AUC are important when evaluating fraud detection.

## Project structure

```text
online-payment-fraud-detection/
├── README.md
├── dashboard.py
├── comparison.ipynb
├── eda.ipynb
├── xgboost_model.ipynb
├── keras_fastai_models.ipynb
├── model_comparison.csv
├── model_comparison.png
├── xgboost_results.csv
├── keras_results.csv
├── fastai_results.csv
├── eda_charts_1.png
├── eda_charts_2.png
├── requirements.txt
├── keras_fastai/
│   ├── keras_model.py
│   ├── pytorch_fastai_model.py
│   ├── keras_results.json
│   └── pytorch_fastai_results.json
└── tabnet_flash_comparison/
    ├── README.md
    ├── prep.py
    ├── common.py
    ├── run_tabnet.py
    ├── infer_ckpt.py
    ├── flash_train.py
    ├── flash_score.py
    ├── make_comparison.py
    ├── tabnet_results.json
    ├── flash_results.json
    ├── comparison_table.md
    └── comparison_chart.png
```

Model checkpoints and training logs are also present in the relevant model folders. The dataset itself is not included.

## Installation

Use a Python environment compatible with the libraries in `requirements.txt`. Some older model components, especially Lightning Flash 0.8.2, may require a separate compatible environment.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd online-payment-fraud-detection

python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\\Scripts\\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install the main project dependencies:

```bash
pip install -r requirements.txt
```

For the TabNet and Lightning Flash experiment, see the pinned environment and instructions in [`tabnet_flash_comparison/README.md`](tabnet_flash_comparison/README.md). That experiment was run with Python 3.10.12 and its own pinned dependency versions.

## Running the dashboard

After placing `new_data.csv` in the project root and installing the dependencies:

```bash
streamlit run dashboard.py
```

The dashboard source currently reads `model_comparison.csv` for its model comparison. That CSV contains the first three consolidated models. To show all five models in the dashboard, merge the TabNet and Flash rows from their JSON files into the comparison data (or update the dashboard to load those JSON files as well). The results are already saved; the two models do not need to be retrained just to display their existing metrics.

## Reproducing the TabNet and Lightning Flash experiments

From the `tabnet_flash_comparison/` directory, follow its README. The documented workflow is:

```bash
pip install -r requirements.txt
python prep.py --csv new_data.csv --out .
python run_tabnet.py --data . --out out
python flash_train.py --rows 750000 --epochs 8 --batch 1024 --data . --out out
python flash_score.py --ckpt out/flash_final.ckpt --info out/flash_train_info.json --data . --out out
python make_comparison.py
```

These commands reproduce the workflow, but exact results can vary with environment, hardware, package versions and training behaviour. The saved JSON files are the source of the reported results in this README.

## Tools and technologies

- Python
- pandas and NumPy
- scikit-learn
- XGBoost
- TensorFlow / Keras
- PyTorch
- fastai
- PyTorch TabNet
- Lightning Flash
- Matplotlib / Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

## Team contributions

The repository combines work from multiple team members. The included artifacts cover exploratory analysis, baseline modelling, Keras and fastai models, and the TabNet / Lightning Flash comparison. Update this section with the exact names and contribution split agreed by your team before publishing.

## Future improvements

- Run all five models using the exact same train/validation/test rows.
- Use a consistent classification objective and threshold-selection procedure across models.
- Tune models using validation data only; keep the test set untouched until final evaluation.
- Compare PR-AUC, recall at a fixed false-positive rate, confusion matrices and cost-sensitive metrics.
- Report hardware, package versions, training data size and preprocessing for every run.
- Add reproducible experiment configuration and automated tests.
- Update the Streamlit dashboard to load all five model results from a single validated results file.
- Evaluate the models on newer or independently collected transaction data.

## Disclaimer

This project is for academic and experimental purposes. Its metrics do not establish that a model is suitable for deployment in a real payment system. A production fraud-detection system would require additional validation, monitoring, security controls, explainability, threshold governance and review of false-positive/false-negative costs.
