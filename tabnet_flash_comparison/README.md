# Person 3: PyTorch TabNet, Lightning Flash and the 5-framework comparison

**Note on the split:** the original shared train/test files were not available to us, so the split recipe and row counts below were reconstructed from `eda.ipynb` (80/20, stratified, seed 42). The counts match the XGBoost notebook, but we could not check that the rows are byte-for-byte the same as the teammates' files.

Dataset: online payment transactions (about 6.36 million rows, 8,213 frauds). Target: `isFraud`, treated as a number (regression), as the task sheet asks.
Everything below was run on a split rebuilt (see the note above) with the steps in `eda.ipynb`:
train 5,090,096 rows (6,570 fraud), test 1,272,524 rows (1,643 fraud). `prep.py` rebuilds it from `new_data.csv`.

## Files
| File | What it is |
|---|---|
| `Person3_TabNet_Flash_comparison.pptx` | The slides for our section (6 slides). |
| `requirements.txt` | Exact versions we tested (Python 3.10.12). |
| `prep.py` | Rebuilds the train/test split from `new_data.csv` (same steps as `eda.ipynb`). |
| `common.py` | Shared preprocessing (signed log on money columns) and the metric code (`sklearn.metrics` RMSE, R2, MAE, plus extras). |
| `run_tabnet.py`, `infer_ckpt.py` | TabNetRegressor training script and the script that scores the saved checkpoint on the full test set. |
| `flash_train.py`, `flash_score.py` | Lightning Flash TabularRegressor training (on a sample) and scoring on the full test set. |
| `make_comparison.py` | Builds `comparison_table.md`, `comparison_table.json`, `comparison_chart.png`. |
| `tabnet_results.json`, `flash_results.json` | Final metrics of the two models. |
| `tabnet_epoch4_best.zip`, `flash_checkpoint_run2.ckpt` | Saved models used for the results. `flash_checkpoint_run1.ckpt` and `flash_results_run1_4ep_bs8192.json` are the first, weaker Flash run (kept for honesty). |
| logs `*_log*.txt` | Training logs. |

## How to reproduce
```
pip install -r requirements.txt      # plain lightning-flash; the [tabular] extra failed for us
python prep.py --csv new_data.csv --out .                 # writes train32.parquet, test32.parquet
python run_tabnet.py --data . --out out                   # TabNet training and test scoring (about 6 min for 4 epochs on 1 CPU)
python flash_train.py --rows 750000 --epochs 8 --batch 1024 --data . --out out
python flash_score.py --ckpt out/flash_final.ckpt --info out/flash_train_info.json --data . --out out
python infer_ckpt.py tabnet_epoch4_best.zip test          # re-scores the saved TabNet checkpoint
python make_comparison.py                                 # needs ../model_comparison.csv from the repo
```
`run_tabnet.py` is a clean rewrite. The reported TabNet numbers came from an earlier version of the script that was interrupted after epoch 6, so a new run may stop at a different epoch and give slightly different numbers. The saved checkpoint and the logs are the record of the reported result. Run `python <script> --help` for all flags. Scripts were syntax-checked and smoke-tested on tiny runs (20k to 50k rows, 1 epoch), without repeating the full training.

## Methodology
Both models use the same 13 features (step, amount, four balances, the two balance-error columns, five one-hot transaction types), the target is not a feature. Money columns got a signed log transform and features were standardised with training-set statistics.
Test rows were never used to fit either model. TabNet's checkpoint was chosen on validation rows only and then scored on the test set. Flash is different: its run 1 test score was seen before the settings of run 2 were chosen (details below), so Flash's test result was not fully blind.

**PyTorch TabNet** (`pytorch-tabnet`, `TabNetRegressor`): default network (n_d = n_a = 8, 3 steps), MSE loss, Adam lr 0.02, batch 8192, virtual batch 512, seed 42. It was fitted on 4,890,096 rows: the 5,090,096 training rows minus the first 200,000 rows, which were kept aside as validation (the 200,000 rows were not used for fitting, but the feature mean and standard deviation were computed on all 5,090,096 training rows). Training was planned for up to 8 epochs but the machine restarted after epoch 6 of the first run. Validation RMSE per epoch was 0.00566, 0.00811, 0.00833, 0.00356, 0.00821, 0.01119. The saved checkpoint is the best one by validation (epoch 4 of 6; the checkpoint file has no epoch label, we identified it by re-scoring it on the validation rows and matching 0.0035619). It took about 6 minutes for 4 epochs on 1 CPU.

**Lightning Flash** (`flash` 0.8.2, `TabularRegressor`, default TabNet backbone): two attempts to run Flash on the full 5.09M training rows were killed by the machine for running out of memory (about 2 GB available). Probes on a smaller scale: 500,000 rows peaked at about 1.5 GB and trained; 1,500,000 rows reached 1.86 GB just after data setup and did not finish training within the 110 s probe. These are observations from this machine only, not a general limit of the library. So Flash was trained on a **750,000-row sample of the training file** (we measured this, see below), drawn proportionally stratified on `isFraud` (968 fraud rows, 0.129%, the same fraud rate as the full training file; seed 42). That is about 15% of the training rows (about 13% are actually used for fitting), so the two are not trained on identical data. Flash's 10% validation split of the sample leaves about 675,000 rows for fitting and 75,000 for validation (from `val_split=0.1`; the batch count of 82 per epoch at batch size 8192 fits this). It was still scored on the **full** test set (1,272,524 rows).
Two Flash runs were made. Run 1 used defaults (4 epochs, batch 8192) and did badly on the test set (RMSE 0.0409, R2 -0.30), a possible explanation is too few update steps (4 epochs of 82 batches); we did not test this. Run 2 (8 epochs, batch 1024, about 4 minutes) is the reported result; the change was made after seeing run 1's test score, so its test result is not a completely blind number. Validation MSE of run 2 was 0.000245 (RMSE about 0.0157), which agrees with its test RMSE.

## Results (full test set, 1,272,524 rows)
| Framework | RMSE | R2 | MAE | Train time (s) |
|---|---|---|---|---|
| XGBoost | 0.002080 | 0.996643 | 0.000022 | 65.5 |
| Keras MLP | 0.004365 | 0.985223 | 0.000317 | 60.9 |
| fastai Tabular | 0.011738 | 0.893148 | 0.009665 | 11.1 |
| PyTorch TabNet | 0.002998 | 0.993029 | 0.000605 | 366 |
| Lightning Flash (750k-row sample) | 0.015972 | 0.802167 | 0.002595 | 246.5 |

Chart: `comparison_chart.png`. Extra numbers at a 0.5 cut-off: TabNet precision 0.9964, recall 0.9976, F1 0.9970 (ROC AUC 0.9996). Flash precision 0.915, recall 0.930, F1 0.922 (ROC AUC 0.975).

**Read the table with care.** The first three rows come from the teammates' `model_comparison.csv` and were not re-run. They are not perfectly like for like: XGBoost, Keras and fastai were trained as classifiers (class weights, log-scaled inputs for Keras/fastai) and their predicted fraud probabilities were scored with the regression metrics, fastai used a 1,000,000-row training sample, and training times came from different machines and settings. TabNet and Flash are trained as MSE regressors with no class weights. TabNet was fitted on 4.89M rows, Flash on about 675k rows of a 750k sample. Training times are not comparable. `keras_fastai/README.md` still shows an older run (R2 about 0.61); the table uses the newer `model_comparison.csv` numbers.

## Which was simplest, most transparent, most effective
Our assessment, not a measured result (nothing here was timed or tested in a controlled way for these points):
- **Simplest:** XGBoost, Keras and Flash each looked short to set up. Flash needs only a data module and one `TabularRegressor` call, but installing it was awkward (the `[tabular]` extra failed, plain `lightning-flash` worked) and it pulled in an older PyTorch. TabNet is also a few lines, but needed more care with memory and had no easy per-epoch checkpoint.
- **Most transparent (our view):** XGBoost (feature importance, trees). TabNet is the most transparent of the neural networks because it has attention masks that show which features each step used; we did not plot them here. Flash hides most choices behind defaults.
- **Most effective:** XGBoost had the lowest RMSE and highest R2, then TabNet, then Keras. Flash and fastai were the weakest, but Flash was fitted on about 13% of the training rows and fastai on a 1M sample, so this does not show those libraries are worse in general.

## Why results differ
Different training data amounts, different loss and class weighting, different preprocessing, and different tuning effort (TabNet trained about 6 minutes, results moved a lot from epoch to epoch because fraud is only 0.13% of rows). Because almost every row is 0, even a model that predicts nearly 0 everywhere gets a very small RMSE and a high R2 is driven by the few fraud rows, so small differences in how well rare frauds are caught change R2 a lot.

## Conclusion
On this split, XGBoost gave the best regression scores, with TabNet close behind (R2 0.993 vs 0.997) and Keras third. Our recommendation is XGBoost for this task: it has the best scores in the table, and in our view it is the easiest to explain. (The teammate's log reports 65.5 s to train it, but that was a different setup from ours, so this is not a controlled speed comparison.) This is our assessment, not measured superiority, because the rows are not like for like. TabNet is a reasonable neural alternative if feature attention is wanted. Flash worked but was limited here by memory and a sample-based training run, and a full-data Flash run on a larger machine is still needed for a fair comparison.

## Limits of this work
- TabNet checkpoint is epoch 4 identified by matching validation RMSE, not by metadata; training did not reach its planned epoch count because of a machine restart.
- Flash trained on a 750k-row stratified sample, not the full training set; run 2 settings were chosen after seeing run 1's test result.
- XGBoost, Keras and fastai rows are the teammates' numbers, not re-run, and are not strictly like for like.
- Row-for-row equality with Person 1's `train.pkl`/`test.pkl` is not verified (those files are not in the repo); the split recipe and row/fraud counts match.
