# Experiment Guide

Step-by-step instructions for running the Python gait analysis pipeline locally using VS Code.

---

## What This Pipeline Does

This Python pipeline classifies ASD vs Non-ASD gait from 72 subjects using traditional ML.
It is a methodologically improved conversion of the original `thesis.R`.

### Key improvements over the original R code

| Component | Original (`thesis.R`) | Python pipeline |
|-----------|----------------------|-----------------|
| Evaluation | Single 70/30 split → 1 result | **Nested 5×5 stratified CV** → all 72 subjects evaluated as test → mean ± std |
| Stratification | Single stratified split | **Every outer and inner fold is stratified** — class ratio preserved throughout |
| Models | LR, RF, SVM | **LR, RF, SVM, LDA, XGBoost + Soft-Vote + Majority-Vote Ensembles** |
| HPO — LR | GridSearchCV | **GridSearchCV** (5×5 = 25 grid points, C × l1\_ratio) |
| HPO — SVM | GridSearchCV | **GridSearchCV** (5×4 = 20 grid points, C × gamma) |
| HPO — RF | GridSearchCV | **Optuna TPE** (3 params, 50 trials) or **GridSearchCV** via `--hpo grid` |
| HPO — XGBoost | GridSearchCV | **Optuna TPE** (7 params, 50 trials) or **GridSearchCV** via `--hpo grid` |
| HPO — LDA | — | **None** — Ledoit-Wolf shrinkage is fully automatic |
| Class imbalance | SMOTE before CV (mild leakage) | **`class_weight='balanced'`** (default) or **SMOTE inside CV** via `--smote` |
| Decision threshold | Fixed 0.5 | **Youden's J** optimised per fold on training data |
| Probability calibration | Platt on training data (SVM) | **`CalibratedClassifierCV`** (held-out folds) for RF (isotonic) and SVM (sigmoid) |

### Models

| Model | HPO method | Notes |
|-------|------------|-------|
| **Logistic Regression** (Elastic Net) | GridSearchCV | `class_weight='balanced'`; L1+L2 regularisation |
| **Random Forest** | Optuna TPE (default) or GridSearch | `class_weight='balanced'`; 200 trees during search, 1 000 for final fit; isotonic calibration |
| **SVM** (RBF kernel) | GridSearchCV | `class_weight='balanced'`; sigmoid (Platt) calibration |
| **LDA** (Ledoit-Wolf) | None | Automatic covariance shrinkage; estimated class priors; ideal for small N |
| **XGBoost** | Optuna TPE (default) or GridSearch | `scale_pos_weight = n_neg/n_pos` per fold; extension beyond published thesis |
| **Soft-Vote Ensemble** | — | Arithmetic mean of 5 AUC-corrected probability arrays |
| **Majority-Vote Ensemble** | — | Each base model casts a 0/1 vote; ASD if strictly > half agree |

### How the nested CV works

The outer loop splits all 72 subjects into 5 **stratified** folds (~14 test subjects each).
Every subject appears as a test subject exactly once, and the ASD/Non-ASD ratio is
preserved in every fold.

For each outer fold, the ~58 training subjects go into the inner loop:

- **LR and SVM**: `GridSearchCV` with a 5-fold stratified inner CV.
- **RF and XGB**: Optuna TPE samples 50 configurations via 5-fold stratified inner CV
  (or `GridSearchCV` when `--hpo grid` is set).
- **LDA**: Fit directly — no tuning needed.

After HPO, a **Youden's J threshold** is found on the outer training probabilities
and applied to test-set predictions (replaces the fixed 0.5 cut-off).

Both ensemble models are built from the five base-model predictions on each outer test fold.

---

## Prerequisites

- Python 3.9 or later
- All files in the repository root (see File Layout below)

---

## File Layout

```
Trdtnl-ML-gait/
├── Front Dataset 25KP New GC.csv      ← raw front-view keypoints
├── Side Dataset 25KP New GC.csv       ← raw side-view keypoints
├── gait_features_rich.csv             ← ground-truth feature CSV (do not delete)
├── extract_rich_features.py           ← original script (reference only)
├── thesis.R                           ← original R pipeline (reference only)
├── pipeline/
│   ├── feature_extraction.py          ← fixed feature extractor (use this)
│   └── ml_pipeline.py                 ← Python ML pipeline (use this)
├── requirements.txt
├── EXPERIMENT_GUIDE.md                ← this file
├── FINDINGS.md                        ← audit notes and discrepancies
└── TECHNICAL_DOCUMENTATION.md        ← full pipeline documentation
```

---

## Setup

### 1. Open the repo in VS Code

Open the `Trdtnl-ML-gait` folder in VS Code.

### 2. Create and activate the virtual environment

Open the VS Code integrated terminal (**Terminal → New Terminal**) and run:

```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# Activate it (macOS / Linux)
source venv/bin/activate
```

You should see `(venv)` at the start of your prompt once activated.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs: numpy, pandas, scipy, scikit-learn, boruta, imbalanced-learn, xgboost, optuna, matplotlib, seaborn.

> **Updating an existing environment:** Run `pip install -r requirements.txt` again.
> New packages will install and old ones remain harmless.

---

## Running the Pipeline

All commands below are run from the **repository root** (the `Trdtnl-ML-gait` folder)
with the `venv` active.

> **Windows note:** Both `python pipeline/ml_pipeline.py` and
> `python pipeline\ml_pipeline.py` work in the VS Code terminal.

---

### Step 1 — Feature Extraction

Run the fixed feature extraction script.
**The original `gait_features_rich.csv` is never overwritten.**
Output goes to `outputs/gait_features_extracted.csv`.

```bash
python pipeline/feature_extraction.py
```

Expected output:
```
Saved: outputs/gait_features_extracted.csv
Shape: (72, 165)
Class distribution:
  ASD       30
  Non-ASD   42
Feature count: 163
```

### Step 2 — Verify Against Ground-Truth CSV (optional)

Compare the extraction output against the existing `gait_features_rich.csv`
to confirm they match. Nothing is written to disk during verification.

```bash
python pipeline/feature_extraction.py --verify
```

Expected output:
```
Running extraction against reference: gait_features_rich.csv
  Shape OK: (72, 165)
  Columns OK
  Values OK across 163 numeric columns (atol=1e-4)

RESULT: PASS — extracted features match the reference CSV.
```

### Step 3 — Run the ML Pipeline

```bash
python pipeline/ml_pipeline.py
```

This runs the full pipeline and saves all outputs to `outputs/`.
**Expected run time: 10–20 minutes** with default settings (RF and XGBoost
Optuna searches are the bottleneck).

For a quick test run (completes in ~1–2 minutes):

```bash
python pipeline/ml_pipeline.py --n-trials 10 --n-outer 3 --n-inner 3
```

---

### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--features` | `gait_features_rich.csv` | Input feature CSV |
| `--out DIR` | `outputs/` | Directory for all output files and plots |
| `--selected-features` | *(run Boruta)* | Skip Boruta; use these exact feature names |
| `--n-outer N` | `5` | Number of outer CV folds |
| `--n-inner N` | `5` | Number of inner CV folds for HPO |
| `--n-trials N` | `50` | Optuna trials per model per outer fold (RF and XGB only; ignored when `--hpo grid`) |
| `--smote` | off | Apply SMOTE oversampling inside each inner CV fold (see below) |
| `--hpo {optuna,grid}` | `optuna` | HPO method for RF and XGB (see below) |
| `--log FILE` | off | Write all terminal output to FILE in addition to the console |

---

### Option: `--smote` — SMOTE oversampling

By default the pipeline handles class imbalance via `class_weight='balanced'` (LR/SVM/RF)
and `scale_pos_weight` (XGB). Pass `--smote` to use SMOTE instead:

```bash
python pipeline/ml_pipeline.py --smote
```

When `--smote` is active:
- `SMOTE` is prepended as a pipeline step and runs **inside every inner CV fold**
  so no synthetic samples from the test fold ever enter training.
- `class_weight` is set to `None` for LR/SVM/RF and `scale_pos_weight=1.0` for XGB
  to avoid double-correcting for class imbalance.
- Requires `imbalanced-learn` (already in `requirements.txt`).

---

### Option: `--hpo {optuna,grid}` — HPO method for RF and XGB

```bash
# Default: Optuna TPE (flexible, samples continuous param space)
python pipeline/ml_pipeline.py --hpo optuna

# Grid search: faster, uses a predefined coarse grid
python pipeline/ml_pipeline.py --hpo grid
```

| Setting | RF | XGB | LR | SVM | LDA |
|---------|-----|-----|-----|-----|-----|
| `optuna` (default) | Optuna TPE, 50 trials | Optuna TPE, 50 trials | GridSearch | GridSearch | none |
| `grid` | GridSearchCV (coarse grid) | GridSearchCV (coarse grid) | GridSearch | GridSearch | none |

`--hpo grid` is faster but less thorough. It is useful for quick experiments or
when Optuna is slow on your machine.

---

### Option: `--log FILE` — save terminal output to a file

```bash
python pipeline/ml_pipeline.py --log outputs/run.log
```

All printed output is written to both the console and `run.log` simultaneously.
Useful for recording results from long runs.

---

### Option: `--out DIR` — custom output directory

```bash
python pipeline/ml_pipeline.py --out results/experiment_smote
```

The directory is created if it does not exist. Use different `--out` values to
keep results from separate experiments side-by-side without overwriting.

---

### Option: `--n-outer` / `--n-inner` — fold counts

```bash
# Full run with more outer folds for better variance estimate
python pipeline/ml_pipeline.py --n-outer 10 --n-inner 5

# Quick 3-fold run for debugging
python pipeline/ml_pipeline.py --n-outer 3 --n-inner 3 --n-trials 5
```

---

### Anchor to thesis features (skip Boruta)

To replicate the 8 features confirmed in the published thesis:

```bash
python pipeline/ml_pipeline.py \
  --selected-features \
    LHip_min RKnee_skew RAnkle_skew LAnkle_kurt \
    LDP_cv TrunkY_max CoM_Y_min StepLength
```

---

### Step 4 — View Results

All outputs are saved in `outputs/` (or the directory specified by `--out`):

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | **mean ± std** of AUC, Accuracy, Sensitivity, Specificity, F1, Precision across all outer folds — one row per model (7 rows) |
| `per_fold_results.csv` | Raw metrics for every (fold × model) combination — 5 folds × 7 models = 35 rows |
| `statistical_test_results.csv` | p-values and test type for all retained features |
| `plot_sig_features_boxplot.png` | Boxplots of statistically significant features (ASD vs Non-ASD) |
| `plot_rf_importance.png` | RF feature importance (Gini) from the fold with the highest RF AUC |
| `plot_roc_all_models.png` | Mean ROC curve per model (bold coloured line) + individual fold curves (light grey) |
| `plot_model_comparison_bar.png` | Bar chart of mean AUC per model with ± 1 std error bars |
| `plot_auc_distributions.png` | Box plots of per-fold AUC distributions for each model |

---

## Interpreting Results

### Metrics

| Metric | What it measures |
|--------|-----------------|
| **AUC** | Area under the ROC curve. Primary metric. Measures discrimination regardless of threshold. 1.0 = perfect, 0.5 = random. |
| **Accuracy** | Fraction of all subjects classified correctly. Can be misleading with class imbalance. |
| **Sensitivity** | True positive rate for ASD (= recall). High sensitivity → fewer missed ASD cases. |
| **Specificity** | True positive rate for Non-ASD. High specificity → fewer false alarms. |
| **F1** | Harmonic mean of precision and recall for the ASD class. Balances both. |
| **Precision** | Fraction of predicted-ASD subjects that are truly ASD. |

### Expected range

The published thesis result (R code, single hold-out test set):

| Model | AUC (thesis R) |
|-------|----------------|
| SVM (RBF) | **0.935** |

The Python pipeline uses a different evaluation framework (nested CV vs single 70/30 split)
and different implementations, so direct numerical comparison is not meaningful.
Expect mean AUC values in the **0.70–0.95 range** with std 0.05–0.15, reflecting genuine
variability on this 72-subject dataset.

### Stochasticity and reproducibility

The outer CV uses `random_state=42`. Each inner CV uses `random_state=fold_idx` (0–4)
so every fold's hyperparameter search is independently seeded.
Optuna studies are seeded from `fold_idx`. Results are fully reproducible
across runs on the same machine and library version.

---

## Example Recipes

```bash
# Default full run
python pipeline/ml_pipeline.py

# Quick 3-fold smoke test (~ 1 minute)
python pipeline/ml_pipeline.py --n-outer 3 --n-inner 3 --n-trials 5

# Thesis features, log output, custom output folder
python pipeline/ml_pipeline.py \
    --selected-features LHip_min RKnee_skew RAnkle_skew LAnkle_kurt \
                        LDP_cv TrunkY_max CoM_Y_min StepLength \
    --out outputs/thesis_features \
    --log outputs/thesis_features/run.log

# SMOTE + grid search (faster run, SMOTE-balanced training)
python pipeline/ml_pipeline.py \
    --smote --hpo grid \
    --out outputs/smote_grid \
    --log outputs/smote_grid/run.log

# SMOTE + Optuna with more trials
python pipeline/ml_pipeline.py \
    --smote --n-trials 100 \
    --out outputs/smote_optuna

# Compare SMOTE vs no-SMOTE side by side
python pipeline/ml_pipeline.py --out outputs/no_smote
python pipeline/ml_pipeline.py --smote --out outputs/with_smote

# Feature extraction with custom paths
python pipeline/feature_extraction.py \
    --side "Side Dataset 25KP New GC.csv" \
    --front "Front Dataset 25KP New GC.csv" \
    --out outputs/my_features.csv
```

---

## Notes

- The `outputs/` directory is listed in `.gitignore` and is not committed to the repository.
  Regenerate it by running the pipeline. Use `--out` to save to a different folder.
- The `venv/` directory is also excluded from git.
- Boruta feature selection runs on the full 72-subject dataset before the CV loop (a
  data-leakage limitation inherited from the original R code; documented in `FINDINGS.md`).
  Use `--selected-features` to bypass Boruta entirely and anchor the pipeline to the 8
  thesis-confirmed features.
- For full technical details (algorithm choices, all parameters, results table, differences
  from thesis), see `TECHNICAL_DOCUMENTATION.md`.
- For methodological discrepancies between the thesis and the code, see `FINDINGS.md`.
