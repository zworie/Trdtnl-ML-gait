# Experiment Guide

Step-by-step instructions for running the Python gait analysis pipeline locally using VS Code.

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
└── FINDINGS.md                        ← audit notes and discrepancies
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

This installs: numpy, pandas, scipy, scikit-learn, imbalanced-learn, boruta, xgboost, optuna, matplotlib, seaborn.

---

## Running the Pipeline

All commands below are run from the **repository root** with the `venv` active.

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
to confirm they match.  Nothing is written to disk during verification.

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

If any mismatches appear, the columns and rows involved are reported.

### Step 3 — Run the ML Pipeline

```bash
python pipeline/ml_pipeline.py
```

This runs the full pipeline and saves all outputs to `outputs/`.
**Expected run time: 20–40 minutes** (RF + XGBoost inside 30 outer folds are
the bottleneck).  For a quick test run use:

```bash
python pipeline/ml_pipeline.py --n-trials 20 --n-repeats 1
```

#### What the improved pipeline does

The pipeline now uses **nested cross-validation** with **Optuna** instead of a
single 70/30 split with grid search:

| Component | Original | Improved |
|-----------|----------|----------|
| Evaluation | Single 70/30 split → 1 test result | Outer 10-fold × 3 repeats → **30 test results → mean ± std** |
| HPO | GridSearchCV (fixed grid) | **Optuna TPE** (continuous log-space, 50 trials) |
| SMOTE | Applied before inner CV (mild leakage) | **Inside each fold via ImbPipeline** (no leakage) |
| SVM AUC | Could be < 0.5 from Platt inversion | **Auto-corrected inside Optuna and evaluation** |

#### Anchor to thesis features (skip Boruta)

```bash
python pipeline/ml_pipeline.py \
  --selected-features \
    LHip_min RKnee_skew RAnkle_skew LAnkle_kurt \
    LDP_cv TrunkY_max CoM_Y_min StepLength
```

#### All CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--features` | `gait_features_rich.csv` | Input feature CSV |
| `--out` | `outputs/` | Output directory |
| `--selected-features` | *(run Boruta)* | Skip Boruta; use these features |
| `--n-trials` | `50` | Optuna trials per model per fold |
| `--n-folds` | `10` | Outer CV folds |
| `--n-repeats` | `3` | Outer CV repeats (total folds = n-folds × n-repeats) |

### Step 4 — View Results

All outputs are saved in `outputs/`:

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | **mean ± std** of AUC, Accuracy, Sensitivity, Specificity, F1, Precision across all outer folds |
| `per_fold_results.csv` | Raw metrics for every individual (fold, model) combination |
| `statistical_test_results.csv` | p-values and test type for all retained features |
| `plot_sig_features_boxplot.png` | Boxplots of significant features (ASD vs Non-ASD) |
| `plot_rf_importance.png` | RF feature importance from the fold with the highest RF AUC |
| `plot_roc_all_models.png` | Mean ROC curves (bold) + individual fold curves (light gray) |
| `plot_model_comparison_bar.png` | Bar chart of mean AUC per model with ± std error bars |
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

### Expected range

The published thesis results (R code, single test set):

| Model | AUC (thesis R) |
|-------|----------------|
| SVM (RBF) | **0.935** |

The improved Python pipeline uses nested CV with a different evaluation
framework, so direct numerical comparison with the thesis is not meaningful.
Expect mean AUC values in the 0.70–0.95 range with std 0.05–0.15 across folds,
reflecting genuine variability on this 72-subject dataset.

### Stochasticity and reproducibility

All random operations are seeded.  Outer fold seeds are `42`.  Optuna studies
use `42 + fold_index` so each fold's hyperparameter search is independently
reproducible.  Results are fully reproducible across runs on the same machine
and library version.

---

## Custom Paths

```bash
# Feature extraction with custom input/output paths
python pipeline/feature_extraction.py \
    --side "Side Dataset 25KP New GC.csv" \
    --front "Front Dataset 25KP New GC.csv" \
    --out outputs/my_features.csv

# Verify against the ground-truth CSV
python pipeline/feature_extraction.py --verify

# ML pipeline — fast test run (10 folds, 20 Optuna trials)
python pipeline/ml_pipeline.py --n-trials 20 --n-repeats 1

# ML pipeline with thesis features and full settings
python pipeline/ml_pipeline.py \
    --selected-features LHip_min RKnee_skew RAnkle_skew LAnkle_kurt \
                        LDP_cv TrunkY_max CoM_Y_min StepLength \
    --n-trials 50 --n-folds 10 --n-repeats 3
```

---

## Notes

- The `outputs/` directory is listed in `.gitignore` and is not committed to the repository. Regenerate it by running the pipeline.
- The `venv/` directory is also excluded from git.
- For details on methodological discrepancies between the thesis and the code, see `FINDINGS.md`.
