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

This installs: numpy, pandas, scipy, scikit-learn, imbalanced-learn, boruta, xgboost, matplotlib, seaborn.

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

This runs the full pipeline (NZV filtering → correlation filtering → statistical
tests → Boruta → train/test split → SMOTE → GridSearchCV → evaluation) and
saves all outputs to the `outputs/` directory.

The pipeline takes several minutes, mainly due to Boruta (200 iterations) and
the SVM/XGBoost grid searches.

### Step 4 — View Results

All outputs are saved in `outputs/`:

| File | Description |
|------|-------------|
| `model_comparison_results.csv` | Accuracy, Sensitivity, Specificity, F1, AUC for each model |
| `statistical_test_results.csv` | p-values and test type for all 142+ features |
| `plot_sig_features_boxplot.png` | Boxplots of significant features (ASD vs Non-ASD) |
| `plot_rf_importance.png` | Random Forest feature importance (Gini) |
| `plot_roc_all_models.png` | ROC curves for all models on the hold-out test set |
| `plot_model_comparison_bar.png` | Grouped bar chart of all metrics |
| `plot_cv_dotplot.png` | CV AUC distributions (5-fold × 3 repeats, median + 95% CI) |

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

The published thesis results (hold-out test set, seed 42):

| Model | AUC (thesis) |
|-------|-------------|
| SVM (RBF) | **0.935** |
| Random Forest | — |
| Logistic Regression | — |

Your results may differ slightly from the thesis due to differences between
R's `caret`/`glmnet` and Python's `scikit-learn` in their optimisation
internals, even with the same grid and seed.

### Stochasticity

All random operations are seeded to `42`:
- `train_test_split(random_state=42)`
- `StandardScaler` — deterministic
- `SMOTE(random_state=42)`
- `RepeatedStratifiedKFold(random_state=42)`
- `GridSearchCV` — deterministic given a fixed CV splitter
- `RandomForestClassifier(random_state=42)`
- `SVC(random_state=42)`
- `BorutaPy(random_state=42)`
- `XGBClassifier(random_state=42)`

Results should be **fully reproducible** across runs on the same machine and
Python/library version.

---

## Custom Paths

You can override default file locations with command-line arguments:

```bash
# Feature extraction with custom input/output paths
python pipeline/feature_extraction.py \
    --side "Side Dataset 25KP New GC.csv" \
    --front "Front Dataset 25KP New GC.csv" \
    --out outputs/my_features.csv

# Verify against a different reference CSV
python pipeline/feature_extraction.py \
    --verify \
    --reference gait_features_rich.csv

# ML pipeline with a custom feature CSV
python pipeline/ml_pipeline.py \
    --features outputs/my_features.csv \
    --out my_outputs/
```

---

## Notes

- The `outputs/` directory is listed in `.gitignore` and is not committed to the repository. Regenerate it by running the pipeline.
- The `venv/` directory is also excluded from git.
- For details on methodological discrepancies between the thesis and the code, see `FINDINGS.md`.
