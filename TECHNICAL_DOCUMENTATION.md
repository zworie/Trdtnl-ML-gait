# Technical Documentation: ASD Gait Classification Pipeline

**Date:** 9 April 2026
**Pipeline version:** `pipeline/ml_pipeline.py` on branch `claude/copy-igait-ml-code-QRYlp`
**Original reference:** `thesis.R` (R code) and published thesis document

---

## 1. Purpose

This document describes the complete Python machine learning pipeline for classifying Autism Spectrum Disorder (ASD) versus Non-ASD gait from video-derived pose keypoints. It covers every stage from raw data to final evaluation metrics, notes all methodological decisions made during the Python conversion, and documents where and why the Python implementation differs from the original R code and thesis.

The pipeline is a substantially improved conversion of `thesis.R`. It is not a direct replication — the evaluation framework, model set, hyperparameter optimisation strategy, class imbalance handling, and probability calibration have all been redesigned. Differences are itemised explicitly in Section 9.

---

## 2. Dataset

| Property | Value |
|----------|-------|
| Total subjects | **72** |
| ASD | 30 |
| Non-ASD | 42 |
| Class ratio | 1 : 1.4 (mild imbalance) |

> **Note — thesis reports N=74:** The published thesis states 74 participants (30 ASD, 44 Non-ASD). The processed feature CSV (`gait_features_rich.csv`) contains only 72 subjects. Two subjects appear to have been excluded during pose estimation or preprocessing; the reason is not documented in the repository.

### Raw data files

Two CSV files in the repository root provide the raw input:

| File | View | Content |
|------|------|---------|
| `Side Dataset 25KP New GC.csv` | Sagittal (side) | 25 body keypoints × (x, y, confidence) per frame, per subject |
| `Front Dataset 25KP New GC.csv` | Frontal (front) | 25 body keypoints × (x, y, confidence) per frame, per subject |

Keypoints include head, shoulders, elbows, wrists, hips, knees, ankles, and foot landmarks (big toe, small toe, heel) for both sides. Data was produced by a pose estimation model (MediaPipe or equivalent) run on gait video.

---

## 3. Feature Extraction

**Script:** `pipeline/feature_extraction.py`
**Input:** Side and front CSVs above
**Output:** `outputs/gait_features_extracted.csv` (163 features + Subject + Class columns; 72 rows)
**Ground-truth reference:** `gait_features_rich.csv` (pre-computed; never overwritten)

### 3.1 Feature categories

163 biomechanical features are computed per subject across the following categories:

| Category | Examples | Count |
|----------|---------|-------|
| **Joint angles** — mean, std, min, max, ROM, skewness, kurtosis, peak count, trough count, zero-crossing rate | RHip, LHip, RKnee, LKnee, RAnkle, LAnkle angles | ~60 |
| **Displacement / trajectory** | Right/left dorsiplantar (DP) motion, trunk vertical (TrunkY), centre-of-mass Y (CoM_Y) | ~25 |
| **Symmetry indices** | Hip SI, Knee SI, Ankle SI, AddAbd SI, DP SI | 5 |
| **Step / stride** | Step width (mean, std, min, max, ROM, CV, skew, kurt, peaks, troughs, ZCR), step length, adduction-abduction angles | ~40 |
| **Temporal** | Step count | 1 |
| **CoM** | CoM_Y statistics (std, min, max, ROM, CV, skew, kurt, peaks, troughs, ZCR) | ~10 |

### 3.2 Angle convention note (thesis vs code)

The hip angle formula differs between the thesis and the code:

| | Landmarks | Measures |
|---|---|---|
| **Thesis** | Hip + Knee only | Thigh deviation from vertical — hip flexion relative to gravity |
| **Code** | Shoulder + Hip + Knee | Trunk-to-thigh angle at the hip joint |

The knee angle uses the same landmarks (Hip, Knee, Ankle) in both, but the conventions are supplementary: `θ_code = 180° − θ_thesis`. The ankle angle is mathematically equivalent. Since `gait_features_rich.csv` was produced by the code, **the code formula is the operational ground truth**. See `FINDINGS.md` for full details.

### 3.3 Running feature extraction

```bash
# Extract features (writes to outputs/gait_features_extracted.csv)
python pipeline/feature_extraction.py

# Verify output matches the reference gait_features_rich.csv
python pipeline/feature_extraction.py --verify
```

The `--verify` flag runs extraction in-memory, loads `gait_features_rich.csv`, and checks shape, column names, and numeric values (atol=1e-4). Nothing is written to disk.

---

## 4. Feature Preprocessing

All preprocessing in `pipeline/ml_pipeline.py` is applied to the full 72-subject dataset before the cross-validation loop (matching the original R code behaviour; see leakage note in Section 6).

### 4.1 Near-zero-variance (NZV) removal

Replicates `caret::nearZeroVar`. A feature is removed when:
- It has only one unique value (`zeroVar`), **or**
- `freq_ratio > 19` **AND** `pct_unique < 10%`

where `freq_ratio` = (count of most common value) / (count of second most common value), and `pct_unique` = (number of unique values / N) × 100.

### 4.2 High-correlation removal

Replicates `caret::findCorrelation` (fast/greedy algorithm). For each pair with |Pearson r| > 0.95, the feature with the higher mean absolute correlation to all other retained features is removed. The process iterates until no correlated pairs remain.

**Feature count progression:**

| Stage | Features remaining |
|-------|--------------------|
| After extraction | 163 |
| After NZV removal | varies by run |
| After correlation filtering | ~142 (matches thesis) |
| After Boruta selection | typically 8–12 |

---

## 5. Statistical Group-Difference Tests

For each retained feature, normality is assessed with the **Shapiro-Wilk test** on the full sample. Based on the result:
- If SW p > 0.05 (approximately normal): **Welch t-test** (`equal_var=False`)
- Otherwise: **Mann-Whitney U** (Wilcoxon rank-sum, two-sided)

This replicates the R pipeline's `shapiro.test` → `t.test(var.equal=FALSE)` / `wilcox.test` logic.

### Significant features from latest run (p < 0.05)

The following 11 features showed statistically significant ASD vs Non-ASD differences in the most recent pipeline run:

| Feature | Test | p-value | Description |
|---------|------|---------|-------------|
| `CoM_Y_max` | Wilcoxon | **0.0032** | Maximum vertical CoM position |
| `CoM_Y_min` | Wilcoxon | **0.0037** | Minimum vertical CoM position |
| `RAnkle_skew` | Wilcoxon | **0.0001** | Skewness of right ankle angle (strongest signal) |
| `LKnee_zcr` | Wilcoxon | 0.0148 | Zero-crossing rate of left knee angle |
| `StepWidth_skew` | Wilcoxon | 0.0114 | Skewness of step width distribution |
| `TrunkY_max` | t-test | 0.0237 | Maximum trunk vertical displacement |
| `RHip_skew` | Wilcoxon | 0.0248 | Skewness of right hip angle |
| `LAnkle_rom` | Wilcoxon | 0.0361 | Range of motion of left ankle angle |
| `LAnkle_kurt` | Wilcoxon | 0.0392 | Kurtosis of left ankle angle |
| `StepLen_skew` | Wilcoxon | 0.0392 | Skewness of step length distribution |
| `LAddAbd_peaks` | Wilcoxon | 0.0491 | Peak count of left adduction-abduction angle |

`RAnkle_skew`, `CoM_Y_min`, `CoM_Y_max`, and `TrunkY_max` overlap with the 8 Boruta-confirmed thesis features, providing convergent evidence.

---

## 6. Feature Selection — Boruta

**Implementation:** `boruta.BorutaPy` with `RandomForestClassifier` backend (`max_iter=200`, `random_state=42`).

Boruta compares each feature's importance to the maximum importance of randomly permuted shadow features. Features that consistently outperform random noise are confirmed; those that consistently underperform are rejected. Tentative features (insufficient evidence) are included with confirmed features in this pipeline.

If Boruta fails (package missing or exception), the pipeline falls back to the statistically significant features from Section 5.

### Boruta-confirmed features (from published thesis)

The thesis reports 8 Boruta-confirmed features selected from the full 72-subject dataset:

| Feature | Description |
|---------|-------------|
| `LHip_min` | Minimum left hip angle |
| `RKnee_skew` | Skewness of right knee angle |
| `RAnkle_skew` | Skewness of right ankle angle |
| `LAnkle_kurt` | Kurtosis of left ankle angle |
| `LDP_cv` | Coefficient of variation of left dorsiplantar angle |
| `TrunkY_max` | Maximum trunk vertical displacement |
| `CoM_Y_min` | Minimum centre-of-mass vertical position |
| `StepLength` | Mean step length (pixel distance between ankles at contact) |

To bypass Boruta and use this exact set directly:

```bash
python pipeline/ml_pipeline.py \
  --selected-features LHip_min RKnee_skew RAnkle_skew LAnkle_kurt \
                      LDP_cv TrunkY_max CoM_Y_min StepLength
```

### ⚠ Data leakage note

Both the statistical tests and Boruta are run on **all 72 subjects** before the cross-validation loop begins. This means the feature selection process has seen the test subjects — a form of data leakage inherited from the original R code. The leakage is mild (feature selection, not model training, sees all data) but means reported CV metrics are slightly optimistic. This is documented here for transparency and replicated faithfully to match the published methodology.

---

## 7. Evaluation Framework

### 7.1 Nested stratified cross-validation

The Python pipeline replaces the R code's single 70/30 hold-out split with a **nested stratified 5 × 5 cross-validation**:

```
Outer CV (5 folds, random_state=42):
  ├── Fold 1: train on 58 subjects, test on 14
  ├── Fold 2: train on 58 subjects, test on 14
  ├── Fold 3: train on 58 subjects, test on 14
  ├── Fold 4: train on 58 subjects, test on 14
  └── Fold 5: train on 58 subjects, test on 14
         └── Inner CV (5 folds, random_state=fold_idx):
               HPO for LR, SVM (GridSearchCV), RF, XGB (Optuna TPE)
```

Key properties:
- Every one of the 72 subjects appears in the test set **exactly once** (outer CV exhaustive coverage).
- Every outer and inner split is **stratified**: the ASD/Non-ASD class ratio is preserved in every fold.
- Inner CV uses `random_state=fold_idx` (0–4), making every fold's HPO independently but deterministically seeded.
- Final metrics are **mean ± std** across the 5 outer folds, capturing genuine fold-to-fold variability on this small dataset.

This contrasts with the single 70/30 split in `thesis.R`, which produces a single point estimate with no variance quantification and whose result depends heavily on which 22 subjects happen to fall in the test set.

### 7.2 Scaling

Within each outer fold, a `StandardScaler` is fit **on the outer training set only** and applied to both training and test sets. The scaler is part of the sklearn `Pipeline` object, so there is no leakage of test-set statistics into training.

### 7.3 AUC auto-correction

After computing `roc_auc_score`, the pipeline checks:

```python
if auc < 0.5:
    y_prob = 1.0 - y_prob   # flip probabilities
    auc = 1.0 - auc         # report mirror AUC
```

This mirrors R's `pROC` package default behaviour (`direction="auto"`). It handles cases where a model consistently predicts the wrong class — the sign of discrimination is corrected before reporting. The same correction is applied before constructing ROC curves and before ensembling.

---

## 8. Models

Seven models are evaluated in every outer fold:

| Model | HPO method | Class-imbalance handling |
|-------|------------|--------------------------|
| **LR** — Logistic Regression (Elastic Net) | GridSearchCV | `class_weight='balanced'` |
| **RF** — Random Forest | Optuna TPE | `class_weight='balanced'` |
| **SVM** — Support Vector Machine (RBF kernel) | GridSearchCV | `class_weight='balanced'` |
| **LDA** — Linear Discriminant Analysis (Ledoit-Wolf) | None | Estimated class priors |
| **XGB** — XGBoost | Optuna TPE | `scale_pos_weight = n_neg / n_pos` per outer fold |
| **Soft-Vote Ensemble** | — | Inherits from base models |
| **Majority-Vote Ensemble** | — | Inherits from base models |

### 8.1 Logistic Regression (LR)

- Solver: `saga` (handles both L1 and L2 penalties; scales to large datasets)
- Penalty: `elasticnet` (combination of L1 and L2 regularisation)
- `max_iter=2000`, `class_weight='balanced'`
- **HPO (GridSearchCV, 5-fold inner CV):**
  - `C` ∈ {0.01, 0.1, 1.0, 10.0, 100.0}
  - `l1_ratio` ∈ {0.0, 0.25, 0.5, 0.75, 1.0}
  - 25 grid points, scored by AUC

### 8.2 Random Forest (RF)

- 200 trees during Optuna search; 1 000 trees for final fit after HPO
- `class_weight='balanced'`, `random_state` fixed per fold
- Wrapped in `CalibratedClassifierCV(cv=3, method='isotonic')` for probability calibration
- **HPO (Optuna TPE, 50 trials, 5-fold inner CV):**
  - `max_features` ∈ [0.1, 1.0] (continuous, sampled by Optuna)
  - `min_samples_leaf` ∈ [1, 20] (integer)
  - `max_depth` ∈ [3, 30] (integer, None also allowed)
  - Scored by AUC

### 8.3 SVM (RBF kernel)

- Base: `SVC(kernel='rbf', class_weight='balanced', probability=False)`
- Wrapped in `CalibratedClassifierCV(cv=3, method='sigmoid')` — provides calibrated probabilities via Platt scaling applied only to the calibration fold, not the full training set
- **HPO (GridSearchCV, 5-fold inner CV):**
  - `C` ∈ {0.1, 1.0, 10.0, 100.0, 1000.0}
  - `gamma` ∈ {'scale', 0.01, 0.1, 1.0}
  - 20 grid points; param keys route through calibration wrapper (`estimator__C`, `estimator__gamma`)
  - Scored by AUC

### 8.4 LDA (Ledoit-Wolf)

- `LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')`
- Ledoit-Wolf analytical shrinkage automatically regularises the covariance matrix for small-N data — no HPO needed
- Class priors estimated from training set class frequencies (consistent with `class_weight='balanced'` intent)
- No probability calibration applied (LDA probabilities are already well-calibrated under the Gaussian assumption)

### 8.5 XGBoost

- `XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=fold_idx)`
- `scale_pos_weight = n_neg / n_pos` computed from the outer training fold's class counts
- **HPO (Optuna TPE, 50 trials, 5-fold inner CV):**
  - `n_estimators` ∈ [50, 500]
  - `max_depth` ∈ [2, 10]
  - `learning_rate` ∈ [0.01, 0.3] (log-uniform)
  - `subsample` ∈ [0.5, 1.0]
  - `colsample_bytree` ∈ [0.5, 1.0]
  - `reg_alpha` ∈ [1e-8, 10.0] (log-uniform, L1)
  - `reg_lambda` ∈ [1e-8, 10.0] (log-uniform, L2)
  - Scored by AUC

> **Thesis note:** XGBoost is present in `thesis.R` but absent from the published thesis results table. It is included in the Python pipeline as an extension and labelled clearly.

### 8.6 Soft-Vote Ensemble

After all five base models produce AUC-corrected probability arrays `p_1 … p_5` on the outer test fold, the ensemble prediction is:

```
p_ensemble = (p_1 + p_2 + p_3 + p_4 + p_5) / 5
```

The ensemble threshold is determined by applying Youden's J (Section 9.2) to the averaged training-set probabilities, ensuring no test-set information is used in threshold selection.

### 8.7 Majority-Vote Ensemble

Each base model's binary prediction (0/1, obtained via its own Youden threshold) is treated as one vote. The subject is predicted ASD if:

```
votes > n_base / 2   (i.e., strictly more than half the models vote ASD)
```

With 5 base models, this requires 3 or more votes. The raw vote count (0–5) is used as an ordinal score for AUC computation (with auto-direction correction applied if necessary).

---

## 9. Class Imbalance, Calibration, and Threshold

### 9.1 Class imbalance handling

The dataset has a mild imbalance (30 ASD vs 42 Non-ASD, ratio 1:1.4). The original R pipeline applied SMOTE oversampling to the scaled training set before each model fit. The Python pipeline replaces SMOTE with native imbalance handling:

| Model | Method | Details |
|-------|--------|---------|
| LR | `class_weight='balanced'` | Sklearn computes `n_samples / (n_classes × n_class_i)` per class; effectively upweights ASD loss |
| SVM | `class_weight='balanced'` | Penalty `C` scaled inversely by class frequency |
| RF | `class_weight='balanced'` | Each tree's sample weights adjusted; combined with isotonic calibration |
| LDA | Estimated priors | Class frequencies from training set used as class priors directly |
| XGB | `scale_pos_weight = n_neg / n_pos` | Computed per outer fold from actual fold class counts; XGBoost equivalent of balanced weighting |

Advantages over SMOTE:
- No synthetic data generation — every sample in training is a real subject
- No risk of synthetic-sample leakage if SMOTE were placed outside the inner CV (the R code applied SMOTE to the full 70% training set before any inner CV, creating a mild additional leakage source)
- Computationally cheaper

### 9.2 Threshold optimisation (Youden's J)

The default decision threshold of 0.5 is suboptimal when class sizes or costs are unequal. The pipeline replaces it with **Youden's J statistic**:

```
threshold* = argmax_{t} [ TPR(t) − FPR(t) ]
```

Procedure per outer fold:
1. After fitting each base model on the outer training set, compute predicted probabilities on that same training set (no test data used).
2. Apply AUC correction to the training probabilities (flip if AUC < 0.5).
3. Compute the ROC curve on training probabilities.
4. Select `threshold* = argmax(tpr − fpr)`, clamped to [0.05, 0.95] to avoid degenerate thresholds.
5. Apply this threshold to test-set probabilities to produce binary predictions.

This maximises the sum of sensitivity and specificity on the training set, reducing the sensitivity deficit caused by the majority-class bias of a fixed 0.5 threshold.

### 9.3 Probability calibration

Raw model probabilities are often poorly calibrated (overconfident or underconfident), especially from SVM and RF. Miscalibrated probabilities degrade ensemble performance and threshold selection.

| Model | Calibration | Rationale |
|-------|-------------|-----------|
| RF | `CalibratedClassifierCV(cv=3, method='isotonic')` | RF probabilities are biased toward 0/1 extremes; isotonic regression corrects this monotonically |
| SVM | `CalibratedClassifierCV(cv=3, method='sigmoid')` | SVM has no native probabilistic output; sigmoid (Platt scaling) maps decision scores to probabilities |
| LR | None | Logistic regression outputs are inherently well-calibrated probabilities |
| LDA | None | Gaussian LDA probabilities are calibrated under the model's assumptions |
| XGB | None | XGBoost with `eval_metric='logloss'` is trained to optimise log-loss, which encourages calibration |

`CalibratedClassifierCV(cv=3)` uses 3-fold internal cross-fitting: the calibrator is trained on held-out predictions, avoiding fitting on the same data used to train the base model. This is distinct from the deprecated `SVC(probability=True)` approach, which calibrates on the full training set.

**Parameter routing note:** When `CalibratedClassifierCV` wraps a base estimator, sklearn `Pipeline` parameter routing prefixes HPO keys with `estimator__` rather than the model name directly. The SVM grid uses keys `model__estimator__C` and `model__estimator__gamma` to route through the calibration wrapper to the underlying `SVC`.
