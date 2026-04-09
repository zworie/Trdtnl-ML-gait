# Findings and Discrepancies

Code-level findings from auditing `extract_rich_features.py` and `thesis.R`
against the thesis document ("Thesis Final - ZW comment 0402.docx").
The Python code is treated as the **operational ground truth**.
The thesis is treated as a **reference document**.

---

## 1. Dataset: N=72, not N=74

The thesis reports **N=74** participants (30 ASD, 44 Non-ASD).
The actual processed CSV (`gait_features_rich.csv`) contains **72 subjects**
(30 ASD, 42 Non-ASD).
Two subjects appear to have been excluded during pose estimation or
preprocessing, but the reason is not documented.

---

## 2. Feature Count: Docstring says 171; actual output is 163

`extract_rich_features.py` docstring says "171 rich gait features."
Counting the actual output columns (excluding Subject and Class) gives **163**.
Fixed in `pipeline/feature_extraction.py`.

---

## 3. Angle Formula Discrepancies (Thesis vs Code)

> The code formulas are the ground truth. Discrepancies are documented here only.

### 3a. Hip Angle — Completely Different

| | Landmarks | Formula | Measures |
|---|---|---|---|
| **Thesis** | Hip + Knee (only) | `180 − arccos(HK · (0,−1) / \|HK\|)` | Thigh deviation from fixed downward vertical axis (hip flexion from gravity). Vertical thigh = 0°. |
| **Code** | Shoulder + Hip + Knee | `arccos((RShoulder−RHip) · (RKnee−RHip) / …)` | Trunk-to-thigh angle at the hip joint. Two downward-pointing vectors nearly collinear ≈ 0°. |

These are geometrically unrelated. A fully upright posture gives ~0° for the thesis formula and a different value (near 0° or 180° depending on torso/thigh alignment) for the code. **Not interchangeable.**

### 3b. Knee Angle — Supplementary (Convention Flip)

Both use the same landmarks (Hip, Knee, Ankle) and the same vectors (KH = Knee→Hip, KA = Knee→Ankle). The only difference is:

- **Thesis**: `θ = 180 − arccos(KH · KA / …)` → straight leg = **0°**, full flexion moves away from 0°
- **Code**: `θ = arccos(Hip→Knee→Ankle)` = `arccos(KH · KA / …)` → straight leg = **180°**, full flexion decreases toward 0°

Relationship: `θ_code = 180° − θ_thesis`. They are supplementary angles.
After centering/normalization (which StandardScaler does), the sign of the
signal is flipped but the discriminative power is preserved — feature rankings
and p-values are unaffected.

### 3c. Ankle Angle — Equivalent

Both formulas compute the unsigned angle at the ankle between the lower-leg
vector and the foot vector. `arccos(dot-product)` and
`|arctan2(toe−ankle) − arctan2(knee−ankle)|` are mathematically equivalent
for angles in [0°, 180°]. **No discrepancy.**

---

## 4. Bugs Found in `extract_rich_features.py`

| ID | Location | Severity | Description | Fixed in pipeline/ |
|----|----------|----------|--------------|--------------------|
| B1 | `__main__` (lines 191–196) | **Critical** | Hardcodes `'Side_Dataset_25KP_New_GC.csv'` (underscores) but actual filenames have spaces. Raises `FileNotFoundError` on any run. | Yes — `argparse` with correct defaults |
| B2 | Line 45 | Minor | `min(11, len(sig)\|1)` uses bitwise OR `\|1`. When `len(sig) >= 11` this always returns 11, so the OR has no effect but is confusing. | Yes — replaced with explicit odd-window logic |
| B3 | Lines 139–142 | Minor | `find_peaks(..., distance=fps*0.3)` passes a float (9.0); `find_peaks` expects an integer. Line 48 uses `int(fps*0.25)`, which is inconsistent. | Yes — `int(fps*0.3)` used throughout |
| B4 | Line 53 | Minor | `if mn != 0` — exact float comparison unsafe. | Yes — `if abs(mn) > 1e-8` |
| B5 | Docstring | Low | Says "171 rich gait features"; actual output is 163. | Yes — corrected to 163 |

---

## 5. Bugs Found in `thesis.R`

| ID | Location | Severity | Description |
|----|----------|----------|-------------|
| R1 | Lines 1, 24 | **Critical** | `setwd("C:/Users/075be/Downloads/thesis")` is a hardcoded Windows path. Fails on any other machine. |
| R2 | Line 136 | **Critical** | `plot(boruta_res, ...)` called unconditionally. If Boruta fails (returns NULL at line 124), this crashes with an error on a NULL object. Needs a `if (!is.null(boruta_res))` guard. |
| R3 | Line 83 | Low | `t.test(df[[col]] ~ df$Class)` — R default is already Welch (`var.equal=FALSE`), but the commented-out code above (lines 55–74) uses `var.equal=FALSE` explicitly. Inconsistency in intent, correct in practice. |
| R4 | Class labels | Note | CSV uses `"Non-ASD"` (hyphen); R normalises to `"NonASD"` at line 29. Intentional. |
| R5 | XGBoost | Note | XGBoost (`xgbTree`) is in the R code (step 13) but **absent from published thesis results**. Thesis reports only LR, RF, SVM. XGBoost is an unofficial extension. |

---

## 6. SMOTE Placement — Confirmed Train-Only

SMOTE is applied **exclusively to the training set**, **after** the 70/30
stratified split and **after** StandardScaler normalisation.
The test set is never resampled.

- R code: lines 153–155 (`step_smote` called on `train_scaled` only)
- Thesis (p. 254): "SMOTE oversampling was applied exclusively to the
  training set after scaling, producing a balanced training set of 30 subjects
  per class."
- Python equivalent: `SMOTE(...).fit_resample(X_train_scaled, y_train)`

---

## 7. Data Leakage in Feature Selection

Both the **statistical tests** (Step 3) and **Boruta** (Step 5) are run on
the **full dataset** (all 72 subjects) in the R code, **before** the 70/30
train/test split (Step 6).

This constitutes a form of data leakage: information from the held-out test
set is used to select which features to include.  This is replicated
faithfully in `pipeline/ml_pipeline.py` to match the published results,
but it should be noted as a methodological limitation.

---

## 8. Model Set: 3 in Thesis, 4 in R Code

- **Thesis reports:** Logistic Regression (Elastic Net), Random Forest, SVM (RBF)
- **R code contains:** all three above **plus** XGBoost (`xgbTree`)
- XGBoost appears to be a later addition to `thesis.R` that was not included
  in the written results.
- `pipeline/ml_pipeline.py` includes XGBoost, labelled clearly as an extension.

---

## 9. Boruta-Confirmed Final Features (from Thesis Results)

After Boruta selection on the 142 retained features, 8 were confirmed:

| Feature | Description |
|---------|-------------|
| `LHip_min` | Minimum left hip angle |
| `RKnee_skew` | Skewness of right knee angle |
| `RAnkle_skew` | Skewness of right ankle angle (strongest signal; p=0.0001) |
| `LAnkle_kurt` | Kurtosis of left ankle angle |
| `LDP_cv` | Coefficient of variation of left dorsiplantar angle |
| `TrunkY_max` | Maximum trunk vertical displacement |
| `CoM_Y_min` | Minimum centre-of-mass vertical position |
| `StepLength` | Mean step length (pixel distance between ankles) |

`RAnkle_skew`, `TrunkY_max`, and `CoM_Y_min` were also identified by the
group-difference statistical tests, providing convergent evidence.

---

## 10. Published Thesis Results (Hold-Out Test Set)

From Table 3 / Figure 9 in the thesis:

| Model | AUC | Notes |
|-------|-----|-------|
| SVM (RBF) | **0.935** | Best model |
| Random Forest | — | — |
| Logistic Regression | — | — |

SVM optimal hyperparameters (from GridSearchCV): C=0.01, sigma=1.

---

## 11. Expected Differences Between R and Python Results

The Python pipeline (`pipeline/ml_pipeline.py`) replicates the R pipeline's logic
as faithfully as possible, but exact numerical results will differ due to
inherent implementation differences between R and Python packages:

| Component | R | Python | Impact |
|-----------|---|--------|--------|
| **Train/test split** | `caret::createDataPartition` | `sklearn.model_selection.train_test_split` | Different stratified sampling algorithms produce different splits even with the same seed value. With only 72 subjects and a 22-subject test set, a few different subjects in test can move metrics substantially. |
| **SMOTE** | `themis::step_smote` | `imblearn.over_sampling.SMOTE` | Different implementations of the SMOTE algorithm (nearest-neighbour search, synthetic sample generation). Training data differs. |
| **Logistic Regression** | `glmnet` (coordinate descent) | `saga` solver (stochastic average gradient) | Different optimisation algorithms for the same objective function. May converge to different solutions. |
| **SVM** | `kernlab::ksvm` | `libsvm` (via sklearn) | Different SVM implementations. `probability=True` in sklearn adds Platt scaling which can shift AUC slightly. |
| **Random Forest** | R's `randomForest` package | sklearn's `RandomForestClassifier` | Different tree-building implementations. Feature importance values will differ. |
| **Boruta** | R's `Boruta` package (uses `ranger`) | `boruta-py` (uses sklearn RF) | Different RF backends may rank features differently, potentially selecting a different final feature set. |
| **CV fold assignments** | caret's internal CV splitter | `RepeatedStratifiedKFold` | Different fold assignments even with the same seed. |

**Bottom line:** Metrics will be in the same ballpark but not identical.
The relative ranking of models (SVM typically best) and the selected features
should be broadly consistent. Exact AUC/F1 values may differ by 0.05–0.15
depending on how the test set falls.

To get identical results to the thesis, run the original `thesis.R` in R.
