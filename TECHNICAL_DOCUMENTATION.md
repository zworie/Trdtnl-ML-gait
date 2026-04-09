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
