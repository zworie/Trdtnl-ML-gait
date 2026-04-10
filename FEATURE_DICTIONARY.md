# Feature Dictionary — Gait Analysis Pipeline

Complete reference for all **163 features** extracted by the feature engineering pipeline.

**Source code:** `pipeline/feature_extraction.py`
**Input data:** Two-view (side + front) pose-estimation keypoint CSVs with 25 body landmarks per frame.

| Category | Signals | Features |
|----------|---------|----------|
| Bilateral joint angles (side view) | RHip, LHip, RKnee, LKnee, RAnkle, LAnkle | 6 x 11 = 66 |
| Dorsiplantar angle (side view) | RDP, LDP | 2 x 11 = 22 |
| Trunk oscillation (side view) | TrunkY | 11 |
| Step width (front view) | StepWidth | 11 |
| Hip add/abduction (front view) | LAddAbd, RAddAbd | 2 x 11 = 22 |
| Step length (front view) | StepLen | 11 |
| Centre-of-mass oscillation (front view) | CoM_Y | 11 |
| Symmetry indices | Hip_SI, Knee_SI, Ankle_SI, DP_SI, AddAbd_SI | 5 |
| Scalar gait parameters | Cadence, Steps, StepLength, StrideLength | 4 |
| **Total** | | **163** |

---

## 1. Statistical Metrics Reference

Every base signal is summarised with these 11 statistics (function `stat_features()`, line 77):

| Suffix | Full Name | Formula / Definition | Interpretation |
|--------|-----------|----------------------|----------------|
| `_mean` | Arithmetic mean | mean(x) | Central tendency of the signal across all frames |
| `_std` | Standard deviation | std(x) (population, ddof=0) | Spread / variability of the signal |
| `_min` | Minimum | min(x) | Lowest value observed during the recording |
| `_max` | Maximum | max(x) | Highest value observed during the recording |
| `_rom` | Range of motion | max(x) - min(x) | Total excursion; larger ROM = more movement |
| `_cv` | Coefficient of variation | std / \|mean\| x 100 (0 if \|mean\| < 1e-8) | Relative variability normalised to the mean (%) |
| `_skew` | Skewness | scipy.stats.skew(x) (3rd standardised moment) | Asymmetry of the distribution; 0 = symmetric, positive = right-tailed |
| `_kurt` | Excess kurtosis | scipy.stats.kurtosis(x) (4th standardised moment - 3) | Tail heaviness; 0 = normal-like, positive = heavy tails, negative = light tails |
| `_peaks` | Peak count | Number of local maxima in Savitzky-Golay-smoothed signal (window=11, polyorder=2, min distance=fps x 0.25) | Proxy for the number of gait cycles (flexion events) |
| `_troughs` | Trough count | Number of local minima (peaks of negated smoothed signal) | Proxy for the number of gait cycles (extension events) |
| `_zcr` | Zero-crossing rate | Fraction of consecutive sample-pairs where the demeaned signal crosses zero | Oscillation frequency indicator; higher = more frequent reversals |

---

## 2. Angle Computation Methods

| Function | Formula | Range | Used for |
|----------|---------|-------|----------|
| `angle_3pts(A, B, C)` | arccos((BA . BC) / (\|BA\| x \|BC\|)) | 0 - 180 deg | Hip, knee, ankle joint angles |
| `signed_angle_2d(v1, v2)` | atan2(v1 x v2, v1 . v2) | -180 to 180 deg | Adduction/abduction, dorsiplantar |
| `dp_ang(heel, toe)` | \|signed_angle_2d([1,0], toe - heel)\| | 0 - 180 deg | Foot tilt relative to horizontal |

**Symmetry Index:** SI = \|L_mean - R_mean\| / (0.5 x (\|L_mean\| + \|R_mean\|)) x 100. Result is 0% for perfect bilateral symmetry; larger values indicate greater asymmetry.

---

## 3. Complete Feature Table

Example values are from Subject 101 (first row of `gait_features_rich.csv`).

### 3.1 Hip Flexion/Extension (23 features)

**View:** Side | **Method:** `angle_3pts(Shoulder, Hip, Knee)` | **Units:** Degrees (0-180) | **Interpretation:** Trunk-to-thigh angle; smaller values indicate greater hip flexion.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 1 | `RHip_mean` | Right hip angle - mean | 164.2047 |
| 2 | `RHip_std` | Right hip angle - standard deviation | 7.452 |
| 3 | `RHip_min` | Right hip angle - minimum | 147.6519 |
| 4 | `RHip_max` | Right hip angle - maximum | 179.8574 |
| 5 | `RHip_rom` | Right hip angle - range of motion | 32.2055 |
| 6 | `RHip_cv` | Right hip angle - coefficient of variation | 4.5383 |
| 7 | `RHip_skew` | Right hip angle - skewness | 0.0854 |
| 8 | `RHip_kurt` | Right hip angle - excess kurtosis | -0.9369 |
| 9 | `RHip_peaks` | Right hip angle - peak count | 8 |
| 10 | `RHip_troughs` | Right hip angle - trough count | 8 |
| 11 | `RHip_zcr` | Right hip angle - zero-crossing rate | 0.2185 |
| 12 | `LHip_mean` | Left hip angle - mean | 163.569 |
| 13 | `LHip_std` | Left hip angle - standard deviation | 7.6753 |
| 14 | `LHip_min` | Left hip angle - minimum | 146.3052 |
| 15 | `LHip_max` | Left hip angle - maximum | 179.7917 |
| 16 | `LHip_rom` | Left hip angle - range of motion | 33.4865 |
| 17 | `LHip_cv` | Left hip angle - coefficient of variation | 4.6924 |
| 18 | `LHip_skew` | Left hip angle - skewness | 0.3387 |
| 19 | `LHip_kurt` | Left hip angle - excess kurtosis | -0.7597 |
| 20 | `LHip_peaks` | Left hip angle - peak count | 10 |
| 21 | `LHip_troughs` | Left hip angle - trough count | 10 |
| 22 | `LHip_zcr` | Left hip angle - zero-crossing rate | 0.2605 |
| 23 | `Hip_SI` | Hip symmetry index | 0.3879 |

### 3.2 Knee Flexion/Extension (23 features)

**View:** Side | **Method:** `angle_3pts(Hip, Knee, Ankle)` | **Units:** Degrees (0-180) | **Interpretation:** Straight leg ~ 180 deg; full flexion < 90 deg.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 24 | `RKnee_mean` | Right knee angle - mean | 153.7842 |
| 25 | `RKnee_std` | Right knee angle - standard deviation | 20.1859 |
| 26 | `RKnee_min` | Right knee angle - minimum | 103.1809 |
| 27 | `RKnee_max` | Right knee angle - maximum | 179.9444 |
| 28 | `RKnee_rom` | Right knee angle - range of motion | 76.7634 |
| 29 | `RKnee_cv` | Right knee angle - coefficient of variation | 13.1261 |
| 30 | `RKnee_skew` | Right knee angle - skewness | -0.8082 |
| 31 | `RKnee_kurt` | Right knee angle - excess kurtosis | -0.3556 |
| 32 | `RKnee_peaks` | Right knee angle - peak count | 7 |
| 33 | `RKnee_troughs` | Right knee angle - trough count | 8 |
| 34 | `RKnee_zcr` | Right knee angle - zero-crossing rate | 0.1849 |
| 35 | `LKnee_mean` | Left knee angle - mean | 154.3775 |
| 36 | `LKnee_std` | Left knee angle - standard deviation | 19.3354 |
| 37 | `LKnee_min` | Left knee angle - minimum | 101.4467 |
| 38 | `LKnee_max` | Left knee angle - maximum | 179.5245 |
| 39 | `LKnee_rom` | Left knee angle - range of motion | 78.0778 |
| 40 | `LKnee_cv` | Left knee angle - coefficient of variation | 12.5248 |
| 41 | `LKnee_skew` | Left knee angle - skewness | -0.8792 |
| 42 | `LKnee_kurt` | Left knee angle - excess kurtosis | -0.1768 |
| 43 | `LKnee_peaks` | Left knee angle - peak count | 7 |
| 44 | `LKnee_troughs` | Left knee angle - trough count | 8 |
| 45 | `LKnee_zcr` | Left knee angle - zero-crossing rate | 0.1849 |
| 46 | `Knee_SI` | Knee symmetry index | 0.385 |

### 3.3 Ankle Dorsiflexion/Plantarflexion (23 features)

**View:** Side | **Method:** `angle_3pts(Knee, Ankle, BigToe)` | **Units:** Degrees (0-180) | **Interpretation:** Angle at the ankle joint; varies with foot strike and push-off phases.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 47 | `RAnkle_mean` | Right ankle angle - mean | 98.7167 |
| 48 | `RAnkle_std` | Right ankle angle - standard deviation | 14.7103 |
| 49 | `RAnkle_min` | Right ankle angle - minimum | 60.663 |
| 50 | `RAnkle_max` | Right ankle angle - maximum | 158.3391 |
| 51 | `RAnkle_rom` | Right ankle angle - range of motion | 97.6762 |
| 52 | `RAnkle_cv` | Right ankle angle - coefficient of variation | 14.9015 |
| 53 | `RAnkle_skew` | Right ankle angle - skewness | 0.9912 |
| 54 | `RAnkle_kurt` | Right ankle angle - excess kurtosis | 2.3744 |
| 55 | `RAnkle_peaks` | Right ankle angle - peak count | 10 |
| 56 | `RAnkle_troughs` | Right ankle angle - trough count | 8 |
| 57 | `RAnkle_zcr` | Right ankle angle - zero-crossing rate | 0.2185 |
| 58 | `LAnkle_mean` | Left ankle angle - mean | 101.2186 |
| 59 | `LAnkle_std` | Left ankle angle - standard deviation | 14.1758 |
| 60 | `LAnkle_min` | Left ankle angle - minimum | 68.313 |
| 61 | `LAnkle_max` | Left ankle angle - maximum | 158.0936 |
| 62 | `LAnkle_rom` | Left ankle angle - range of motion | 89.7806 |
| 63 | `LAnkle_cv` | Left ankle angle - coefficient of variation | 14.0051 |
| 64 | `LAnkle_skew` | Left ankle angle - skewness | 1.4493 |
| 65 | `LAnkle_kurt` | Left ankle angle - excess kurtosis | 3.3451 |
| 66 | `LAnkle_peaks` | Left ankle angle - peak count | 9 |
| 67 | `LAnkle_troughs` | Left ankle angle - trough count | 7 |
| 68 | `LAnkle_zcr` | Left ankle angle - zero-crossing rate | 0.2185 |
| 69 | `Ankle_SI` | Ankle symmetry index | 2.5027 |

### 3.4 Dorsiplantar Angle (23 features)

**View:** Side | **Method:** `|signed_angle_2d([1,0], BigToe - Heel)|` | **Units:** Degrees (0-180) | **Interpretation:** Foot segment tilt relative to horizontal; 0 deg = flat foot, higher = steeper tilt (dorsiflexion or plantarflexion).

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 70 | `RDP_mean` | Right dorsiplantar angle - mean | 84.6526 |
| 71 | `RDP_std` | Right dorsiplantar angle - standard deviation | 70.3207 |
| 72 | `RDP_min` | Right dorsiplantar angle - minimum | 0.1474 |
| 73 | `RDP_max` | Right dorsiplantar angle - maximum | 179.8201 |
| 74 | `RDP_rom` | Right dorsiplantar angle - range of motion | 179.6727 |
| 75 | `RDP_cv` | Right dorsiplantar angle - coefficient of variation | 83.0697 |
| 76 | `RDP_skew` | Right dorsiplantar angle - skewness | 0.1025 |
| 77 | `RDP_kurt` | Right dorsiplantar angle - excess kurtosis | -1.7318 |
| 78 | `RDP_peaks` | Right dorsiplantar angle - peak count | 9 |
| 79 | `RDP_troughs` | Right dorsiplantar angle - trough count | 10 |
| 80 | `RDP_zcr` | Right dorsiplantar angle - zero-crossing rate | 0.0252 |
| 81 | `LDP_mean` | Left dorsiplantar angle - mean | 96.824 |
| 82 | `LDP_std` | Left dorsiplantar angle - standard deviation | 68.0742 |
| 83 | `LDP_min` | Left dorsiplantar angle - minimum | 4.2248 |
| 84 | `LDP_max` | Left dorsiplantar angle - maximum | 179.8754 |
| 85 | `LDP_rom` | Left dorsiplantar angle - range of motion | 175.6506 |
| 86 | `LDP_cv` | Left dorsiplantar angle - coefficient of variation | 70.3071 |
| 87 | `LDP_skew` | Left dorsiplantar angle - skewness | -0.0364 |
| 88 | `LDP_kurt` | Left dorsiplantar angle - excess kurtosis | -1.7747 |
| 89 | `LDP_peaks` | Left dorsiplantar angle - peak count | 9 |
| 90 | `LDP_troughs` | Left dorsiplantar angle - trough count | 10 |
| 91 | `LDP_zcr` | Left dorsiplantar angle - zero-crossing rate | 0.0252 |
| 92 | `DP_SI` | Dorsiplantar symmetry index | 13.4138 |

### 3.5 Trunk Vertical Oscillation (11 features)

**View:** Side | **Signal:** MHipY (mid-hip Y coordinate) | **Units:** Pixels | **Interpretation:** Vertical position of the trunk over time; oscillations correspond to the up-and-down bounce during gait.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 93 | `TrunkY_mean` | Trunk vertical position - mean | 712.4718 |
| 94 | `TrunkY_std` | Trunk vertical position - standard deviation | 13.1625 |
| 95 | `TrunkY_min` | Trunk vertical position - minimum | 691.3808 |
| 96 | `TrunkY_max` | Trunk vertical position - maximum | 745.1969 |
| 97 | `TrunkY_rom` | Trunk vertical position - range of motion | 53.8161 |
| 98 | `TrunkY_cv` | Trunk vertical position - coefficient of variation | 1.8474 |
| 99 | `TrunkY_skew` | Trunk vertical position - skewness | 0.4232 |
| 100 | `TrunkY_kurt` | Trunk vertical position - excess kurtosis | -0.1745 |
| 101 | `TrunkY_peaks` | Trunk vertical position - peak count | 7 |
| 102 | `TrunkY_troughs` | Trunk vertical position - trough count | 8 |
| 103 | `TrunkY_zcr` | Trunk vertical position - zero-crossing rate | 0.1008 |

### 3.6 Step Width (11 features)

**View:** Front | **Signal:** |RAnkleX - LAnkleX| | **Units:** Pixels | **Interpretation:** Lateral distance between ankles; wider step width may indicate balance compensation.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 104 | `StepWidth_mean` | Step width - mean | 37.0783 |
| 105 | `StepWidth_std` | Step width - standard deviation | 11.4774 |
| 106 | `StepWidth_min` | Step width - minimum | 6.0488 |
| 107 | `StepWidth_max` | Step width - maximum | 67.749 |
| 108 | `StepWidth_rom` | Step width - range of motion | 61.7002 |
| 109 | `StepWidth_cv` | Step width - coefficient of variation | 30.9544 |
| 110 | `StepWidth_skew` | Step width - skewness | -0.1938 |
| 111 | `StepWidth_kurt` | Step width - excess kurtosis | 0.0334 |
| 112 | `StepWidth_peaks` | Step width - peak count | 7 |
| 113 | `StepWidth_troughs` | Step width - trough count | 7 |
| 114 | `StepWidth_zcr` | Step width - zero-crossing rate | 0.0756 |

### 3.7 Hip Adduction/Abduction (23 features)

**View:** Front | **Method:** `signed_angle_2d(Shoulder - Hip, Knee - Hip)` | **Units:** Degrees (-180 to 180) | **Interpretation:** Frontal-plane hip angle; positive values indicate adduction (leg moving inward), negative values indicate abduction (leg moving outward).

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 115 | `LAddAbd_mean` | Left hip adduction/abduction angle - mean | 35.0364 |
| 116 | `LAddAbd_std` | Left hip adduction/abduction angle - standard deviation | 173.2585 |
| 117 | `LAddAbd_min` | Left hip adduction/abduction angle - minimum | -179.9912 |
| 118 | `LAddAbd_max` | Left hip adduction/abduction angle - maximum | 179.7699 |
| 119 | `LAddAbd_rom` | Left hip adduction/abduction angle - range of motion | 359.7611 |
| 120 | `LAddAbd_cv` | Left hip adduction/abduction angle - coefficient of variation | 494.5104 |
| 121 | `LAddAbd_skew` | Left hip adduction/abduction angle - skewness | -0.408 |
| 122 | `LAddAbd_kurt` | Left hip adduction/abduction angle - excess kurtosis | -1.8329 |
| 123 | `LAddAbd_peaks` | Left hip adduction/abduction angle - peak count | 9 |
| 124 | `LAddAbd_troughs` | Left hip adduction/abduction angle - trough count | 10 |
| 125 | `LAddAbd_zcr` | Left hip adduction/abduction angle - zero-crossing rate | 0.2017 |
| 126 | `RAddAbd_mean` | Right hip adduction/abduction angle - mean | 3.1205 |
| 127 | `RAddAbd_std` | Right hip adduction/abduction angle - standard deviation | 174.9782 |
| 128 | `RAddAbd_min` | Right hip adduction/abduction angle - minimum | -179.9944 |
| 129 | `RAddAbd_max` | Right hip adduction/abduction angle - maximum | 179.4459 |
| 130 | `RAddAbd_rom` | Right hip adduction/abduction angle - range of motion | 359.4403 |
| 131 | `RAddAbd_cv` | Right hip adduction/abduction angle - coefficient of variation | 5607.3929 |
| 132 | `RAddAbd_skew` | Right hip adduction/abduction angle - skewness | -0.0335 |
| 133 | `RAddAbd_kurt` | Right hip adduction/abduction angle - excess kurtosis | -1.9978 |
| 134 | `RAddAbd_peaks` | Right hip adduction/abduction angle - peak count | 11 |
| 135 | `RAddAbd_troughs` | Right hip adduction/abduction angle - trough count | 11 |
| 136 | `RAddAbd_zcr` | Right hip adduction/abduction angle - zero-crossing rate | 0.1261 |
| 137 | `AddAbd_SI` | Adduction/abduction symmetry index | 167.2877 |

### 3.8 Step Length (11 features)

**View:** Front | **Signal:** Euclidean distance between RAnkle and LAnkle (2D) per frame | **Units:** Pixels | **Interpretation:** Inter-ankle distance time series; captures gait cycle oscillation between feet-apart and feet-together phases.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 138 | `StepLen_mean` | Step length - mean | 44.5297 |
| 139 | `StepLen_std` | Step length - standard deviation | 13.2174 |
| 140 | `StepLen_min` | Step length - minimum | 13.6008 |
| 141 | `StepLen_max` | Step length - maximum | 73.2353 |
| 142 | `StepLen_rom` | Step length - range of motion | 59.6344 |
| 143 | `StepLen_cv` | Step length - coefficient of variation | 29.6823 |
| 144 | `StepLen_skew` | Step length - skewness | 0.1172 |
| 145 | `StepLen_kurt` | Step length - excess kurtosis | -0.5959 |
| 146 | `StepLen_peaks` | Step length - peak count | 9 |
| 147 | `StepLen_troughs` | Step length - trough count | 8 |
| 148 | `StepLen_zcr` | Step length - zero-crossing rate | 0.1345 |

### 3.9 Centre of Mass Vertical Oscillation (11 features)

**View:** Front | **Signal:** MHipY (mid-hip Y coordinate) | **Units:** Pixels | **Interpretation:** Vertical CoM proxy from the frontal view; complements TrunkY (side view) and captures frontal-plane vertical dynamics.

| # | Feature | Full Name | Example |
|---|---------|-----------|---------|
| 149 | `CoM_Y_mean` | Centre-of-mass vertical position - mean | 715.4475 |
| 150 | `CoM_Y_std` | Centre-of-mass vertical position - standard deviation | 14.4791 |
| 151 | `CoM_Y_min` | Centre-of-mass vertical position - minimum | 689.0037 |
| 152 | `CoM_Y_max` | Centre-of-mass vertical position - maximum | 742.276 |
| 153 | `CoM_Y_rom` | Centre-of-mass vertical position - range of motion | 53.2723 |
| 154 | `CoM_Y_cv` | Centre-of-mass vertical position - coefficient of variation | 2.0238 |
| 155 | `CoM_Y_skew` | Centre-of-mass vertical position - skewness | 0.1728 |
| 156 | `CoM_Y_kurt` | Centre-of-mass vertical position - excess kurtosis | -0.8753 |
| 157 | `CoM_Y_peaks` | Centre-of-mass vertical position - peak count | 8 |
| 158 | `CoM_Y_troughs` | Centre-of-mass vertical position - trough count | 9 |
| 159 | `CoM_Y_zcr` | Centre-of-mass vertical position - zero-crossing rate | 0.084 |

### 3.10 Scalar Gait Parameters (4 features)

| # | Feature | Full Name | Definition | Units | Example |
|---|---------|-----------|------------|-------|---------|
| 160 | `Cadence` | Cadence (steps per minute) | (R_ankle_peaks + L_ankle_peaks) / (n_frames / fps / 60); peaks detected from ankle Y oscillation with min distance = fps x 0.3 | steps/min | 285.0 |
| 161 | `Steps` | Total step count | R_ankle_peaks + L_ankle_peaks; if < 2 peaks found in a signal, peaks of the inverted signal are used instead | count | 19 |
| 162 | `StepLength` | Mean step length | mean(Euclidean distance between RAnkle and LAnkle across all frames) | pixels | 44.5297 |
| 163 | `StrideLength` | Mean stride length | 2 x StepLength (one full gait cycle = two steps) | pixels | 89.0594 |

---

## 4. Symmetry Index Reference

**Formula:** SI = |L_mean - R_mean| / (0.5 x (|L_mean| + |R_mean|)) x 100

| Feature | Paired Signals | Interpretation |
|---------|----------------|----------------|
| `Hip_SI` | LHip vs RHip | Hip angle bilateral asymmetry (%) |
| `Knee_SI` | LKnee vs RKnee | Knee angle bilateral asymmetry (%) |
| `Ankle_SI` | LAnkle vs RAnkle | Ankle angle bilateral asymmetry (%) |
| `DP_SI` | LDP vs RDP | Dorsiplantar angle bilateral asymmetry (%) |
| `AddAbd_SI` | LAddAbd vs RAddAbd | Hip adduction/abduction bilateral asymmetry (%) |

A value of 0% indicates perfect symmetry. Values > 10% are generally considered clinically meaningful asymmetry.

---

## 5. Signal Source Summary

| Signal | View | Landmarks | Angle Method | Units | Typical Range |
|--------|------|-----------|--------------|-------|---------------|
| RHip / LHip | Side | Shoulder -> Hip -> Knee | `angle_3pts` (unsigned) | Degrees | 140 - 180 |
| RKnee / LKnee | Side | Hip -> Knee -> Ankle | `angle_3pts` (unsigned) | Degrees | 100 - 180 |
| RAnkle / LAnkle | Side | Knee -> Ankle -> BigToe | `angle_3pts` (unsigned) | Degrees | 60 - 160 |
| RDP / LDP | Side | Heel -> BigToe vs horizontal | `signed_angle_2d` (absolute) | Degrees | 0 - 180 |
| TrunkY | Side | MHipY coordinate | Direct Y value | Pixels | varies |
| StepWidth | Front | \|RAnkleX - LAnkleX\| | Absolute difference | Pixels | varies |
| LAddAbd / RAddAbd | Front | Shoulder-Hip vs Hip-Knee vectors | `signed_angle_2d` (signed) | Degrees | -180 to 180 |
| StepLen | Front | RAnkle - LAnkle | Euclidean norm | Pixels | varies |
| CoM_Y | Front | MHipY coordinate | Direct Y value | Pixels | varies |
