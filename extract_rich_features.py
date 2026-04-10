"""
Rich Gait Feature Extraction from Keypoint Time-Series
=======================================================
Instead of just mean angle, we extract per-subject:
  - Mean, Std, Range, Min, Max, Median
  - ROM (Range of Motion = max - min)
  - CV (Coefficient of Variation)
  - Peak count (oscillation frequency)
  - Symmetry indices (left vs right)
  - Temporal features (zero-crossing rate, smoothness)
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import skew, kurtosis

# ─── Angle helpers ────────────────────────────────────────────────────────────

def angle_3pts(A, B, C):
    BA = A - B; BC = C - B
    dot = np.einsum('ij,ij->i', BA, BC)
    n1 = np.linalg.norm(BA, axis=1); n2 = np.linalg.norm(BC, axis=1)
    d  = n1 * n2
    cos_a = np.where(d > 1e-8, dot / d, 0.0)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def signed_angle_2d(v1, v2):
    cross = v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0]
    dot   = np.einsum('ij,ij->i', v1, v2)
    return np.degrees(np.arctan2(cross, dot))

def pts(df, name):
    return df[[name+'X', name+'Y']].values.astype(float)

# ─── Statistical feature set from a 1-D signal ───────────────────────────────

def stat_features(sig, prefix, fps=30):
    sig = np.array(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) == 0:
        sig = np.array([0.0])

    # Smooth for peak detection
    smooth = savgol_filter(sig, window_length=min(11, len(sig)|1), polyorder=2) \
             if len(sig) >= 11 else sig

    peaks, _  = find_peaks(smooth, distance=int(fps*0.25))
    troughs,_ = find_peaks(-smooth, distance=int(fps*0.25))

    rom = sig.max() - sig.min()
    mn  = sig.mean()
    cv  = (sig.std() / mn * 100) if mn != 0 else 0.0

    # Zero-crossing rate of de-meaned signal
    dm    = sig - mn
    zcr   = ((dm[:-1] * dm[1:]) < 0).sum() / max(len(sig)-1, 1)

    return {
        f'{prefix}_mean'    : round(float(mn), 4),
        f'{prefix}_std'     : round(float(sig.std()), 4),
        f'{prefix}_min'     : round(float(sig.min()), 4),
        f'{prefix}_max'     : round(float(sig.max()), 4),
        f'{prefix}_rom'     : round(float(rom), 4),
        f'{prefix}_cv'      : round(float(cv), 4),
        f'{prefix}_skew'    : round(float(skew(sig)), 4),
        f'{prefix}_kurt'    : round(float(kurtosis(sig)), 4),
        f'{prefix}_peaks'   : int(len(peaks)),
        f'{prefix}_troughs' : int(len(troughs)),
        f'{prefix}_zcr'     : round(float(zcr), 4),
    }

def symmetry_index(left_sig, right_sig, prefix):
    """|(L-R)| / (0.5*(L+R)) * 100  — Symmetry Index (SI%)"""
    l = np.nanmean(left_sig); r = np.nanmean(right_sig)
    denom = 0.5*(abs(l)+abs(r))
    si = abs(l-r)/denom*100 if denom > 1e-8 else 0.0
    return {f'{prefix}_SI': round(si, 4)}

# ─── Per-subject feature extraction ──────────────────────────────────────────

def extract_subject(sg, fg, fps=30):
    feats = {}

    # ── SIDE VIEW ─────────────────────────────────────────────────────────────
    # Hip angles
    r_hip = angle_3pts(pts(sg,'RShoulder'), pts(sg,'RHip'), pts(sg,'RKnee'))
    l_hip = angle_3pts(pts(sg,'LShoulder'), pts(sg,'LHip'), pts(sg,'LKnee'))
    feats.update(stat_features(r_hip, 'RHip', fps))
    feats.update(stat_features(l_hip, 'LHip', fps))
    feats.update(symmetry_index(l_hip, r_hip, 'Hip'))

    # Knee angles
    r_kn = angle_3pts(pts(sg,'RHip'), pts(sg,'RKnee'), pts(sg,'RAnkle'))
    l_kn = angle_3pts(pts(sg,'LHip'), pts(sg,'LKnee'), pts(sg,'LAnkle'))
    feats.update(stat_features(r_kn, 'RKnee', fps))
    feats.update(stat_features(l_kn, 'LKnee', fps))
    feats.update(symmetry_index(l_kn, r_kn, 'Knee'))

    # Ankle angles
    r_ank = angle_3pts(pts(sg,'RKnee'), pts(sg,'RAnkle'), pts(sg,'RBigToe'))
    l_ank = angle_3pts(pts(sg,'LKnee'), pts(sg,'LAnkle'), pts(sg,'LBigToe'))
    feats.update(stat_features(r_ank, 'RAnkle', fps))
    feats.update(stat_features(l_ank, 'LAnkle', fps))
    feats.update(symmetry_index(l_ank, r_ank, 'Ankle'))

    # Dorsiplantar angles
    def dp_ang(heel, toe):
        v = toe - heel; h = np.zeros_like(v); h[:,0]=1
        return np.abs(signed_angle_2d(h, v))
    r_dp = dp_ang(pts(sg,'RHeel'), pts(sg,'RBigToe'))
    l_dp = dp_ang(pts(sg,'LHeel'), pts(sg,'LBigToe'))
    feats.update(stat_features(r_dp, 'RDP', fps))
    feats.update(stat_features(l_dp, 'LDP', fps))
    feats.update(symmetry_index(l_dp, r_dp, 'DP'))

    # Trunk sway (MHip vertical oscillation as proxy)
    trunk_y = sg['MHipY'].values.astype(float)
    feats.update(stat_features(trunk_y, 'TrunkY', fps))

    # ── FRONT VIEW ────────────────────────────────────────────────────────────
    # Step width (lateral ankle separation)
    sw = np.abs(fg['RAnkleX'].values - fg['LAnkleX'].values)
    feats.update(stat_features(sw, 'StepWidth', fps))

    # Adduction/Abduction
    l_add = signed_angle_2d(
        pts(fg,'LShoulder') - pts(fg,'LHip'),
        pts(fg,'LKnee')     - pts(fg,'LHip'))
    r_add = signed_angle_2d(
        pts(fg,'RShoulder') - pts(fg,'RHip'),
        pts(fg,'RKnee')     - pts(fg,'RHip'))
    feats.update(stat_features(l_add, 'LAddAbd', fps))
    feats.update(stat_features(r_add, 'RAddAbd', fps))
    feats.update(symmetry_index(l_add, r_add, 'AddAbd'))

    # Cadence & steps from ankle Y oscillation
    ry = fg['RAnkleY'].values; ly = fg['LAnkleY'].values
    rp,_ = find_peaks(ry, distance=fps*0.3)
    lp,_ = find_peaks(ly, distance=fps*0.3)
    if len(rp)<2: rp,_ = find_peaks(-ry, distance=fps*0.3)
    if len(lp)<2: lp,_ = find_peaks(-ly, distance=fps*0.3)
    steps    = len(rp)+len(lp)
    cadence  = steps / (len(fg)/fps/60)

    # Step & stride length
    dists = np.linalg.norm(pts(fg,'RAnkle') - pts(fg,'LAnkle'), axis=1)
    feats.update(stat_features(dists, 'StepLen', fps))

    # Vertical CoM oscillation (MHipY from front)
    feats.update(stat_features(fg['MHipY'].values, 'CoM_Y', fps))

    # Scalar summary features
    feats['Cadence']      = round(cadence, 4)
    feats['Steps']        = steps
    feats['StepLength']   = round(float(dists.mean()), 4)
    feats['StrideLength'] = round(float(dists.mean()*2), 4)

    return feats


# ─── Main ─────────────────────────────────────────────────────────────────────

def extract_all(side_csv, front_csv, fps=30):
    ds = pd.read_csv(side_csv)
    df = pd.read_csv(front_csv)
    for d in (ds, df):
        if 'Label' in d.columns:
            d.rename(columns={'Label':'Class'}, inplace=True)

    subjects = sorted(set(ds['Subject'].unique()) | set(df['Subject'].unique()))
    records  = []
    for subj in subjects:
        sg = ds[ds['Subject']==subj]
        fg = df[df['Subject']==subj]
        if sg.empty or fg.empty:
            print(f"[SKIP] {subj}")
            continue
        row = {'Subject': subj, 'Class': sg['Class'].iloc[0]}
        row.update(extract_subject(sg, fg, fps))
        records.append(row)

    out = pd.DataFrame(records)
    # Move Class to end
    cols = [c for c in out.columns if c not in ('Subject','Class')] 
    out  = out[['Subject'] + cols + ['Class']]
    return out


if __name__ == '__main__':
    out = extract_all(
        'Side_Dataset_25KP_New_GC.csv',
        'Front_Dataset_25KP_New_GC.csv',
        fps=30
    )
    out.to_csv('gait_features_rich.csv', index=False)
    print(f"Shape: {out.shape}")
    print(f"Class:\n{out['Class'].value_counts()}")
    print(f"\nFeatures: {[c for c in out.columns if c not in ('Subject','Class')][:20]} ...")
