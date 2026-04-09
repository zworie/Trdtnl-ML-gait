"""
Rich Gait Feature Extraction from Keypoint Time-Series
=======================================================
Extracts per-subject biomechanical gait features from two-view pose keypoint
CSV files (side view and front view).  Produces 163 statistical and spatial
summary features per subject.

Features per joint (11 per signal):
  mean, std, min, max, rom (range), cv (coeff. of variation),
  skew, kurt, peaks, troughs, zcr (zero-crossing rate)

Signals extracted:
  Side view:  RHip, LHip, RKnee, LKnee, RAnkle, LAnkle, RDP, LDP, TrunkY
  Front view: StepWidth, LAddAbd, RAddAbd, CoM_Y
  Symmetry indices: Hip, Knee, Ankle, DP, AddAbd
  Scalars: Cadence, Steps, StepLength, StrideLength

Usage
-----
  # Normal run – writes to outputs/gait_features_extracted.csv
  python pipeline/feature_extraction.py

  # Verify against existing gait_features_rich.csv (nothing written to disk)
  python pipeline/feature_extraction.py --verify

  # Custom paths
  python pipeline/feature_extraction.py \\
      --side "Side Dataset 25KP New GC.csv" \\
      --front "Front Dataset 25KP New GC.csv" \\
      --out outputs/gait_features_extracted.csv

Notes
-----
- The existing gait_features_rich.csv is NEVER overwritten by this script.
- Angle formulas follow the Python code conventions (not the thesis formulas);
  see FINDINGS.md for a detailed comparison.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import kurtosis, skew


# ── Angle helpers ──────────────────────────────────────────────────────────────

def angle_3pts(A, B, C):
    """Unsigned angle at vertex B formed by rays B→A and B→C (degrees)."""
    BA = A - B
    BC = C - B
    dot = np.einsum('ij,ij->i', BA, BC)
    n1 = np.linalg.norm(BA, axis=1)
    n2 = np.linalg.norm(BC, axis=1)
    d = n1 * n2
    cos_a = np.where(d > 1e-8, dot / d, 0.0)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def signed_angle_2d(v1, v2):
    """Signed angle from v1 to v2 in the 2-D plane (degrees)."""
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = np.einsum('ij,ij->i', v1, v2)
    return np.degrees(np.arctan2(cross, dot))


def pts(df, name):
    """Return (N,2) float array for landmark <name> from columns <name>X, <name>Y."""
    return df[[name + 'X', name + 'Y']].values.astype(float)


# ── Statistical features from a 1-D signal ────────────────────────────────────

def stat_features(sig, prefix, fps=30):
    """Compute 11 summary statistics for signal *sig* and return as a dict."""
    sig = np.array(sig, dtype=float)
    sig = sig[~np.isnan(sig)]
    if len(sig) == 0:
        sig = np.array([0.0])

    # Savitzky-Golay smoothing for peak detection (need at least 11 samples)
    if len(sig) >= 11:
        wl = min(11, len(sig))
        if wl % 2 == 0:       # window length must be odd
            wl -= 1
        smooth = savgol_filter(sig, window_length=wl, polyorder=2)
    else:
        smooth = sig

    peaks, _ = find_peaks(smooth, distance=int(fps * 0.25))
    troughs, _ = find_peaks(-smooth, distance=int(fps * 0.25))

    rom = sig.max() - sig.min()
    mn = sig.mean()
    cv = (sig.std() / mn * 100) if abs(mn) > 1e-8 else 0.0

    dm = sig - mn
    zcr = ((dm[:-1] * dm[1:]) < 0).sum() / max(len(sig) - 1, 1)

    return {
        f'{prefix}_mean'   : round(float(mn), 4),
        f'{prefix}_std'    : round(float(sig.std()), 4),
        f'{prefix}_min'    : round(float(sig.min()), 4),
        f'{prefix}_max'    : round(float(sig.max()), 4),
        f'{prefix}_rom'    : round(float(rom), 4),
        f'{prefix}_cv'     : round(float(cv), 4),
        f'{prefix}_skew'   : round(float(skew(sig)), 4),
        f'{prefix}_kurt'   : round(float(kurtosis(sig)), 4),
        f'{prefix}_peaks'  : int(len(peaks)),
        f'{prefix}_troughs': int(len(troughs)),
        f'{prefix}_zcr'    : round(float(zcr), 4),
    }


def symmetry_index(left_sig, right_sig, prefix):
    """Symmetry Index: |(L−R)| / (0.5*(|L|+|R|)) * 100."""
    l = np.nanmean(left_sig)
    r = np.nanmean(right_sig)
    denom = 0.5 * (abs(l) + abs(r))
    si = abs(l - r) / denom * 100 if denom > 1e-8 else 0.0
    return {f'{prefix}_SI': round(si, 4)}


# ── Per-subject feature extraction ────────────────────────────────────────────

def extract_subject(sg, fg, fps=30):
    """
    Extract all features for one subject.

    Parameters
    ----------
    sg : DataFrame  – side-view keypoint rows for this subject
    fg : DataFrame  – front-view keypoint rows for this subject
    fps : int       – frames per second (default 30)

    Returns
    -------
    dict of feature_name → value
    """
    feats = {}

    # ── SIDE VIEW ─────────────────────────────────────────────────────────────

    # Hip angles (trunk-to-thigh angle at hip: Shoulder→Hip→Knee)
    r_hip = angle_3pts(pts(sg, 'RShoulder'), pts(sg, 'RHip'), pts(sg, 'RKnee'))
    l_hip = angle_3pts(pts(sg, 'LShoulder'), pts(sg, 'LHip'), pts(sg, 'LKnee'))
    feats.update(stat_features(r_hip, 'RHip', fps))
    feats.update(stat_features(l_hip, 'LHip', fps))
    feats.update(symmetry_index(l_hip, r_hip, 'Hip'))

    # Knee angles (Hip→Knee→Ankle; straight leg = 180°, full flexion < 90°)
    r_kn = angle_3pts(pts(sg, 'RHip'), pts(sg, 'RKnee'), pts(sg, 'RAnkle'))
    l_kn = angle_3pts(pts(sg, 'LHip'), pts(sg, 'LKnee'), pts(sg, 'LAnkle'))
    feats.update(stat_features(r_kn, 'RKnee', fps))
    feats.update(stat_features(l_kn, 'LKnee', fps))
    feats.update(symmetry_index(l_kn, r_kn, 'Knee'))

    # Ankle angles (Knee→Ankle→BigToe)
    r_ank = angle_3pts(pts(sg, 'RKnee'), pts(sg, 'RAnkle'), pts(sg, 'RBigToe'))
    l_ank = angle_3pts(pts(sg, 'LKnee'), pts(sg, 'LAnkle'), pts(sg, 'LBigToe'))
    feats.update(stat_features(r_ank, 'RAnkle', fps))
    feats.update(stat_features(l_ank, 'LAnkle', fps))
    feats.update(symmetry_index(l_ank, r_ank, 'Ankle'))

    # Dorsiplantar angles (angle of foot segment relative to horizontal)
    def dp_ang(heel, toe):
        v = toe - heel
        h = np.zeros_like(v)
        h[:, 0] = 1
        return np.abs(signed_angle_2d(h, v))

    r_dp = dp_ang(pts(sg, 'RHeel'), pts(sg, 'RBigToe'))
    l_dp = dp_ang(pts(sg, 'LHeel'), pts(sg, 'LBigToe'))
    feats.update(stat_features(r_dp, 'RDP', fps))
    feats.update(stat_features(l_dp, 'LDP', fps))
    feats.update(symmetry_index(l_dp, r_dp, 'DP'))

    # Trunk vertical oscillation (MHipY as proxy)
    trunk_y = sg['MHipY'].values.astype(float)
    feats.update(stat_features(trunk_y, 'TrunkY', fps))

    # ── FRONT VIEW ────────────────────────────────────────────────────────────

    # Step width (lateral ankle separation in pixels)
    sw = np.abs(fg['RAnkleX'].values - fg['LAnkleX'].values)
    feats.update(stat_features(sw, 'StepWidth', fps))

    # Hip adduction/abduction (signed angle: Shoulder-Hip vector vs Hip-Knee vector)
    l_add = signed_angle_2d(
        pts(fg, 'LShoulder') - pts(fg, 'LHip'),
        pts(fg, 'LKnee') - pts(fg, 'LHip'))
    r_add = signed_angle_2d(
        pts(fg, 'RShoulder') - pts(fg, 'RHip'),
        pts(fg, 'RKnee') - pts(fg, 'RHip'))
    feats.update(stat_features(l_add, 'LAddAbd', fps))
    feats.update(stat_features(r_add, 'RAddAbd', fps))
    feats.update(symmetry_index(l_add, r_add, 'AddAbd'))

    # Cadence and step count from ankle Y oscillation
    ry = fg['RAnkleY'].values
    ly = fg['LAnkleY'].values
    rp, _ = find_peaks(ry, distance=int(fps * 0.3))
    lp, _ = find_peaks(ly, distance=int(fps * 0.3))
    if len(rp) < 2:
        rp, _ = find_peaks(-ry, distance=int(fps * 0.3))
    if len(lp) < 2:
        lp, _ = find_peaks(-ly, distance=int(fps * 0.3))
    steps = len(rp) + len(lp)
    cadence = steps / (len(fg) / fps / 60)

    # Step and stride length (Euclidean distance between ankle centres)
    dists = np.linalg.norm(pts(fg, 'RAnkle') - pts(fg, 'LAnkle'), axis=1)
    feats.update(stat_features(dists, 'StepLen', fps))

    # Vertical CoM oscillation (MHipY from front view)
    feats.update(stat_features(fg['MHipY'].values, 'CoM_Y', fps))

    # Scalar summary features
    feats['Cadence'] = round(cadence, 4)
    feats['Steps'] = steps
    feats['StepLength'] = round(float(dists.mean()), 4)
    feats['StrideLength'] = round(float(dists.mean() * 2), 4)

    return feats


# ── Dataset-level extraction ───────────────────────────────────────────────────

def extract_all(side_csv, front_csv, fps=30):
    """
    Run extraction for every subject in the two CSVs.

    Returns a DataFrame with columns [Subject, <163 features>, Class].
    """
    ds = pd.read_csv(side_csv)
    df = pd.read_csv(front_csv)
    for d in (ds, df):
        if 'Label' in d.columns:
            d.rename(columns={'Label': 'Class'}, inplace=True)

    subjects = sorted(set(ds['Subject'].unique()) | set(df['Subject'].unique()))
    records = []
    for subj in subjects:
        sg = ds[ds['Subject'] == subj]
        fg = df[df['Subject'] == subj]
        if sg.empty or fg.empty:
            print(f'[SKIP] {subj} – missing data in one view')
            continue
        row = {'Subject': subj, 'Class': sg['Class'].iloc[0]}
        row.update(extract_subject(sg, fg, fps))
        records.append(row)

    out = pd.DataFrame(records)
    feature_cols = [c for c in out.columns if c not in ('Subject', 'Class')]
    out = out[['Subject'] + feature_cols + ['Class']]
    return out


# ── Verification helper ────────────────────────────────────────────────────────

def verify(side_csv, front_csv, reference_csv, fps=30):
    """
    Run extraction into memory and compare against *reference_csv*.
    Nothing is written to disk.  Prints a pass/fail report.
    """
    print(f'Running extraction against reference: {reference_csv}')
    extracted = extract_all(side_csv, front_csv, fps)
    reference = pd.read_csv(reference_csv)

    ok = True

    # 1. Shape check
    if extracted.shape != reference.shape:
        print(f'  SHAPE MISMATCH  extracted={extracted.shape}  reference={reference.shape}')
        ok = False
    else:
        print(f'  Shape OK: {extracted.shape}')

    # 2. Column names
    extra_cols = set(extracted.columns) - set(reference.columns)
    missing_cols = set(reference.columns) - set(extracted.columns)
    if extra_cols:
        print(f'  EXTRA columns in extracted  : {sorted(extra_cols)}')
        ok = False
    if missing_cols:
        print(f'  MISSING columns vs reference: {sorted(missing_cols)}')
        ok = False
    if not extra_cols and not missing_cols:
        print('  Columns OK')

    # 3. Numeric value comparison (shared columns only)
    shared = [c for c in extracted.columns if c in reference.columns
              and c not in ('Subject', 'Class')
              and pd.api.types.is_numeric_dtype(extracted[c])]

    mismatched = []
    for col in shared:
        try:
            pd.testing.assert_series_equal(
                extracted[col].reset_index(drop=True),
                reference[col].reset_index(drop=True),
                check_names=False,
                check_exact=False,
                atol=1e-4,
            )
        except AssertionError as e:
            mismatched.append((col, str(e).split('\n')[0]))

    if mismatched:
        print(f'  VALUE MISMATCHES in {len(mismatched)} column(s):')
        for col, msg in mismatched[:10]:
            print(f'    {col}: {msg}')
        if len(mismatched) > 10:
            print(f'    ... and {len(mismatched)-10} more')
        ok = False
    else:
        print(f'  Values OK across {len(shared)} numeric columns (atol=1e-4)')

    print()
    if ok:
        print('RESULT: PASS — extracted features match the reference CSV.')
    else:
        print('RESULT: FAIL — see mismatches above.')
    return ok


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='Extract rich gait features from pose keypoint CSVs.')
    parser.add_argument(
        '--side', default=os.path.join(root, 'Side Dataset 25KP New GC.csv'),
        help='Path to side-view keypoint CSV')
    parser.add_argument(
        '--front', default=os.path.join(root, 'Front Dataset 25KP New GC.csv'),
        help='Path to front-view keypoint CSV')
    parser.add_argument(
        '--out', default=os.path.join(root, 'outputs', 'gait_features_extracted.csv'),
        help='Output CSV path (default: outputs/gait_features_extracted.csv)')
    parser.add_argument(
        '--verify', action='store_true',
        help='Compare extraction output against gait_features_rich.csv; nothing written')
    parser.add_argument(
        '--reference', default=os.path.join(root, 'gait_features_rich.csv'),
        help='Reference CSV for --verify (default: gait_features_rich.csv)')
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()

    if args.verify:
        ok = verify(args.side, args.front, args.reference, args.fps)
        sys.exit(0 if ok else 1)

    out = extract_all(args.side, args.front, args.fps)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f'Saved: {args.out}')
    print(f'Shape: {out.shape}')
    print(f'Class distribution:\n{out["Class"].value_counts().to_string()}')
    feature_cols = [c for c in out.columns if c not in ('Subject', 'Class')]
    print(f'Feature count: {len(feature_cols)}')


if __name__ == '__main__':
    main()
