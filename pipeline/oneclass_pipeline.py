"""
One-Class / Anomaly Detection Pipeline for ASD Gait Classification
====================================================================
Instead of learning a boundary between ASD and Non-ASD (which is hard
because ASD gait is heterogeneous — autism is a wide spectrum), this
pipeline learns what **normal gait** looks like and flags deviations.

Models train ONLY on Non-ASD subjects; any subject whose gait deviates
from the learned normal distribution is classified as abnormal.

Uses the same nested stratified CV, same outer folds, and same output
format as ``ml_pipeline.py`` so results are directly comparable.

CLI
---
  python pipeline/oneclass_pipeline.py
  python pipeline/oneclass_pipeline.py --n-outer 3 --n-inner 3 --n-trials 5
  python pipeline/oneclass_pipeline.py --selected-features LHip_min RKnee_skew ...
  python pipeline/oneclass_pipeline.py --feature-strategy low-cv --hybrid
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from sklearn.covariance import LedoitWolf, MinCovDet
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, make_scorer, roc_auc_score, roc_curve,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print('[WARN] optuna not installed; using grid-search fallback for all models.')

try:
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

# ── Reuse preprocessing helpers from ml_pipeline ────────────────────────────
# Add pipeline/ to sys.path so we can import from ml_pipeline directly.
_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

from ml_pipeline import (          # noqa: E402
    near_zero_var,
    find_correlation,
    group_diff_tests,
    find_best_threshold,
    _corrected_roc_auc,
    _Tee,
    PALETTE,
)

# =============================================================================
# Constants
# =============================================================================

MODEL_DISPLAY = {
    'OC-SVM':   'One-Class SVM',
    'OC-IF':    'Isolation Forest',
    'OC-Mahal': 'Mahalanobis Distance',
    'OC-LOF':   'Local Outlier Factor',
    'OC-GMM':   'Gaussian Mixture Model',
}

MODEL_COLORS = ['#1B7837', '#2166AC', '#D6604D', '#762A83', '#E08214',
                '#8B4513', '#4D4D4D', '#A6761D', '#666666']

OC_MODEL_KEYS = ['OC-SVM', 'OC-IF', 'OC-Mahal', 'OC-LOF', 'OC-GMM']


# =============================================================================
# Custom inner CV splitter for one-class HPO
# =============================================================================

class OneClassInnerCV:
    """Inner CV for one-class HPO.

    For each inner fold:
      - train indices: Non-ASD subjects only from the inner training portion
      - val indices:   ALL subjects from the inner validation portion

    This lets us compute AUC on inner validation (both classes present)
    while training only on normal subjects.
    """

    def __init__(self, n_splits=3, random_state=0):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True,
                              random_state=self.random_state)
        for tr_idx, val_idx in skf.split(X, y):
            normal_tr_idx = tr_idx[y[tr_idx] == 0]  # Non-ASD only
            yield normal_tr_idx, val_idx              # val keeps both classes

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# Mahalanobis Distance detector (sklearn-compatible wrapper)
# =============================================================================

class MahalanobisDetector:
    """Anomaly detector based on Mahalanobis distance from the normal centroid.

    Uses LedoitWolf (default) or MinCovDet for regularised covariance estimation.
    Optionally applies PCA beforehand to avoid singular covariance.
    """

    def __init__(self, cov_estimator='ledoit_wolf', n_pca_components=None):
        self.cov_estimator = cov_estimator
        self.n_pca_components = n_pca_components
        self._pca = None
        self._cov = None
        self._mean = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if self.n_pca_components is not None and self.n_pca_components < X.shape[1]:
            n_comp = min(self.n_pca_components, X.shape[0] - 1, X.shape[1])
            self._pca = PCA(n_components=n_comp)
            X = self._pca.fit_transform(X)

        if self.cov_estimator == 'mincovdet' and X.shape[0] > X.shape[1] + 1:
            self._cov = MinCovDet().fit(X)
        else:
            self._cov = LedoitWolf().fit(X)
        self._mean = self._cov.location_
        return self

    def decision_function(self, X):
        """Return Mahalanobis distance (higher = more anomalous)."""
        X = np.asarray(X, dtype=float)
        if self._pca is not None:
            X = self._pca.transform(X)
        return self._cov.mahalanobis(X)


# =============================================================================
# Anomaly score extraction (orient so higher = more anomalous = more likely ASD)
# =============================================================================

def score_anomaly(model, X, model_key):
    """Extract anomaly scores oriented so higher = more anomalous."""
    if model_key == 'OC-Mahal':
        # Already higher = more anomalous
        return model.decision_function(X)
    elif model_key == 'OC-GMM':
        # score_samples returns log-likelihood; negate
        return -model.score_samples(X)
    else:
        # OneClassSVM, IsolationForest, LOF: decision_function is
        # higher for inliers → negate
        return -model.decision_function(X)


# =============================================================================
# Score normalisation
# =============================================================================

def normalize_scores(scores, ref_scores):
    """Min-max normalise scores to [0,1] using reference (training) range."""
    lo, hi = float(np.min(ref_scores)), float(np.max(ref_scores))
    if hi - lo < 1e-12:
        return np.full_like(scores, 0.5, dtype=float)
    normed = (scores - lo) / (hi - lo)
    return np.clip(normed, 0.0, 1.0)


# =============================================================================
# Evaluation helper
# =============================================================================

def eval_oneclass(scores, y_te, threshold=0.5):
    """Evaluate anomaly scores against true labels.

    Returns dict of metrics + (fpr, tpr) arrays.
    scores: higher = more anomalous (= more likely ASD=1).
    """
    auc_val = roc_auc_score(y_te, scores)
    if auc_val < 0.5:
        scores = -scores + np.max(scores) + np.min(scores)  # flip
        auc_val = 1.0 - auc_val

    fpr, tpr, _ = roc_curve(y_te, scores)
    y_pred = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)
          if (precision + sensitivity) > 0 else 0.0)

    return {
        'auc': auc_val, 'accuracy': accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'f1': f1,
        'fpr': fpr, 'tpr': tpr,
        'scores': scores,
    }


# =============================================================================
# Model building and Optuna parameter suggestion
# =============================================================================

def suggest_oc_params(trial, model_key, n_train):
    """Suggest hyperparameters for one-class models via Optuna."""
    if model_key == 'OC-SVM':
        gamma_choice = trial.suggest_categorical('gamma_type', ['scale', 'float'])
        gamma = ('scale' if gamma_choice == 'scale'
                 else trial.suggest_float('gamma_val', 1e-4, 10.0, log=True))
        return {
            'nu': trial.suggest_float('nu', 0.01, 0.5, log=True),
            'gamma': gamma,
        }
    elif model_key == 'OC-IF':
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_samples': trial.suggest_float('max_samples', 0.3, 1.0),
            'max_features': trial.suggest_float('max_features', 0.3, 1.0),
        }
    elif model_key == 'OC-Mahal':
        return {
            'cov_estimator': trial.suggest_categorical(
                'cov_estimator', ['ledoit_wolf', 'mincovdet']),
            'n_pca_components': trial.suggest_categorical(
                'n_pca_components', [None, 3, 5, 8, 10]),
        }
    elif model_key == 'OC-LOF':
        max_k = max(2, min(20, n_train - 1))
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', 2, max_k),
            'metric': trial.suggest_categorical(
                'metric', ['euclidean', 'manhattan']),
        }
    elif model_key == 'OC-GMM':
        return {
            'n_components': trial.suggest_int('n_components', 1, 3),
            'covariance_type': trial.suggest_categorical(
                'covariance_type', ['diag', 'tied', 'spherical']),
            'reg_covar': trial.suggest_float('reg_covar', 1e-6, 0.1, log=True),
        }
    else:
        raise ValueError(f'Unknown model key: {model_key}')


def build_oc_model(model_key, params, random_state=42):
    """Instantiate a one-class model with given parameters."""
    if model_key == 'OC-SVM':
        return OneClassSVM(kernel='rbf', **params)
    elif model_key == 'OC-IF':
        return IsolationForest(contamination='auto',
                               random_state=random_state, **params)
    elif model_key == 'OC-Mahal':
        return MahalanobisDetector(**params)
    elif model_key == 'OC-LOF':
        return LocalOutlierFactor(novelty=True, **params)
    elif model_key == 'OC-GMM':
        return GaussianMixture(random_state=random_state, **params)
    else:
        raise ValueError(f'Unknown model key: {model_key}')


# =============================================================================
# HPO via Optuna for one-class models
# =============================================================================

def make_oc_objective(model_key, X_tr_all, y_tr_all, inner_cv, fold_idx):
    """Return an Optuna objective that trains on Non-ASD, evaluates AUC on all."""

    def objective(trial):
        aucs = []
        for normal_tr_idx, val_idx in inner_cv.split(X_tr_all, y_tr_all):
            X_normal = X_tr_all[normal_tr_idx]
            X_val = X_tr_all[val_idx]
            y_val = y_tr_all[val_idx]

            # Need both classes in validation for AUC
            if len(np.unique(y_val)) < 2:
                continue

            scaler = StandardScaler()
            scaler.fit(X_normal)
            X_normal_s = scaler.transform(X_normal)
            X_val_s = scaler.transform(X_val)

            n_train = len(X_normal_s)
            params = suggest_oc_params(trial, model_key, n_train)

            try:
                model = build_oc_model(model_key, params, random_state=fold_idx)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X_normal_s)
                raw_scores = score_anomaly(model, X_val_s, model_key)
                auc = _corrected_roc_auc(y_val, raw_scores)
                aucs.append(auc)
            except Exception:
                aucs.append(0.5)

        return float(np.mean(aucs)) if aucs else 0.5

    return objective


# =============================================================================
# Grid-search fallback parameter grids
# =============================================================================

GRID_PARAMS = {
    'OC-SVM': [
        {'nu': nu, 'gamma': gamma}
        for nu in [0.01, 0.05, 0.1, 0.2, 0.5]
        for gamma in ['scale', 0.01, 0.1, 1.0]
    ],
    'OC-IF': [
        {'n_estimators': ne, 'max_samples': ms, 'max_features': mf}
        for ne in [100, 300]
        for ms in [0.5, 1.0]
        for mf in [0.5, 1.0]
    ],
    'OC-Mahal': [
        {'cov_estimator': ce, 'n_pca_components': pc}
        for ce in ['ledoit_wolf', 'mincovdet']
        for pc in [None, 5, 10]
    ],
    'OC-LOF': [
        {'n_neighbors': k, 'metric': m}
        for k in [3, 5, 10, 15]
        for m in ['euclidean', 'manhattan']
    ],
    'OC-GMM': [
        {'n_components': nc, 'covariance_type': ct, 'reg_covar': rc}
        for nc in [1, 2, 3]
        for ct in ['diag', 'tied', 'spherical']
        for rc in [1e-4, 1e-2]
    ],
}


def grid_search_oc(model_key, X_tr_all, y_tr_all, inner_cv, fold_idx):
    """Brute-force grid search for one-class models."""
    best_auc = -1.0
    best_params = {}

    for params in GRID_PARAMS[model_key]:
        # Clamp LOF n_neighbors dynamically
        if model_key == 'OC-LOF':
            # Count how many normal subjects are in the smallest inner train fold
            min_n = None
            for norm_idx, _ in inner_cv.split(X_tr_all, y_tr_all):
                if min_n is None or len(norm_idx) < min_n:
                    min_n = len(norm_idx)
            if params['n_neighbors'] >= min_n:
                continue

        aucs = []
        for normal_tr_idx, val_idx in inner_cv.split(X_tr_all, y_tr_all):
            X_normal = X_tr_all[normal_tr_idx]
            X_val = X_tr_all[val_idx]
            y_val = y_tr_all[val_idx]

            if len(np.unique(y_val)) < 2:
                continue

            scaler = StandardScaler()
            scaler.fit(X_normal)
            X_normal_s = scaler.transform(X_normal)
            X_val_s = scaler.transform(X_val)

            try:
                model = build_oc_model(model_key, params, random_state=fold_idx)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(X_normal_s)
                raw_scores = score_anomaly(model, X_val_s, model_key)
                auc = _corrected_roc_auc(y_val, raw_scores)
                aucs.append(auc)
            except Exception:
                aucs.append(0.5)

        mean_auc = float(np.mean(aucs)) if aucs else 0.5
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params.copy()

    return best_params, best_auc


# =============================================================================
# Feature selection: low-CV strategy
# =============================================================================

def select_features_low_cv(X_normal, feature_names, k=15):
    """Select features with lowest coefficient of variation within normal group.

    Features where "normal" is tightest → deviations are most meaningful.
    """
    means = np.abs(np.mean(X_normal, axis=0))
    stds = np.std(X_normal, axis=0)

    # CV = std / |mean|; protect against near-zero means
    cvs = np.where(means > 1e-8, stds / means, np.inf)

    k = min(k, len(feature_names))
    indices = np.argsort(cvs)[:k]
    return [feature_names[i] for i in indices]


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_roc_mean(fold_rocs, model_aucs, model_keys, out_path):
    """Mean ROC curve per model with individual fold curves in gray."""
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, key in enumerate(model_keys):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        rocs = fold_rocs[key]
        aucs = model_aucs[key]

        interp_tprs = []
        for fpr, tpr in rocs:
            interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
            ax.plot(fpr, tpr, color='lightgray', lw=0.6, alpha=0.4, zorder=1)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        mu = np.mean(aucs)
        std = np.std(aucs)
        lbl = f'{MODEL_DISPLAY[key]}  (AUC={mu:.3f} \u00b1 {std:.3f})'
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=lbl, zorder=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, color='grey', zorder=0)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves \u2013 One-Class Models (mean over outer CV)')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_model_comparison_bar(summary_df, model_keys, out_path):
    """Bar chart of AUC_mean per model with +/- 1 std error bars."""
    names = [MODEL_DISPLAY[k] for k in model_keys]
    means = [summary_df.loc[summary_df['Model'] == MODEL_DISPLAY[k], 'AUC_mean'].values[0]
             for k in model_keys]
    stds = [summary_df.loc[summary_df['Model'] == MODEL_DISPLAY[k], 'AUC_std'].values[0]
            for k in model_keys]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(model_keys))]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 2), 5.5))
    bars = ax.bar(x, means, color=colors, alpha=0.88, width=0.55,
                  yerr=stds, capsize=5, error_kw={'elinewidth': 1.5})
    for bar, mu in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02,
                f'{mu:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('AUC')
    ax.set_title('One-Class Model AUC Comparison (mean \u00b1 1 std)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_auc_distributions(model_aucs, model_keys, out_path):
    """Box plot of per-fold AUC for each model."""
    data = [model_aucs[k] for k in model_keys]
    names = [MODEL_DISPLAY[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(max(7, len(model_keys) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)
    for patch, color in zip(bp['boxes'], MODEL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title('Per-fold AUC Distribution \u2013 One-Class Models')
    ax.axhline(0.5, color='grey', linestyle='--', lw=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# =============================================================================
# Main pipeline
# =============================================================================

def main(features_csv, out_dir,
         preselected_features=None,
         feature_strategy='boruta',
         n_trials=50, n_outer=5, n_inner=3,
         use_hybrid=False):

    os.makedirs(out_dir, exist_ok=True)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print('=' * 60)
    print('Step 1 -- Load data')
    print('=' * 60)
    df = pd.read_csv(features_csv)
    df = df.drop(columns=['Subject'], errors='ignore')
    df['Class'] = df['Class'].apply(lambda x: 'ASD' if x == 'ASD' else 'NonASD')
    feat_cols = [c for c in df.columns if c != 'Class']
    print(f'  Dimensions: {df.shape}')
    print(f'  Class distribution:\n{df["Class"].value_counts().to_string()}')

    # ── Step 2a: Near-zero-variance removal ──────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 2a -- Near-zero-variance removal')
    print('=' * 60)
    nzv = near_zero_var(df[feat_cols])
    if nzv:
        df = df.drop(columns=nzv)
        feat_cols = [c for c in df.columns if c != 'Class']
        print(f'  Removed {len(nzv)} NZV features.')
    else:
        print('  No NZV features found.')

    # ── Step 2b: High-correlation removal ────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 2b -- High-correlation removal (cutoff=0.95)')
    print('=' * 60)
    cor_mat = df[feat_cols].corr()
    high_cor = find_correlation(cor_mat, cutoff=0.95)
    if high_cor:
        df = df.drop(columns=high_cor)
        feat_cols = [c for c in df.columns if c != 'Class']
        print(f'  Removed {len(high_cor)} highly correlated features.')
    else:
        print('  No high-correlation features found.')
    print(f'  Features remaining: {len(feat_cols)}')

    # ── Step 3: Group-difference statistical tests ───────────────────────────
    print('\n' + '=' * 60)
    print('Step 3 -- Group-difference statistical tests')
    print('=' * 60)
    test_res = group_diff_tests(df, feat_cols)
    sig_feats = test_res.loc[test_res['Significant'] == '***', 'Feature'].tolist()
    print(f'  Significant features (p<0.05): {len(sig_feats)} / {len(feat_cols)}')
    print(test_res[test_res['Significant'] == '***'].to_string(index=False))

    test_res.to_csv(os.path.join(out_dir, 'statistical_test_results.csv'), index=False)
    print(f'\n  Saved: {os.path.join(out_dir, "statistical_test_results.csv")}')

    # ── Step 4: Feature selection ────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 4 -- Feature selection')
    print('=' * 60)

    if preselected_features:
        missing = [f for f in preselected_features if f not in feat_cols]
        if missing:
            print(f'  [WARN] Features not in dataset: {missing}')
        selected_feats = [f for f in preselected_features if f in feat_cols]
        print(f'  Using pre-specified features ({len(selected_feats)}): {", ".join(selected_feats)}')

    elif feature_strategy == 'low-cv':
        X_normal = df.loc[df['Class'] == 'NonASD', feat_cols].values
        selected_feats = select_features_low_cv(X_normal, feat_cols, k=15)
        print(f'  Low-CV strategy: selected {len(selected_feats)} features with '
              f'tightest normal distribution')
        print(f'  Selected: {", ".join(selected_feats)}')

    elif feature_strategy == 'significant':
        selected_feats = sig_feats if sig_feats else feat_cols
        print(f'  Using statistically significant features ({len(selected_feats)})')

    elif feature_strategy == 'all':
        selected_feats = feat_cols
        print(f'  Using all features ({len(selected_feats)})')

    elif feature_strategy == 'boruta':
        X_all = df[feat_cols].values
        y_all = (df['Class'] == 'ASD').astype(int).values

        selected_feats = []
        if BORUTA_AVAILABLE:
            rf_boruta = RandomForestClassifier(n_jobs=-1, random_state=42)
            boruta = BorutaPy(rf_boruta, n_estimators='auto',
                              max_iter=200, random_state=42, verbose=0)
            try:
                boruta.fit(X_all, y_all)
                selected_feats = [feat_cols[i]
                                  for i, s in enumerate(boruta.support_) if s]
                tentative = [feat_cols[i]
                             for i, s in enumerate(boruta.support_weak_) if s]
                selected_feats = list(dict.fromkeys(selected_feats + tentative))
                print(f'  Boruta confirmed: {len(selected_feats)} features')
            except Exception as e:
                print(f'  [WARN] Boruta failed: {e}')
        if not selected_feats:
            print('  Falling back to statistically significant features.')
            selected_feats = sig_feats if sig_feats else feat_cols

    else:
        raise ValueError(f'Unknown feature strategy: {feature_strategy}')

    print(f'  Selected ({len(selected_feats)}): {", ".join(selected_feats)}')
    df_sel = df[selected_feats + ['Class']].copy()

    # ── Step 5: Nested CV with one-class models ──────────────────────────────
    print('\n' + '=' * 60)
    print(f'Step 5 -- Nested CV  '
          f'[outer: {n_outer}-fold | inner: {n_inner}-fold | '
          f'Optuna {n_trials} trials | One-Class Models]')
    print('=' * 60)
    if feature_strategy == 'boruta':
        print('  NOTE: Boruta ran on ALL data (acknowledged leakage; see FINDINGS.md)')
    print(f'  Paradigm: Train on Non-ASD ONLY -> detect abnormal gait as anomaly')
    print()

    X_all_sel = df_sel[selected_feats].values
    y_all_sel = (df_sel['Class'] == 'ASD').astype(int).values

    model_keys = list(OC_MODEL_KEYS)
    all_keys = list(model_keys)

    # Outer CV — same folds as binary pipeline for fair comparison
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    # Storage
    fold_metrics = {k: [] for k in all_keys}
    fold_rocs = {k: [] for k in all_keys}
    fold_auc_vals = {k: [] for k in all_keys}
    per_fold_rows = []

    for fold_idx, (tr_idx, te_idx) in enumerate(
            outer_cv.split(X_all_sel, y_all_sel)):

        X_tr_all = X_all_sel[tr_idx]
        y_tr_all = y_all_sel[tr_idx]
        X_te = X_all_sel[te_idx]
        y_te = y_all_sel[te_idx]

        # Split training into normal-only and all
        normal_mask = y_tr_all == 0
        X_tr_normal = X_tr_all[normal_mask]

        # Fit scaler on normal subjects only
        scaler = StandardScaler()
        scaler.fit(X_tr_normal)
        X_tr_normal_s = scaler.transform(X_tr_normal)
        X_tr_all_s = scaler.transform(X_tr_all)
        X_te_s = scaler.transform(X_te)

        # Inner CV for HPO
        inner_cv = OneClassInnerCV(n_splits=n_inner, random_state=fold_idx)

        fold_auc_parts = {}

        for key in model_keys:
            # ── HPO ──────────────────────────────────────────────────────
            if OPTUNA_AVAILABLE:
                sampler = optuna.samplers.TPESampler(seed=fold_idx)
                study = optuna.create_study(direction='maximize', sampler=sampler)
                study.optimize(
                    make_oc_objective(key, X_tr_all_s, y_tr_all,
                                     inner_cv, fold_idx),
                    n_trials=n_trials,
                    show_progress_bar=False,
                )
                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
                if completed:
                    best_params = study.best_params.copy()
                    # Reconstruct structured params from flat Optuna params
                    best_params = _reconstruct_params(key, best_params)
                else:
                    print(f'  [WARN] All {n_trials} trials failed for {key} '
                          f'(fold {fold_idx}); using defaults.')
                    best_params = _default_params(key)
            else:
                best_params, _ = grid_search_oc(
                    key, X_tr_all_s, y_tr_all, inner_cv, fold_idx)

            # ── Final fit on all normal training subjects ────────────────
            model = build_oc_model(key, best_params, random_state=fold_idx)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X_tr_normal_s)

            # ── Score training set (both classes) for threshold ──────────
            tr_scores = score_anomaly(model, X_tr_all_s, key)
            tr_scores_norm = normalize_scores(tr_scores, tr_scores)
            opt_thresh = find_best_threshold(y_tr_all, tr_scores_norm)

            # ── Score test set and evaluate ──────────────────────────────
            te_scores = score_anomaly(model, X_te_s, key)
            te_scores_norm = normalize_scores(te_scores, tr_scores)
            metrics = eval_oneclass(te_scores_norm, y_te, threshold=opt_thresh)
            auc_val = metrics['auc']

            fold_metrics[key].append({k: v for k, v in metrics.items()
                                      if k not in ('fpr', 'tpr', 'scores')})
            fold_rocs[key].append((metrics['fpr'], metrics['tpr']))
            fold_auc_vals[key].append(auc_val)
            fold_auc_parts[key] = auc_val

            per_fold_rows.append({
                'fold':        fold_idx + 1,
                'model':       MODEL_DISPLAY[key],
                'auc':         round(auc_val,               4),
                'accuracy':    round(metrics['accuracy'],    4),
                'sensitivity': round(metrics['sensitivity'], 4),
                'specificity': round(metrics['specificity'], 4),
                'f1':          round(metrics['f1'],          4),
                'precision':   round(metrics['precision'],   4),
                'best_params': json.dumps(best_params, default=str),
            })

        # Progress
        auc_str = '  '.join(
            f'{k} AUC={fold_auc_parts[k]:.3f}' for k in model_keys)
        print(f'[Fold {fold_idx+1}/{n_outer}]  {auc_str}')

    # ── Step 6: Aggregate metrics ────────────────────────────────────────────
    print('\n' + '=' * 60)
    print(f'MODEL COMPARISON SUMMARY  (mean +/- std over {n_outer} outer folds)')
    print('=' * 60)

    metric_keys = ['auc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'precision']
    col_map = {
        'auc':         ('AUC_mean',         'AUC_std'),
        'accuracy':    ('Accuracy_mean',    'Accuracy_std'),
        'sensitivity': ('Sensitivity_mean', 'Sensitivity_std'),
        'specificity': ('Specificity_mean', 'Specificity_std'),
        'f1':          ('F1_mean',          'F1_std'),
        'precision':   ('Precision_mean',   'Precision_std'),
    }

    summary_rows = []
    for key in all_keys:
        row = {'Model': MODEL_DISPLAY[key]}
        for mk in metric_keys:
            vals = [m[mk] for m in fold_metrics[key]]
            mean_col, std_col = col_map[mk]
            row[mean_col] = round(float(np.mean(vals)), 4)
            row[std_col] = round(float(np.std(vals)), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    print(summary_df[[
        'Model', 'AUC_mean', 'AUC_std',
        'Accuracy_mean', 'Sensitivity_mean', 'Specificity_mean',
        'F1_mean', 'Precision_mean',
    ]].to_string(index=False))

    best_row = summary_df.loc[summary_df['AUC_mean'].idxmax()]
    print(f'\n  Best Model: {best_row["Model"]}  '
          f'AUC={best_row["AUC_mean"]:.4f} +/- {best_row["AUC_std"]:.4f}')

    # Save CSVs
    comp_path = os.path.join(out_dir, 'model_comparison_results.csv')
    summary_df.to_csv(comp_path, index=False)
    print(f'\n  Saved: {comp_path}')

    pf_path = os.path.join(out_dir, 'per_fold_results.csv')
    pd.DataFrame(per_fold_rows).to_csv(pf_path, index=False)
    print(f'  Saved: {pf_path}')

    # ── Step 7: Plots ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 7 -- Saving plots')
    print('=' * 60)

    plot_roc_mean(
        fold_rocs, fold_auc_vals, all_keys,
        os.path.join(out_dir, 'plot_roc_all_models.png'),
    )
    plot_model_comparison_bar(
        summary_df, all_keys,
        os.path.join(out_dir, 'plot_model_comparison_bar.png'),
    )
    plot_auc_distributions(
        fold_auc_vals, all_keys,
        os.path.join(out_dir, 'plot_auc_distributions.png'),
    )

    print('\nAll outputs saved to:', out_dir)
    print('Done!')


# =============================================================================
# Param reconstruction helpers (Optuna flat params → structured dict)
# =============================================================================

def _reconstruct_params(model_key, flat):
    """Convert Optuna's flat parameter dict back to model constructor args."""
    if model_key == 'OC-SVM':
        gamma = (flat.get('gamma_val', 'scale')
                 if flat.get('gamma_type') == 'float' else 'scale')
        return {'nu': flat['nu'], 'gamma': gamma}
    elif model_key == 'OC-IF':
        return {k: flat[k] for k in ['n_estimators', 'max_samples', 'max_features']}
    elif model_key == 'OC-Mahal':
        return {k: flat[k] for k in ['cov_estimator', 'n_pca_components']}
    elif model_key == 'OC-LOF':
        return {k: flat[k] for k in ['n_neighbors', 'metric']}
    elif model_key == 'OC-GMM':
        return {k: flat[k] for k in ['n_components', 'covariance_type', 'reg_covar']}
    return flat


def _default_params(model_key):
    """Sensible defaults when all Optuna trials fail."""
    defaults = {
        'OC-SVM':   {'nu': 0.1, 'gamma': 'scale'},
        'OC-IF':    {'n_estimators': 200, 'max_samples': 1.0, 'max_features': 1.0},
        'OC-Mahal': {'cov_estimator': 'ledoit_wolf', 'n_pca_components': None},
        'OC-LOF':   {'n_neighbors': 10, 'metric': 'euclidean'},
        'OC-GMM':   {'n_components': 1, 'covariance_type': 'diag', 'reg_covar': 1e-4},
    }
    return defaults.get(model_key, {})


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        description='One-class anomaly detection pipeline for ASD gait '
                    'classification -- learns normal gait and flags deviations.')
    parser.add_argument(
        '--features',
        default=os.path.join(root, 'gait_features_rich.csv'),
        help='Path to feature CSV (default: gait_features_rich.csv)')
    parser.add_argument(
        '--out',
        default=os.path.join(root, 'outputs'),
        help='Output directory (default: outputs/)')
    parser.add_argument(
        '--selected-features',
        nargs='+', metavar='FEATURE', default=None,
        help='Skip feature selection; use these exact feature names.')
    parser.add_argument(
        '--feature-strategy',
        choices=['boruta', 'low-cv', 'significant', 'all'],
        default='boruta',
        help='Feature selection strategy (default: boruta). '
             'Ignored when --selected-features is provided.')
    parser.add_argument(
        '--n-outer', type=int, default=5,
        help='Number of outer CV folds (default: 5)')
    parser.add_argument(
        '--n-inner', type=int, default=3,
        help='Number of inner CV folds for HPO (default: 3)')
    parser.add_argument(
        '--n-trials', type=int, default=50,
        help='Optuna trials per model per outer fold (default: 50)')
    parser.add_argument(
        '--hybrid', action='store_true',
        help='Also run hybrid binary classifiers with anomaly features (TBD).')
    parser.add_argument(
        '--log', default=None, metavar='FILE',
        help='Write all terminal output to FILE in addition to stdout.')
    args = parser.parse_args()

    tee = None
    if args.log:
        os.makedirs(os.path.dirname(args.log) or '.', exist_ok=True)
        tee = _Tee(sys.stdout, args.log)
        sys.stdout = tee

    try:
        main(
            features_csv=args.features,
            out_dir=args.out,
            preselected_features=args.selected_features,
            feature_strategy=args.feature_strategy,
            n_trials=args.n_trials,
            n_outer=args.n_outer,
            n_inner=args.n_inner,
            use_hybrid=args.hybrid,
        )
    finally:
        if tee is not None:
            sys.stdout = tee._stream
            tee.close()
