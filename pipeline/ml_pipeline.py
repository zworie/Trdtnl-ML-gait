"""
Traditional ML Pipeline for ASD Gait Classification
=====================================================
Python conversion of thesis.R — nested stratified cross-validation.

Evaluation framework: Nested Stratified CV (outer × inner folds)
  Outer loop (default 5-fold): each fold's ~14 test subjects serve as the
  held-out evaluation set.  All 72 subjects appear as test subjects exactly
  once.  Both class proportions are preserved in every outer fold.
  Inner loop (default 5-fold): stratified HPO CV on the outer training set
  (~58 subjects).  Class proportions are preserved in every inner fold.

Steps:
  1.  Load gait_features_rich.csv
  2.  Remove near-zero-variance features  (caret::nearZeroVar)
  3.  Remove highly correlated features   (caret::findCorrelation, cutoff=0.95)
  4.  Group-difference statistical tests  (Shapiro-Wilk -> t-test or Wilcoxon)
  5.  Boxplots of significant features
  6.  Boruta feature selection            (fallback: use significant features)
  7.  Outer N-fold stratified CV loop
  8.  For each outer fold: inner M-fold stratified Optuna HPO on training set
  9.  StandardScaler + SMOTE inside ImbPipeline (no leakage into outer test fold)
  10. Evaluate on outer test fold: AUC, Accuracy, Sensitivity, Specificity, F1, Precision
  11. Aggregate metrics: mean +/- std over N outer folds
  12. Plots: ROC (mean + per-fold), bar chart, RF importance, AUC box plots
  13. Save model_comparison_results.csv, per_fold_results.csv

Faithfulness notes
------------------
- Boruta and statistical tests are run on the FULL dataset (same as R code),
  before the outer CV loop.  This is a data-leakage issue documented in
  FINDINGS.md but replicated here for reproducibility.
- SMOTE runs inside each inner fold (ImbPipeline) — prevents synthetic sample
  leakage into the outer test fold.
- All random seeds derive from the outer fold index; Optuna studies seeded
  as fold_idx so each fold's search is independently reproducible.
- XGBoost is an extension present in thesis.R but absent from published
  thesis results.  Included here for completeness.

Usage
-----
  python pipeline/ml_pipeline.py
  python pipeline/ml_pipeline.py --selected-features LHip_min RKnee_skew
  python pipeline/ml_pipeline.py --n-trials 20 --n-outer 5 --n-inner 5
"""

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# Optional imports
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    print('[WARN] boruta not installed; will fall back to statistical tests.')

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print('[WARN] xgboost not installed; XGBoost model will be skipped.')


# =============================================================================
# Constants / display names
# =============================================================================

MODEL_DISPLAY = {
    'LR':  'Logistic Regression (Elastic Net)',
    'RF':  'Random Forest',
    'SVM': 'SVM (RBF Kernel)',
    'LDA': 'LDA (Ledoit-Wolf)',
    'XGB': 'XGBoost',
}

PALETTE      = {'ASD': '#D6604D', 'NonASD': '#4393C3'}
MODEL_COLORS = ['#1B7837', '#2166AC', '#D6604D', '#762A83', '#E08214', '#8B4513']

# Grid search parameter grids for LR and SVM.
# Keys use the 'model__' pipeline prefix so GridSearchCV can set them directly.
PARAM_GRID = {
    'LR': {
        'model__C':        [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        'model__l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
    },
    'SVM': {
        'model__C':     [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
        'model__gamma': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    },
}


# =============================================================================
# Corrected AUC scorer  (replicates R pROC auto-direction)
# =============================================================================

def _corrected_roc_auc(y_true, y_score):
    s = roc_auc_score(y_true, y_score)
    return max(s, 1.0 - s)

corrected_auc_scorer = make_scorer(_corrected_roc_auc, needs_proba=True)


# =============================================================================
# Steps 1-2: caret-compatible NZV and correlation helpers
# =============================================================================

def near_zero_var(df, freq_cut=19.0, unique_cut=10.0):
    """
    Replicate caret::nearZeroVar.

    A feature is near-zero-variance when:
      zeroVar  (only one unique value)
      OR
      (freq_ratio > freq_cut  AND  pct_unique < unique_cut)

    This matches the caret source:
      which(zeroVar | (freqRatio > freqCut & percentUnique < uniqueCut))

    Returns list of column names to remove.
    """
    to_remove = []
    n = len(df)
    for col in df.columns:
        vc = df[col].value_counts()
        if len(vc) <= 1:
            to_remove.append(col)
            continue
        freq_ratio = vc.iloc[0] / vc.iloc[1]
        pct_unique = (len(vc) / n) * 100
        if freq_ratio > freq_cut and pct_unique < unique_cut:
            to_remove.append(col)
    return to_remove


def find_correlation(cor_mat, cutoff=0.95):
    """
    Replicate caret::findCorrelation_fast (used when ncol >= 100).

    Iterates through column pairs in index order (i < j).  For each pair
    with |correlation| > cutoff, the variable with the higher mean absolute
    correlation to all remaining variables is marked for deletion.  When
    variable i is deleted, the inner loop breaks (matches R's ``break``).

    Returns list of column names to remove.
    """
    cols = list(cor_mat.columns)
    n = len(cols)
    abs_cor = cor_mat.abs().values.copy()
    np.fill_diagonal(abs_cor, 0)

    deleted = [False] * n
    for i in range(n - 1):
        if deleted[i]:
            continue
        for j in range(i + 1, n):
            if deleted[j]:
                continue
            if abs_cor[i, j] > cutoff:
                mean_i = abs_cor[i, :].mean()
                mean_j = abs_cor[j, :].mean()
                if mean_i >= mean_j:
                    deleted[i] = True
                    abs_cor[i, :] = 0
                    abs_cor[:, i] = 0
                    break
                else:
                    deleted[j] = True
                    abs_cor[j, :] = 0
                    abs_cor[:, j] = 0
    return [cols[k] for k in range(n) if deleted[k]]


# =============================================================================
# Step 3: Statistical tests
# =============================================================================

def group_diff_tests(df, feat_cols, class_col='Class',
                     pos_class='ASD', neg_class='NonASD'):
    """
    Replicate R step 3: Shapiro-Wilk normality test -> t-test or Wilcoxon.

    For each feature:
      - Shapiro-Wilk on the full sample
      - If SW p > 0.05 (approx. normal): Welch t-test
      - Otherwise: Mann-Whitney U (= Wilcoxon rank-sum)

    Returns a DataFrame with columns:
      Feature, Test, p_value, Significant
    """
    pos = df[df[class_col] == pos_class]
    neg = df[df[class_col] == neg_class]
    rows = []
    for col in feat_cols:
        x = df[col].dropna().values
        try:
            _, sw_p = shapiro(x)
        except Exception:
            sw_p = np.nan

        if not np.isnan(sw_p) and sw_p > 0.05:
            _, p = ttest_ind(pos[col].dropna(), neg[col].dropna(),
                             equal_var=False)
            test = 't-test'
        else:
            _, p = mannwhitneyu(pos[col].dropna(), neg[col].dropna(),
                                alternative='two-sided')
            test = 'Wilcoxon'

        rows.append({'Feature': col, 'Test': test,
                     'p_value': round(float(p), 4),
                     'Significant': '***' if p < 0.05 else 'ns'})
    return pd.DataFrame(rows)


# =============================================================================
# Step 4: Boxplots
# =============================================================================

def plot_sig_boxplots(df, sig_feats, out_path):
    """Boxplots for up to 9 significant features (replicates R step 4)."""
    feats = sig_feats[:min(9, len(sig_feats))]
    n = len(feats)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        for cls, grp in df.groupby('Class'):
            ax.boxplot(grp[feat].dropna(),
                       positions=[list(df['Class'].unique()).index(cls)],
                       widths=0.5, patch_artist=True,
                       boxprops=dict(facecolor=PALETTE.get(cls, '#888888'), alpha=0.85),
                       medianprops=dict(color='black', linewidth=1.5),
                       flierprops=dict(marker='o', markersize=4, alpha=0.5))
        ax.set_xticks([0, 1])
        ax.set_xticklabels(list(df['Class'].unique()))
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Significant Features: ASD vs Non-ASD', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')



# =============================================================================
# Optuna helpers: suggest_params, build_pipe
# =============================================================================

def suggest_params(trial, model_key):
    """Return a dict of model parameters suggested by the Optuna trial."""
    if model_key == 'LR':
        return {
            'C':        trial.suggest_float('C', 1e-3, 1e3, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        }
    elif model_key == 'RF':
        return {
            'max_features':       trial.suggest_float('max_features', 0.1, 1.0),
            'min_samples_leaf':   trial.suggest_int('min_samples_leaf', 1, 15),
            'max_depth':          trial.suggest_int('max_depth', 3, 20),
            # n_estimators fixed at 200 during search, bumped to 1000 for final fit
            'n_estimators': 200,
        }
    elif model_key == 'SVM':
        return {
            'C':     trial.suggest_float('C', 1e-3, 1e2, log=True),
            'gamma': trial.suggest_float('gamma', 1e-4, 1e1, log=True),
        }
    elif model_key == 'XGB':
        return {
            'n_estimators':    trial.suggest_int('n_estimators', 50, 300),
            'max_depth':       trial.suggest_int('max_depth', 2, 8),
            'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample':       trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':       trial.suggest_float('reg_alpha', 1e-4, 1.0, log=True),
            'reg_lambda':      trial.suggest_float('reg_lambda', 1e-4, 1.0, log=True),
        }
    else:
        raise ValueError(f'Unknown model_key: {model_key}')


def build_pipe(model_key, params, rng=42):
    """
    Build an ImbPipeline with StandardScaler -> SMOTE -> model.
    params keys are raw model parameter names (no 'model__' prefix).
    """
    if model_key == 'LR':
        estimator = LogisticRegression(
            penalty='elasticnet', solver='saga',
            max_iter=2000, random_state=rng,
            C=params.get('C', 1.0),
            l1_ratio=params.get('l1_ratio', 0.5),
        )
    elif model_key == 'RF':
        estimator = RandomForestClassifier(
            random_state=rng, n_jobs=-1,
            n_estimators=params.get('n_estimators', 200),
            max_features=params.get('max_features', 'sqrt'),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            max_depth=params.get('max_depth', None),
        )
    elif model_key == 'SVM':
        estimator = SVC(
            kernel='rbf', probability=True, random_state=rng,
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
        )
    elif model_key == 'LDA':
        # Ledoit-Wolf automatic shrinkage — no hyperparameters to tune.
        # Works well for small-sample, low-feature-count datasets.
        estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    elif model_key == 'XGB':
        if not XGB_AVAILABLE:
            raise RuntimeError('XGBoost not installed')
        estimator = XGBClassifier(
            eval_metric='logloss', verbosity=0,
            random_state=rng, n_jobs=-1,
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            subsample=params.get('subsample', 0.8),
            colsample_bytree=params.get('colsample_bytree', 0.8),
            reg_alpha=params.get('reg_alpha', 0.0),
            reg_lambda=params.get('reg_lambda', 1.0),
        )
    else:
        raise ValueError(f'Unknown model_key: {model_key}')

    pipe = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote',  SMOTE(random_state=rng, sampling_strategy=1.0)),
        ('model',  estimator),
    ])
    return pipe


def make_objective(model_key, X_tr, y_tr, inner_cv, rng):
    """Return an Optuna objective function for the given model and fold data."""
    def objective(trial):
        params = suggest_params(trial, model_key)
        pipe   = build_pipe(model_key, params, rng)
        # n_jobs=1: avoids joblib pool corruption when called from Optuna's loop.
        # error_score=0.5: failed folds return 0.5 (random chance) rather than NaN.
        scores = cross_val_score(
            pipe, X_tr, y_tr,
            cv=inner_cv,
            scoring=corrected_auc_scorer,
            n_jobs=1,
            error_score=0.5,
        )
        return float(np.nanmean(scores))
    return objective


# =============================================================================
# Per-fold evaluation helper
# =============================================================================

def eval_holdout(final_pipe, X_te, y_te):
    """
    Evaluate a fitted pipeline on the held-out test set.
    Returns dict of metrics and (fpr, tpr) arrays for ROC plotting.
    """
    y_prob = final_pipe.predict_proba(X_te)[:, 1]
    y_pred = final_pipe.predict(X_te)

    auc_val = roc_auc_score(y_te, y_prob)
    if auc_val < 0.5:
        y_prob = 1.0 - y_prob
        auc_val = 1.0 - auc_val

    fpr, tpr, _ = roc_curve(y_te, y_prob)

    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)

    return {
        'auc': auc_val, 'accuracy': accuracy,
        'sensitivity': sensitivity, 'specificity': specificity,
        'precision': precision, 'f1': f1,
        'fpr': fpr, 'tpr': tpr,
    }


# =============================================================================
# Plotting helpers (Steps 12-13)
# =============================================================================

def plot_rf_importance(rf_estimator, feature_names, out_path):
    """Horizontal bar chart of RF feature importances (Gini)."""
    imp   = rf_estimator.feature_importances_
    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(7, max(4, len(feature_names) * 0.4)))
    ax.barh(np.array(feature_names)[order], imp[order],
            color='#4393C3', edgecolor='white')
    ax.set_xlabel('Mean Decrease in Gini Impurity')
    ax.set_title('Random Forest – Variable Importance')
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_roc_mean(fold_rocs, model_aucs, model_keys, out_path):
    """
    Mean ROC curve per model (bold) with all individual fold curves as
    light gray lines.  fold_rocs[key] = list of (fpr, tpr) per fold.
    model_aucs[key] = list of per-fold AUC values.
    """
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax  = plt.subplots(figsize=(7, 6))

    for i, key in enumerate(model_keys):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        rocs  = fold_rocs[key]
        aucs  = model_aucs[key]

        interp_tprs = []
        for fpr, tpr in rocs:
            interp_tprs.append(np.interp(mean_fpr, fpr, tpr))
            ax.plot(fpr, tpr, color='lightgray', lw=0.6, alpha=0.4, zorder=1)

        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[0]  = 0.0
        mean_tpr[-1] = 1.0
        mu  = np.mean(aucs)
        std = np.std(aucs)
        lbl = f'{MODEL_DISPLAY[key]}  (AUC={mu:.3f} ± {std:.3f})'
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=lbl, zorder=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, color='grey', zorder=0)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – Mean over Outer CV Folds')
    ax.legend(loc='lower right', fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_model_comparison_bar(summary_df, model_keys, out_path):
    """Grouped bar chart of AUC_mean per model with ± 1 std error bars."""
    names  = [MODEL_DISPLAY[k] for k in model_keys]
    means  = [summary_df.loc[summary_df['Model'] == MODEL_DISPLAY[k], 'AUC_mean'].values[0]
              for k in model_keys]
    stds   = [summary_df.loc[summary_df['Model'] == MODEL_DISPLAY[k], 'AUC_std'].values[0]
              for k in model_keys]
    colors = [MODEL_COLORS[i % len(MODEL_COLORS)] for i in range(len(model_keys))]

    x   = np.arange(len(names))
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
    ax.set_title('Model AUC Comparison (mean ± 1 std, outer CV folds)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_auc_distributions(model_aucs, model_keys, out_path):
    """Box plot of per-fold AUC for each model."""
    data  = [model_aucs[k] for k in model_keys]
    names = [MODEL_DISPLAY[k] for k in model_keys]

    fig, ax = plt.subplots(figsize=(max(7, len(model_keys) * 2), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True)
    for patch, color in zip(bp['boxes'], MODEL_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('AUC')
    ax.set_title('Per-seed AUC Distribution by Model')
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
         n_trials=50, n_outer=5, n_inner=5):

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Load data
    # -------------------------------------------------------------------------
    print('=' * 60)
    print('Step 1 – Load data')
    print('=' * 60)
    df = pd.read_csv(features_csv)
    df = df.drop(columns=['Subject'], errors='ignore')
    df['Class'] = df['Class'].apply(lambda x: 'ASD' if x == 'ASD' else 'NonASD')
    feat_cols = [c for c in df.columns if c != 'Class']
    print(f'  Dimensions: {df.shape}')
    print(f'  Class distribution:\n{df["Class"].value_counts().to_string()}')

    # -------------------------------------------------------------------------
    # Step 2a: Near-zero-variance removal
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 2a – Near-zero-variance removal')
    print('=' * 60)
    nzv = near_zero_var(df[feat_cols])
    if nzv:
        df = df.drop(columns=nzv)
        feat_cols = [c for c in df.columns if c != 'Class']
        print(f'  Removed {len(nzv)} NZV features.')
    else:
        print('  No NZV features found.')

    # -------------------------------------------------------------------------
    # Step 2b: High-correlation removal
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 2b – High-correlation removal (cutoff=0.95)')
    print('=' * 60)
    cor_mat  = df[feat_cols].corr()
    high_cor = find_correlation(cor_mat, cutoff=0.95)
    if high_cor:
        df = df.drop(columns=high_cor)
        feat_cols = [c for c in df.columns if c != 'Class']
        print(f'  Removed {len(high_cor)} highly correlated features.')
    else:
        print('  No high-correlation features found.')
    print(f'  Features remaining: {len(feat_cols)}')

    # -------------------------------------------------------------------------
    # Step 3: Group-difference statistical tests
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 3 – Group-difference statistical tests')
    print('=' * 60)
    test_res  = group_diff_tests(df, feat_cols)
    sig_feats = test_res.loc[test_res['Significant'] == '***', 'Feature'].tolist()
    print(f'  Significant features (p<0.05): {len(sig_feats)} / {len(feat_cols)}')
    print(test_res[test_res['Significant'] == '***'].to_string(index=False))
    test_res.to_csv(os.path.join(out_dir, 'statistical_test_results.csv'), index=False)
    print(f'\n  Saved: {os.path.join(out_dir, "statistical_test_results.csv")}')

    # -------------------------------------------------------------------------
    # Step 4: Boxplots of significant features
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 4 – Boxplots of significant features')
    print('=' * 60)
    if len(sig_feats) >= 2:
        plot_sig_boxplots(df, sig_feats,
                          os.path.join(out_dir, 'plot_sig_features_boxplot.png'))
    else:
        print('  Fewer than 2 significant features; skipping boxplot.')

    # -------------------------------------------------------------------------
    # Step 5: Boruta feature selection
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 5 – Boruta feature selection (max_iter=200)')
    print('=' * 60)

    if preselected_features:
        missing = [f for f in preselected_features if f not in feat_cols]
        if missing:
            print(f'  [WARN] Features not in dataset (may have been removed by NZV/correlation): {missing}')
        selected_feats = [f for f in preselected_features if f in feat_cols]
        print(f'  Using pre-specified features ({len(selected_feats)}): {", ".join(selected_feats)}')
        print('  (Boruta skipped — feature list supplied via --selected-features)')
    else:
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
                selected_feats = []
        else:
            print('  Boruta not available.')

        if not selected_feats:
            print('  Falling back to statistically significant features.')
            selected_feats = sig_feats if sig_feats else feat_cols

    print(f'  Selected ({len(selected_feats)}): {", ".join(selected_feats)}')
    df_sel = df[selected_feats + ['Class']].copy()

    # -------------------------------------------------------------------------
    # Steps 6-11: Nested stratified CV with Optuna HPO
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'Steps 6–11 – Nested CV  '
          f'[outer: {n_outer}-fold | inner: {n_inner}-fold | '
          f'GridSearch (LR/SVM) | Optuna {n_trials} trials (RF/XGB) | LDA: none]')
    print('=' * 60)
    print('  NOTE: Boruta ran on ALL data (acknowledged leakage; see FINDINGS.md)')
    print()

    X_all_sel = df_sel[selected_feats].values
    y_all_sel = (df_sel['Class'] == 'ASD').astype(int).values

    # Determine which models to run
    model_keys = ['LR', 'RF', 'SVM', 'LDA']
    if XGB_AVAILABLE:
        model_keys.append('XGB')

    # Outer CV — stratified so every fold preserves the ASD/NonASD ratio
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    # Storage
    fold_metrics  = {k: [] for k in model_keys}
    fold_rocs     = {k: [] for k in model_keys}
    fold_auc_vals = {k: [] for k in model_keys}
    fold_pipes    = {k: [] for k in model_keys}  # for RF importance

    per_fold_rows = []

    for fold_idx, (tr_idx, te_idx) in enumerate(
            outer_cv.split(X_all_sel, y_all_sel)):

        X_tr, X_te = X_all_sel[tr_idx], X_all_sel[te_idx]
        y_tr, y_te = y_all_sel[tr_idx], y_all_sel[te_idx]

        # Inner CV — stratified, seeded per fold for reproducibility
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                   random_state=fold_idx)

        fold_auc_parts = {}

        for key in model_keys:
            if key == 'LDA':
                # No HPO — Ledoit-Wolf shrinkage is fully automatic.
                final_pipe  = build_pipe('LDA', {}, fold_idx)
                final_pipe.fit(X_tr, y_tr)
                best_params = {}

            elif key in ('LR', 'SVM'):
                # GridSearchCV — efficient for 2-parameter grids.
                # inner_cv is stratified, n_jobs=1 avoids pool issues inside fold loop.
                # refit=True refits the best estimator on the full outer training fold.
                base_pipe  = build_pipe(key, {}, fold_idx)
                gs = GridSearchCV(
                    base_pipe, PARAM_GRID[key],
                    cv=inner_cv,
                    scoring=corrected_auc_scorer,
                    refit=True,
                    n_jobs=1,
                )
                gs.fit(X_tr, y_tr)
                final_pipe  = gs.best_estimator_
                best_params = {k.replace('model__', ''): v
                               for k, v in gs.best_params_.items()}

            else:
                # Optuna TPE — efficient for higher-dimensional spaces (RF: 3, XGB: 7).
                sampler = optuna.samplers.TPESampler(seed=fold_idx)
                study   = optuna.create_study(direction='maximize', sampler=sampler)
                study.optimize(
                    make_objective(key, X_tr, y_tr, inner_cv, fold_idx),
                    n_trials=n_trials,
                    show_progress_bar=False,
                )

                completed = [t for t in study.trials
                             if t.state == optuna.trial.TrialState.COMPLETE]
                if not completed:
                    print(f'  [WARN] All {n_trials} trials failed for {key} '
                          f'(fold {fold_idx}); using default params.')
                    best_params = {}
                else:
                    best_params = study.best_params.copy()

                # Refit on full outer training fold; RF bumped to 1000 trees
                final_params = best_params.copy()
                if key == 'RF':
                    final_params['n_estimators'] = 1000
                final_pipe = build_pipe(key, final_params, fold_idx)
                final_pipe.fit(X_tr, y_tr)

            # Evaluate on outer test fold
            metrics = eval_holdout(final_pipe, X_te, y_te)
            auc_val = metrics['auc']

            fold_metrics[key].append({k: v for k, v in metrics.items()
                                      if k not in ('fpr', 'tpr')})
            fold_rocs[key].append((metrics['fpr'], metrics['tpr']))
            fold_auc_vals[key].append(auc_val)
            fold_pipes[key].append(final_pipe)
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
                'best_params': json.dumps(best_params),
            })

        # Progress line
        auc_str = '  '.join(
            f'{k} AUC={fold_auc_parts[k]:.3f}' for k in model_keys)
        print(f'[Fold {fold_idx+1}/{n_outer}]  {auc_str}')

    # -------------------------------------------------------------------------
    # Step 12: Aggregate metrics
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print(f'MODEL COMPARISON SUMMARY  (mean ± std over {n_outer} outer folds)')
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
    for key in model_keys:
        row = {'Model': MODEL_DISPLAY[key]}
        for mk in metric_keys:
            vals = [m[mk] for m in fold_metrics[key]]
            mean_col, std_col = col_map[mk]
            row[mean_col] = round(float(np.mean(vals)), 4)
            row[std_col]  = round(float(np.std(vals)),  4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Print summary
    print(summary_df[[
        'Model', 'AUC_mean', 'AUC_std',
        'Accuracy_mean', 'Sensitivity_mean', 'Specificity_mean',
        'F1_mean', 'Precision_mean',
    ]].to_string(index=False))

    best_row = summary_df.loc[summary_df['AUC_mean'].idxmax()]
    print(f'\n  Best Model: {best_row["Model"]}  '
          f'AUC={best_row["AUC_mean"]:.4f} ± {best_row["AUC_std"]:.4f}')

    # Save CSVs
    comp_path = os.path.join(out_dir, 'model_comparison_results.csv')
    summary_df.to_csv(comp_path, index=False)
    print(f'\n  Saved: {comp_path}')

    pf_path = os.path.join(out_dir, 'per_fold_results.csv')
    pd.DataFrame(per_fold_rows).to_csv(pf_path, index=False)
    print(f'  Saved: {pf_path}')

    # -------------------------------------------------------------------------
    # Step 13: Plots
    # -------------------------------------------------------------------------
    print('\n' + '=' * 60)
    print('Step 13 – Saving plots')
    print('=' * 60)

    plot_roc_mean(
        fold_rocs, fold_auc_vals, model_keys,
        os.path.join(out_dir, 'plot_roc_all_models.png'),
    )
    plot_model_comparison_bar(
        summary_df, model_keys,
        os.path.join(out_dir, 'plot_model_comparison_bar.png'),
    )
    plot_auc_distributions(
        fold_auc_vals, model_keys,
        os.path.join(out_dir, 'plot_auc_distributions.png'),
    )

    # RF importance: fold with highest RF AUC
    best_rf_fold = int(np.argmax(fold_auc_vals['RF']))
    best_rf_pipe = fold_pipes['RF'][best_rf_fold]
    plot_rf_importance(
        best_rf_pipe.named_steps['model'], selected_feats,
        os.path.join(out_dir, 'plot_rf_importance.png'),
    )

    print('\nAll outputs saved to:', out_dir)
    print('Done!')


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description='Traditional ML pipeline for ASD gait classification '
                    '— nested stratified cross-validation.')
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
        nargs='+',
        metavar='FEATURE',
        default=None,
        help=(
            'Skip Boruta and use this exact feature list instead.  '
            'Pass the names confirmed by the thesis to anchor the pipeline '
            'to the published results.  Example: '
            '--selected-features LHip_min RKnee_skew RAnkle_skew '
            'LAnkle_kurt LDP_cv TrunkY_max CoM_Y_min StepLength'
        ))
    parser.add_argument(
        '--n-trials',
        type=int, default=50,
        help='Optuna trials per model per outer fold (default: 50)')
    parser.add_argument(
        '--n-outer',
        type=int, default=5,
        help='Number of outer CV folds (default: 5)')
    parser.add_argument(
        '--n-inner',
        type=int, default=5,
        help='Number of inner CV folds for HPO (default: 5)')
    args = parser.parse_args()
    main(
        features_csv=args.features,
        out_dir=args.out,
        preselected_features=args.selected_features,
        n_trials=args.n_trials,
        n_outer=args.n_outer,
        n_inner=args.n_inner,
    )
