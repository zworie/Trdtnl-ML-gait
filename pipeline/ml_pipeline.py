"""
Traditional ML Pipeline for ASD Gait Classification
=====================================================
Python conversion of thesis.R, faithfully replicating the full pipeline:

  1.  Load gait_features_rich.csv
  2.  Remove near-zero-variance features  (caret::nearZeroVar)
  3.  Remove highly correlated features   (caret::findCorrelation, cutoff=0.95)
  4.  Group-difference statistical tests  (Shapiro-Wilk → t-test or Wilcoxon)
  5.  Boxplots of significant features
  6.  Boruta feature selection            (fallback: use significant features)
  7.  Stratified 70/30 train/test split
  8.  StandardScaler (fit on train only)
  9.  SMOTE on training set only         (over_ratio=1 → balanced classes)
  10. Repeated 5-fold CV (3 repeats)     (scoring=roc_auc)
  11. Model training + grid search:
        Logistic Regression (Elastic Net)
        Random Forest (1 000 trees)
        SVM (RBF kernel)
        XGBoost  [extension; not in published thesis results]
  12. Hold-out test evaluation + metrics
  13. ROC overlay, performance bar chart, RF importance, CV dotplot
  14. model_comparison_results.csv + statistical_test_results.csv

Faithfulness notes
------------------
- Boruta and statistical tests are run on the FULL dataset (same as R code),
  before the train/test split.  This is a data-leakage issue documented in
  FINDINGS.md but replicated here for reproducibility.
- SMOTE is applied exclusively to the training set, after scaling.
- All random seeds are fixed to 42 throughout.
- XGBoost is an extension present in thesis.R but absent from published
  thesis results.  It is included here for completeness.

Usage
-----
  python pipeline/ml_pipeline.py
  python pipeline/ml_pipeline.py --features outputs/gait_features_extracted.csv
"""

import argparse
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ── Optional imports (fail gracefully) ────────────────────────────────────────

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

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print('[WARN] imbalanced-learn not installed; SMOTE will be skipped.')


# ══════════════════════════════════════════════════════════════════════════════
# 1. caret-compatible helpers
# ══════════════════════════════════════════════════════════════════════════════

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
            # zero variance — always remove
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
                    break          # match R: stop checking j's for this i
                else:
                    deleted[j] = True
                    abs_cor[j, :] = 0
                    abs_cor[:, j] = 0
    return [cols[k] for k in range(n) if deleted[k]]


# ══════════════════════════════════════════════════════════════════════════════
# 2. Statistical tests
# ══════════════════════════════════════════════════════════════════════════════

def group_diff_tests(df, feat_cols, class_col='Class',
                     pos_class='ASD', neg_class='NonASD'):
    """
    Replicate R step 3: Shapiro-Wilk normality test → t-test or Wilcoxon.

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
            # Normal → Welch t-test (var.equal=FALSE)
            _, p = ttest_ind(pos[col].dropna(), neg[col].dropna(),
                             equal_var=False)
            test = 't-test'
        else:
            # Non-normal → Wilcoxon / Mann-Whitney (two-sided)
            _, p = mannwhitneyu(pos[col].dropna(), neg[col].dropna(),
                                alternative='two-sided')
            test = 'Wilcoxon'

        rows.append({'Feature': col, 'Test': test,
                     'p_value': round(float(p), 4),
                     'Significant': '***' if p < 0.05 else 'ns'})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Evaluation helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(estimator, X_test, y_test, name, pos_label=1):
    """
    Compute and print confusion matrix + metrics for one model.
    Returns a result dict compatible with the comparison table.
    """
    y_prob = estimator.predict_proba(X_test)[:, 1]
    y_pred = estimator.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    # labels=[0,1] → rows/cols = [NonASD, ASD]
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1          = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    roc_auc     = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    print(f'\n{"═"*46}')
    print(f'  {name}')
    print(f'{"═"*46}')
    print(f'  Confusion Matrix (rows=actual, cols=pred):')
    print(f'            NonASD   ASD')
    print(f'  NonASD  {tn:6d}  {fp:6d}')
    print(f'  ASD     {fn:6d}  {tp:6d}')
    print(f'  Accuracy   : {accuracy:.4f}')
    print(f'  Sensitivity: {sensitivity:.4f}')
    print(f'  Specificity: {specificity:.4f}')
    print(f'  Precision  : {precision:.4f}')
    print(f'  F1 Score   : {f1:.4f}')
    print(f'  AUC        : {roc_auc:.4f}')

    return {
        'name': name, 'estimator': estimator,
        'fpr': fpr, 'tpr': tpr,
        'accuracy': accuracy, 'sensitivity': sensitivity,
        'specificity': specificity, 'precision': precision,
        'f1': f1, 'auc': roc_auc,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {'ASD': '#D6604D', 'NonASD': '#4393C3'}
MODEL_COLORS = ['#1B7837', '#2166AC', '#D6604D', '#762A83', '#E08214']


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
            ax.boxplot(grp[feat].dropna(), positions=[list(df['Class'].unique()).index(cls)],
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


def plot_rf_importance(rf_estimator, feature_names, out_path):
    """Horizontal bar chart of RF feature importances (Gini)."""
    imp = rf_estimator.feature_importances_
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


def plot_roc_curves(results_list, out_path):
    """ROC overlay for all models (replicates R step 16)."""
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, res in enumerate(results_list):
        lbl = f"{res['name']}  (AUC={res['auc']:.3f})"
        ax.plot(res['fpr'], res['tpr'],
                color=MODEL_COLORS[i % len(MODEL_COLORS)],
                lw=2.5, label=lbl)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, color='grey')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves – All Models')
    ax.legend(loc='lower right', fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_model_comparison_bar(comparison_df, out_path):
    """Grouped bar chart of all metrics per model (replicates R step 17)."""
    metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC']
    melted = comparison_df[['Model'] + metrics].melt(
        id_vars='Model', var_name='Metric', value_name='Value')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=melted, x='Metric', y='Value', hue='Model',
                palette='Set1', alpha=0.9, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=7.5, padding=2)
    ax.set_ylim(0, 1.15)
    ax.set_xlabel('')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.tick_params(axis='x', rotation=15)
    ax.legend(title='Model', fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_cv_dotplot(cv_scores_dict, out_path):
    """
    Horizontal dotplot of CV AUC distributions (replicates caret's dotplot).
    Shows median + 95% CI across the 15 resampling folds.
    """
    names = list(cv_scores_dict.keys())
    medians = [np.median(cv_scores_dict[n]) for n in names]
    ci_lo = [np.percentile(cv_scores_dict[n], 2.5) for n in names]
    ci_hi = [np.percentile(cv_scores_dict[n], 97.5) for n in names]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.7)))
    y = np.arange(len(names))
    for i, n in enumerate(names):
        scores = cv_scores_dict[n]
        ax.scatter(scores, np.full_like(scores, i, dtype=float),
                   alpha=0.35, s=25, color=MODEL_COLORS[i % len(MODEL_COLORS)])
        ax.plot([ci_lo[i], ci_hi[i]], [i, i],
                color=MODEL_COLORS[i % len(MODEL_COLORS)], lw=2)
        ax.scatter([medians[i]], [i], s=60, zorder=5,
                   color=MODEL_COLORS[i % len(MODEL_COLORS)])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('CV AUC Distribution by Model\n(5-fold × 3 repeats, median + 95% CI)')
    ax.axvline(0.5, color='grey', linestyle='--', lw=1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def main(features_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = 42

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    print('=' * 60)
    print('Step 1 – Load data')
    print('=' * 60)
    df = pd.read_csv(features_csv)
    df = df.drop(columns=['Subject'], errors='ignore')
    df['Class'] = df['Class'].apply(lambda x: 'ASD' if x == 'ASD' else 'NonASD')
    feat_cols = [c for c in df.columns if c != 'Class']
    print(f'  Dimensions: {df.shape}')
    print(f'  Class distribution:\n{df["Class"].value_counts().to_string()}')

    # ── Step 2a: Near-zero-variance removal ───────────────────────────────────
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

    # ── Step 2b: High-correlation removal ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 2b – High-correlation removal (cutoff=0.95)')
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

    # ── Step 3: Group-difference statistical tests ────────────────────────────
    print('\n' + '=' * 60)
    print('Step 3 – Group-difference statistical tests')
    print('=' * 60)
    test_res = group_diff_tests(df, feat_cols)
    sig_feats = test_res.loc[test_res['Significant'] == '***', 'Feature'].tolist()
    print(f'  Significant features (p<0.05): {len(sig_feats)} / {len(feat_cols)}')
    print(test_res[test_res['Significant'] == '***'].to_string(index=False))
    test_res.to_csv(os.path.join(out_dir, 'statistical_test_results.csv'), index=False)
    print(f'\n  Saved: {os.path.join(out_dir, "statistical_test_results.csv")}')

    # ── Step 4: Boxplots of significant features ──────────────────────────────
    print('\n' + '=' * 60)
    print('Step 4 – Boxplots of significant features')
    print('=' * 60)
    if len(sig_feats) >= 2:
        plot_sig_boxplots(df, sig_feats,
                          os.path.join(out_dir, 'plot_sig_features_boxplot.png'))
    else:
        print('  Fewer than 2 significant features; skipping boxplot.')

    # ── Step 5: Boruta feature selection ──────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 5 – Boruta feature selection (max_iter=200)')
    print('=' * 60)
    X_all = df[feat_cols].values
    y_all = (df['Class'] == 'ASD').astype(int).values

    selected_feats = []
    if BORUTA_AVAILABLE:
        # R's Boruta uses default randomForest (no depth limit, no class weight).
        # Match those defaults here — max_depth=None, class_weight=None.
        rf_boruta = RandomForestClassifier(
            n_jobs=-1, random_state=rng)
        boruta = BorutaPy(rf_boruta, n_estimators='auto',
                          max_iter=200, random_state=rng, verbose=0)
        try:
            boruta.fit(X_all, y_all)
            selected_feats = [feat_cols[i]
                              for i, s in enumerate(boruta.support_) if s]
            # Include tentative features (TentativeRoughFix equivalent)
            tentative = [feat_cols[i]
                         for i, s in enumerate(boruta.support_weak_) if s]
            selected_feats = list(dict.fromkeys(selected_feats + tentative))
            print(f'  Boruta confirmed: {len(selected_feats)} features')
        except Exception as e:
            print(f'  [WARN] Boruta failed: {e}')
            selected_feats = []
    else:
        print('  Boruta not available.')

    # Fallback: use statistically significant features
    if not selected_feats:
        print('  Falling back to statistically significant features.')
        selected_feats = sig_feats if sig_feats else feat_cols

    print(f'  Selected ({len(selected_feats)}): {", ".join(selected_feats)}')

    df_sel = df[selected_feats + ['Class']].copy()

    # ── Step 6: Train/test split (stratified 70/30) ───────────────────────────
    print('\n' + '=' * 60)
    print('Step 6 – Train/test split (70/30, stratified)')
    print('=' * 60)
    X = df_sel[selected_feats].values
    y = (df_sel['Class'] == 'ASD').astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=rng)
    print(f'  Train: {len(y_train)}  |  Test: {len(y_test)}')
    print(f'  Train class dist: ASD={y_train.sum()}, NonASD={(y_train==0).sum()}')
    print(f'  Test  class dist: ASD={y_test.sum()}, NonASD={(y_test==0).sum()}')

    # ── Step 7: StandardScaler (fit on train only) ────────────────────────────
    print('\n' + '=' * 60)
    print('Step 7 – StandardScaler (fit on train, applied to both sets)')
    print('=' * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    print('  Scaler fitted on training data only.')

    # ── Step 8: SMOTE on training set only ────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 8 – SMOTE (training set only, over_ratio=1)')
    print('=' * 60)
    if SMOTE_AVAILABLE:
        smote = SMOTE(sampling_strategy=1.0, random_state=rng)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
        print(f'  After SMOTE – ASD: {y_train_bal.sum()}, '
              f'NonASD: {(y_train_bal==0).sum()}')
    else:
        print('  SMOTE not available; using unbalanced training set.')
        X_train_bal, y_train_bal = X_train_scaled, y_train

    # ── CV control ────────────────────────────────────────────────────────────
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=rng)

    # ── Steps 9–12: Train models ───────────────────────────────────────────────
    results_list = []
    best_estimators = {}

    # ── Logistic Regression (Elastic Net) ─────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 9 – Logistic Regression (Elastic Net)')
    print('=' * 60)
    # R glmnet: alpha ∈ {0,.25,.5,.75,1}, lambda = 10^seq(-4,1,len=40)
    # sklearn: l1_ratio = alpha, C = 1/lambda
    lambda_vals = np.logspace(-4, 1, 40)
    C_vals = sorted(set(np.round(1.0 / lambda_vals, 8)))
    lr_grid = {
        'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
        'C': C_vals,
    }
    lr_base = LogisticRegression(
        penalty='elasticnet', solver='saga',
        max_iter=2000, random_state=rng)
    lr_cv = GridSearchCV(lr_base, lr_grid, cv=rskf,
                         scoring='roc_auc', n_jobs=-1, refit=True)
    lr_cv.fit(X_train_bal, y_train_bal)
    best_lr = lr_cv.best_estimator_
    print(f'  Best l1_ratio={lr_cv.best_params_["l1_ratio"]}, '
          f'C={lr_cv.best_params_["C"]:.5f} '
          f'(≈ lambda={1/lr_cv.best_params_["C"]:.5f})')
    results_list.append(
        evaluate_model(best_lr, X_test_scaled, y_test,
                       'Logistic Regression (Elastic Net)'))
    best_estimators['LR_ElasticNet'] = best_lr

    # ── Random Forest ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 10 – Random Forest (1 000 trees)')
    print('=' * 60)
    n_feats = len(selected_feats)
    mtry_vals = sorted(set([2, 3, 4, round(np.sqrt(n_feats)),
                             round(n_feats / 2)]))
    rf_grid = {'max_features': mtry_vals}
    rf_base = RandomForestClassifier(
        n_estimators=1000, random_state=rng, n_jobs=-1)
    rf_cv = GridSearchCV(rf_base, rf_grid, cv=rskf,
                         scoring='roc_auc', n_jobs=-1, refit=True)
    rf_cv.fit(X_train_bal, y_train_bal)
    best_rf = rf_cv.best_estimator_
    print(f'  Best mtry (max_features): {rf_cv.best_params_["max_features"]}')
    results_list.append(
        evaluate_model(best_rf, X_test_scaled, y_test, 'Random Forest'))
    best_estimators['RandomForest'] = best_rf

    # RF importance plot
    plot_rf_importance(
        best_rf, selected_feats,
        os.path.join(out_dir, 'plot_rf_importance.png'))

    # ── SVM (RBF) ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 11 – SVM (RBF kernel)')
    print('=' * 60)
    svm_grid = {
        'C':     [0.01, 0.1, 0.5, 1, 5, 10, 50, 100],
        'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    }
    svm_base = SVC(kernel='rbf', probability=True, random_state=rng)
    svm_cv = GridSearchCV(svm_base, svm_grid, cv=rskf,
                          scoring='roc_auc', n_jobs=-1, refit=True)
    svm_cv.fit(X_train_bal, y_train_bal)
    best_svm = svm_cv.best_estimator_
    print(f'  Best C={svm_cv.best_params_["C"]}, '
          f'gamma={svm_cv.best_params_["gamma"]}')
    results_list.append(
        evaluate_model(best_svm, X_test_scaled, y_test, 'SVM (RBF Kernel)'))
    best_estimators['SVM_RBF'] = best_svm

    # ── XGBoost (extension) ───────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print('\n' + '=' * 60)
        print('Step 12 – XGBoost [extension; not in published thesis results]')
        print('=' * 60)
        xgb_grid = {
            'n_estimators':      [25, 50, 100],
            'max_depth':         [2, 3],
            'learning_rate':     [0.05, 0.1, 0.3],
            'gamma':             [0, 0.1],
            'colsample_bytree':  [0.8, 1.0],
            'min_child_weight':  [1, 3],
            'subsample':         [0.8, 1.0],
        }
        print(f'  Grid size: '
              f'{3*2*3*2*2*2*2} combinations')
        xgb_base = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=rng,
            n_jobs=-1)
        xgb_cv = GridSearchCV(xgb_base, xgb_grid, cv=rskf,
                               scoring='roc_auc', n_jobs=-1, refit=True)
        try:
            xgb_cv.fit(X_train_bal, y_train_bal)
            best_xgb = xgb_cv.best_estimator_
            print(f'  Best params: {xgb_cv.best_params_}')
            results_list.append(
                evaluate_model(best_xgb, X_test_scaled, y_test, 'XGBoost'))
            best_estimators['XGBoost'] = best_xgb
        except Exception as e:
            print(f'  [WARN] XGBoost failed: {e}')
    else:
        print('\n[INFO] XGBoost skipped — not installed.')

    # ── Step 13: Comparison table ──────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('MODEL COMPARISON SUMMARY')
    print('=' * 60)
    comparison = pd.DataFrame([{
        'Model':       r['name'],
        'Accuracy':    round(r['accuracy'],    4),
        'Sensitivity': round(r['sensitivity'], 4),
        'Specificity': round(r['specificity'], 4),
        'Precision':   round(r['precision'],   4),
        'F1':          round(r['f1'],          4),
        'AUC':         round(r['auc'],         4),
    } for r in results_list])
    print(comparison.to_string(index=False))
    best_row = comparison.loc[comparison['AUC'].idxmax()]
    print(f'\n★  Best Model: {best_row["Model"]}  |  '
          f'AUC={best_row["AUC"]:.4f}  '
          f'Acc={best_row["Accuracy"]:.4f}  '
          f'F1={best_row["F1"]:.4f}')
    comp_path = os.path.join(out_dir, 'model_comparison_results.csv')
    comparison.to_csv(comp_path, index=False)
    print(f'\n  Saved: {comp_path}')

    # ── Step 14: Plots ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('Step 14 – Saving plots')
    print('=' * 60)
    plot_roc_curves(results_list,
                    os.path.join(out_dir, 'plot_roc_all_models.png'))
    plot_model_comparison_bar(comparison,
                              os.path.join(out_dir, 'plot_model_comparison_bar.png'))

    # CV dotplot: re-run cross_val_score with best estimators on balanced train
    print('  Computing CV AUC distributions for dotplot...')
    cv_scores = {}
    for label, est in best_estimators.items():
        scores = cross_val_score(
            est, X_train_bal, y_train_bal,
            cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=rng),
            scoring='roc_auc', n_jobs=-1)
        cv_scores[label] = scores
    if cv_scores:
        plot_cv_dotplot(cv_scores,
                        os.path.join(out_dir, 'plot_cv_dotplot.png'))

    # Print CV summary (mirrors R's summary(resamps)$statistics$ROC)
    print('\n=== Cross-Validation ROC-AUC Summary ===')
    for label, scores in cv_scores.items():
        print(f'  {label:20s}  median={np.median(scores):.4f}  '
              f'[{np.percentile(scores,2.5):.4f} – '
              f'{np.percentile(scores,97.5):.4f}]  '
              f'(n={len(scores)} folds)')

    print('\nAll outputs saved to:', out_dir)
    print('Done!')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(
        description='Traditional ML pipeline for ASD gait classification.')
    parser.add_argument(
        '--features',
        default=os.path.join(root, 'gait_features_rich.csv'),
        help='Path to feature CSV (default: gait_features_rich.csv)')
    parser.add_argument(
        '--out',
        default=os.path.join(root, 'outputs'),
        help='Output directory (default: outputs/)')
    args = parser.parse_args()
    main(args.features, args.out)
