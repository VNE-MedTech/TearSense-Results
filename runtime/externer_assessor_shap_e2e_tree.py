"""
TearSense TreeSHAP E2E — Two-Stage Chained SHAP Through Meta-Learner
═══════════════════════════════════════════════════════════════════════════

True end-to-end SHAP using TreeSHAP (fast & exact), evaluated through the
actual meta-stacking pipeline:

    Stage 1: SHAP on meta-learner → per-sample importance of each base model
    Stage 2: TreeSHAP on each base model → per-feature attribution
    Chain:   feature_shap[j] = Σ_m (meta_weight_m / base_sum_m) × base_shap_m[j]

This preserves SHAP additivity: Σ_j feature_shap[j] = f(x) - E[f(x)]

For weighted-average models (no meta-learner), falls back to static-weight
combination (equivalent to ensemble SHAP).

Logit_Retear_Risk SHAP is redistributed to its constituent clinical
features using the logistic regression coefficients.

Outputs:
    shap_summary_{serial}.png
    shap_bar_{serial}.png
    feature_importance_{serial}.csv
    shap_values_{serial}.csv
    shap_metadata_{serial}.json
"""

import os
import json
import time
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from runtime.externer_assessor_shap import (
    NumpyEncoder,
    generate_meta_features,
    plot_shap_summary,
    plot_shap_bar,
    export_feature_importance,
)
from runtime.externer_assessor_LogR import (
    prepare_data_for_lr,
    prepare_data_for_lr_transform,
    align_columns,
)
from shap_config import rename_features


# ══════════════════════════════════════════════════════════════════════════════
# META-FEATURE → BASE MODEL MAPPING
# ══════════════════════════════════════════════════════════════════════════════

MODELS = ['cat', 'xgb', 'lgbm', 'rf']

# Pairwise interaction ordering (matches generate_meta_features)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


def map_meta_shap_to_base_weights(meta_shap_vals, clean_preds, use_interactions):
    """
    Map meta-feature SHAP values back to per-sample base model weights.

    Meta-feature layout (with interactions):
        [0-3]   Direct base predictions: cat, xgb, lgbm, rf
        [4-9]   Pairwise products: cat*xgb, cat*lgbm, cat*rf, xgb*lgbm, xgb*rf, lgbm*rf
        [10]    Mean of all predictions
        [11]    Std of all predictions
        [12]    Range (max - min) of all predictions

    Returns: dict {model_name: (n_samples,) weight array}
    """
    n_samples = meta_shap_vals.shape[0]
    weights = {m: np.zeros(n_samples) for m in MODELS}

    # Direct predictions (cols 0-3) → 100% to corresponding model
    for i, m in enumerate(MODELS):
        weights[m] += meta_shap_vals[:, i]

    if not use_interactions:
        return weights

    # Pairwise interactions (cols 4-9) → 50/50 split between pair
    for idx, (i, j) in enumerate(PAIRS):
        col = 4 + idx
        weights[MODELS[i]] += meta_shap_vals[:, col] * 0.5
        weights[MODELS[j]] += meta_shap_vals[:, col] * 0.5

    # Mean (col 10) → equal split across all 4 models
    for m in MODELS:
        weights[m] += meta_shap_vals[:, 10] * 0.25

    # Std (col 11) → proportional to |pred_m - mean|
    all_preds = np.array([clean_preds[m] for m in MODELS])  # (4, n_samples)
    mean_pred = np.mean(all_preds, axis=0)
    deviations = np.abs(all_preds - mean_pred[None, :])     # (4, n_samples)
    dev_sum = deviations.sum(axis=0, keepdims=True)
    dev_sum[dev_sum == 0] = 1e-9
    dev_proportions = deviations / dev_sum                   # (4, n_samples)
    for i, m in enumerate(MODELS):
        weights[m] += meta_shap_vals[:, 11] * dev_proportions[i]

    # Range (col 12) → 50% to max model, 50% to min model
    max_idx = np.argmax(all_preds, axis=0)  # (n_samples,)
    min_idx = np.argmin(all_preds, axis=0)
    for i, m in enumerate(MODELS):
        is_max = (max_idx == i).astype(float)
        is_min = (min_idx == i).astype(float)
        weights[m] += meta_shap_vals[:, 12] * 0.5 * is_max
        weights[m] += meta_shap_vals[:, 12] * 0.5 * is_min

    return weights


def chain_shap_values(meta_weights, base_shap_dict):
    """
    Chain meta-learner weights with base model TreeSHAP values.

    For each sample:
        feature_shap[j] = Σ_m (meta_weight_m / base_sum_m) × base_shap_m[j]

    Where base_sum_m = Σ_j base_shap_m[j] = pred_m - E[pred_m].
    This preserves SHAP additivity:
        Σ_j feature_shap[j] = Σ_m meta_weight_m = f(x) - E[f(x)]
    """
    n_samples, n_features = base_shap_dict['cat'].shape
    result = np.zeros((n_samples, n_features))

    for m in MODELS:
        base_shap = base_shap_dict[m]                           # (n, F)
        base_sum = base_shap.sum(axis=1, keepdims=True)         # (n, 1)
        w = meta_weights[m][:, None]                            # (n, 1)

        # Scale factor: meta_weight / base_deviation
        # When base model didn't deviate (sum ≈ 0), meta-learner weight
        # should also be ≈ 0 (model contributed nothing). Safe to zero out.
        scale = np.where(np.abs(base_sum) > 1e-8, w / base_sum, 0.0)
        result += base_shap * scale

    return result


# ══════════════════════════════════════════════════════════════════════════════
# META-LEARNER SHAP
# ══════════════════════════════════════════════════════════════════════════════

def compute_meta_learner_shap(meta_model, X_meta):
    """
    Compute SHAP values for the meta-learner.
    Tries TreeExplainer (XGB/LGBM), then LinearExplainer (LR Pipeline),
    then falls back to KernelExplainer (any model, only 13 features → fast).
    """
    # Try TreeExplainer (works for XGBoost, LightGBM meta-learners)
    try:
        explainer = shap.TreeExplainer(meta_model)
        sv = explainer.shap_values(X_meta)
        if isinstance(sv, list):
            sv = sv[1]  # class 1 for binary
        print("[TreeSHAP-E2E] Meta-learner: used TreeExplainer")
        return sv
    except Exception:
        pass

    # Try LinearExplainer (works for sklearn Pipeline with StandardScaler + LR)
    try:
        explainer = shap.LinearExplainer(meta_model, X_meta)
        sv = explainer.shap_values(X_meta)
        if isinstance(sv, list):
            sv = sv[1]
        print("[TreeSHAP-E2E] Meta-learner: used LinearExplainer")
        return sv
    except Exception:
        pass

    # Fallback: KernelExplainer (only 13 meta-features, so this is fast)
    print("[TreeSHAP-E2E] Meta-learner: using KernelExplainer (13 meta-features)...")
    background = shap.kmeans(X_meta, min(50, len(X_meta)))

    def meta_predict(X):
        proba = meta_model.predict_proba(X)
        return proba[:, 1] if proba.ndim > 1 else proba

    explainer = shap.KernelExplainer(meta_predict, background)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = explainer.shap_values(X_meta, silent=True)
    print("[TreeSHAP-E2E] Meta-learner: used KernelExplainer")
    return sv


# ══════════════════════════════════════════════════════════════════════════════
# LOGIT P REDISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def redistribute_logit_shap(ensemble_shap, all_feat, bundle, serial):
    """
    Redistribute Logit_Retear_Risk SHAP values to the underlying
    clinical features using the logistic regression coefficients.

    Returns (modified_shap, indices_to_keep).
    """
    if 'Logit_Retear_Risk' not in all_feat:
        print("[TreeSHAP-E2E] Logit_Retear_Risk not in features, nothing to redistribute.")
        return ensemble_shap, list(range(len(all_feat)))

    logit_idx = all_feat.index('Logit_Retear_Risk')

    coef_path = os.path.join("outputs", serial, "stats", "lr_coefficients.csv")
    if not os.path.exists(coef_path):
        coef_path = os.path.join("external_assessor", serial,
                                 "Logistic_Regression_Baseline", "lr_coefficients.csv")

    if not os.path.exists(coef_path):
        print(f"[TreeSHAP-E2E] LR coefficients not found. Excluding Logit_Retear_Risk.")
        keep = [i for i in range(len(all_feat)) if i != logit_idx]
        return ensemble_shap, keep

    print(f"[TreeSHAP-E2E] Redistributing Logit_Retear_Risk using {coef_path}")
    coefs = pd.read_csv(coef_path)
    w_dict = dict(zip(coefs['Feature'], coefs['Coefficient']))

    cat_cols = list(bundle.get('cat_cols', []))

    X_test_raw = bundle.get('X_test_exact')
    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=all_feat)
    else:
        X_test_raw = pd.DataFrame(X_test_raw.values, columns=all_feat)

    X_train_raw = bundle.get('X_train_all')
    if X_train_raw is not None:
        if not isinstance(X_train_raw, pd.DataFrame):
            X_train_raw = pd.DataFrame(X_train_raw, columns=all_feat)
        else:
            X_train_raw = pd.DataFrame(X_train_raw.values, columns=all_feat)
    else:
        X_train_raw = X_test_raw.copy()

    # Rebuild the LR design matrix EXACTLY as the trainer's LR did
    # (fillna 'Missing' + uppercase + get_dummies(drop_first=True, dummy_na=False)),
    # so the one-hot column names match the coefficient names in lr_coefficients.csv.
    # A bare get_dummies(dummy_na=True) mismatches names (e.g. 'InsuranceType_nan' vs
    # the trainer's 'InsuranceType_NAN') and — when missing values are stored as the
    # literal string 'nan' — emits a duplicate '_nan' column that breaks reindex.
    cat_in = [c for c in cat_cols if c in X_train_raw.columns]
    X_train_enc, train_medians = prepare_data_for_lr(X_train_raw, cat_in)
    X_test_enc = prepare_data_for_lr_transform(X_test_raw, cat_in, train_medians)
    X_train_enc, X_test_enc = align_columns(X_train_enc, X_test_enc)

    lr_cols = [c for c in w_dict.keys() if c in X_train_enc.columns]
    if not lr_cols:
        print("[TreeSHAP-E2E] No LR columns matched. Excluding Logit_Retear_Risk.")
        keep = [i for i in range(len(all_feat)) if i != logit_idx]
        return ensemble_shap, keep

    X_train_lr = X_train_enc[lr_cols].fillna(0).astype(float)
    X_test_lr = X_test_enc[lr_cols].fillna(0).astype(float)

    mean_vals = X_train_lr.mean()
    std_vals = X_train_lr.std()
    std_vals[std_vals == 0] = 1.0
    X_test_scaled = (X_test_lr - mean_vals) / std_vals

    lr_weights = np.array([w_dict[c] for c in lr_cols])
    shap_lr_expanded = X_test_scaled * lr_weights

    shap_lr_orig = np.zeros((len(X_test_raw), len(all_feat)))
    for i, col in enumerate(lr_cols):
        orig_col = col
        if col not in all_feat:
            for cat in cat_cols:
                if col.startswith(cat + '_'):
                    orig_col = cat
                    break
        if orig_col in all_feat:
            orig_idx = all_feat.index(orig_col)
            shap_lr_orig[:, orig_idx] += shap_lr_expanded[col].values

    logit_shap_vals = ensemble_shap[:, logit_idx].copy()
    abs_lr_sum = np.abs(shap_lr_orig).sum(axis=1, keepdims=True)
    abs_lr_sum[abs_lr_sum == 0] = 1e-9

    distribution_matrix = (shap_lr_orig / abs_lr_sum) * logit_shap_vals[:, None]

    modified_shap = ensemble_shap.copy()
    modified_shap += distribution_matrix
    modified_shap[:, logit_idx] = 0.0

    n_receiving = np.sum(np.abs(distribution_matrix).sum(axis=0) > 0)
    print(f"[TreeSHAP-E2E] Redistributed Logit_Retear_Risk SHAP to {n_receiving} features.")

    keep = [i for i in range(len(all_feat)) if i != logit_idx]
    return modified_shap, keep


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(model_path, output_dir=None):
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — TreeSHAP E2E (Two-Stage Chained Through Meta-Learner)")
    print("=" * 70)

    bundle = joblib.load(model_path)
    serial = bundle.get('serial_number', 'unknown')

    if output_dir is None:
        output_dir = os.path.join("external_assessor", serial, "shap_e2e_tree")
    os.makedirs(output_dir, exist_ok=True)

    all_feat = list(bundle.get('feature_names', []))
    cat_cols = list(bundle.get('cat_cols', []))
    cat_idx_full = list(bundle.get('cat_indices', []))
    fold_models = bundle.get('fold_models', {})
    n_folds = bundle.get('n_folds', 5)
    is_wa = bundle.get('is_weighted_avg', False)
    meta_model = bundle.get('meta_model')
    sc = bundle.get('stacking_config', {})
    use_inter = sc.get('use_interactions', True)
    use_rank = sc.get('use_rank_features', False)
    best_method = bundle.get('best_method', 'unknown')

    print(f"[TreeSHAP-E2E] Serial:          {serial}")
    print(f"[TreeSHAP-E2E] Best method:     {best_method}")
    print(f"[TreeSHAP-E2E] is_weighted_avg: {is_wa}")
    print(f"[TreeSHAP-E2E] Interactions:    {use_inter}")

    # ── Data preparation ──
    X_test_raw = bundle.get('X_test_exact')
    if not isinstance(X_test_raw, pd.DataFrame):
        X_test_raw = pd.DataFrame(X_test_raw, columns=all_feat)
    else:
        X_test_raw = pd.DataFrame(X_test_raw.values, columns=all_feat)

    X_full = X_test_raw.copy()
    for col in cat_cols:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype(str).replace('nan', 'Missing').fillna('Missing')

    X_enc = X_full.copy()
    for col in cat_cols:
        if col in X_enc.columns:
            X_enc[col] = X_enc[col].astype('category').cat.codes
    for col in X_enc.columns:
        X_enc[col] = pd.to_numeric(X_enc[col], errors='coerce')

    X_rf = X_enc.copy().fillna(-999)
    X_xgb_lgbm = X_enc.copy()

    n_samples = len(X_test_raw)
    n_features = len(all_feat)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Base model TreeSHAP + Predictions
    # ══════════════════════════════════════════════════════════════════════
    base_shap_dict = {}
    base_preds = {}
    t0 = time.time()

    for algo in MODELS:
        models = fold_models.get(algo, [])
        if not models:
            print(f"   [WARN] No models found for {algo}.")
            base_shap_dict[algo] = np.zeros((n_samples, n_features))
            base_preds[algo] = np.full(n_samples, 0.5)
            continue

        print(f"[TreeSHAP-E2E] Computing TreeSHAP + predictions for {algo}...")
        fold_shaps = []
        fold_preds = []

        for model in models:
            if algo == 'cat':
                from catboost import Pool
                pool = Pool(X_full, cat_features=cat_idx_full)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(pool)
                fold_shaps.append(sv)
                preds = model.predict_proba(pool)[:, 1]
                fold_preds.append(preds)
            elif algo == 'xgb':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_xgb_lgbm)
                fold_shaps.append(sv)
                preds = model.predict_proba(X_xgb_lgbm)[:, 1]
                fold_preds.append(preds)
            elif algo == 'lgbm':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_xgb_lgbm)
                if isinstance(sv, list):
                    sv = sv[1]
                fold_shaps.append(sv)
                preds = model.predict_proba(X_xgb_lgbm)[:, 1]
                fold_preds.append(preds)
            elif algo == 'rf':
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_rf, check_additivity=False)
                if isinstance(sv, list):
                    sv = sv[1]
                elif len(sv.shape) == 3:
                    sv = sv[:, :, 1]
                fold_shaps.append(sv)
                preds = model.predict_proba(X_rf)[:, 1]
                fold_preds.append(preds)

        base_shap_dict[algo] = np.mean(fold_shaps, axis=0)
        base_preds[algo] = np.mean(fold_preds, axis=0)

    t_base = time.time() - t0
    print(f"[TreeSHAP-E2E] Base model SHAP done in {t_base:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Meta-Learner SHAP → Chain
    # ══════════════════════════════════════════════════════════════════════

    if is_wa:
        # Weighted average: use static weights (no meta-learner)
        weights = bundle.get('weights', {})
        weights_arr = bundle.get('weights_array')

        if isinstance(weights, dict) and weights:
            w = weights
        elif weights_arr is not None:
            w = {m: float(weights_arr[i]) for i, m in enumerate(MODELS)}
        else:
            # Fallback: read from hyperparameters JSON
            config_path = os.path.join("outputs", serial, "configs",
                                       "best_hyperparameters.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    hp = json.load(f)
                ew = hp.get("Ensemble_Weights", {})
                w = {
                    'cat': ew.get('Cat', 0.25),
                    'xgb': ew.get('XGB', 0.25),
                    'lgbm': ew.get('LGBM', 0.25),
                    'rf': ew.get('RF', 0.25),
                }
            else:
                w = {m: 0.25 for m in MODELS}

        print(f"[TreeSHAP-E2E] Weighted average mode. Weights: {w}")
        ensemble_shap = sum(w[m] * base_shap_dict[m] for m in MODELS)

    else:
        # Actual meta-learner: two-stage chained SHAP
        print("[TreeSHAP-E2E] Meta-learner detected. Running two-stage chain...")

        # Clean base predictions (same as generate_meta_features does)
        clean_preds = {}
        for m in MODELS:
            c = np.nan_to_num(base_preds[m], nan=0.5)
            clean_preds[m] = np.clip(c, 1e-7, 1 - 1e-7)

        # Generate meta-features
        X_meta = generate_meta_features(clean_preds, use_inter, use_rank)
        print(f"[TreeSHAP-E2E] Meta-features shape: {X_meta.shape}")

        # Compute meta-learner SHAP
        t_meta = time.time()
        meta_shap_vals = compute_meta_learner_shap(meta_model, X_meta)
        print(f"[TreeSHAP-E2E] Meta-learner SHAP done in {time.time() - t_meta:.1f}s")

        # Verify: meta-SHAP sum should ≈ meta-learner output - E[output]
        meta_shap_sum = meta_shap_vals.sum(axis=1)
        print(f"[TreeSHAP-E2E] Meta-SHAP sum range: [{meta_shap_sum.min():.4f}, {meta_shap_sum.max():.4f}]")

        # Map meta-feature SHAP to per-sample base model weights
        meta_weights = map_meta_shap_to_base_weights(
            meta_shap_vals, clean_preds, use_inter,
        )

        # Print per-model weight statistics
        for m in MODELS:
            w_m = meta_weights[m]
            print(f"   {m:>4}: mean_weight={w_m.mean():.4f}, "
                  f"std={w_m.std():.4f}, range=[{w_m.min():.4f}, {w_m.max():.4f}]")

        # Chain: distribute meta-weights through base model SHAP
        ensemble_shap = chain_shap_values(meta_weights, base_shap_dict)

    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Redistribute Logit_Retear_Risk
    # ══════════════════════════════════════════════════════════════════════
    ensemble_shap, indices_to_keep = redistribute_logit_shap(
        ensemble_shap, all_feat, bundle, serial,
    )

    elapsed = time.time() - t0
    print(f"[TreeSHAP-E2E] Total elapsed: {elapsed:.1f}s")

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT: Plots & Data
    # ══════════════════════════════════════════════════════════════════════
    print("[TreeSHAP-E2E] Generating plots...")

    X_display = X_test_raw.copy()
    for col in cat_cols:
        if col in X_display.columns:
            X_display[col] = X_display[col].astype(str).astype('category').cat.codes
    for col in X_display.columns:
        X_display[col] = pd.to_numeric(X_display[col], errors='coerce').fillna(0).astype(float)

    filtered_feat = [all_feat[i] for i in indices_to_keep]
    ensemble_shap_filtered = ensemble_shap[:, indices_to_keep]
    X_display_filtered = X_display.iloc[:, indices_to_keep]

    # Rename to doctor-friendly labels
    filtered_feat = rename_features(filtered_feat)

    plot_shap_summary(
        ensemble_shap_filtered, X_display_filtered, filtered_feat,
        os.path.join(output_dir, f"shap_summary_{serial}.png"), serial,
    )
    plot_shap_bar(
        ensemble_shap_filtered, filtered_feat,
        os.path.join(output_dir, f"shap_bar_{serial}.png"), serial,
    )

    importance_df = export_feature_importance(
        ensemble_shap_filtered, filtered_feat,
        os.path.join(output_dir, f"feature_importance_{serial}.csv"),
        X_display=X_display_filtered,
    )

    sv_df = pd.DataFrame(ensemble_shap_filtered, columns=filtered_feat)
    sv_path = os.path.join(output_dir, f"shap_values_{serial}.csv")
    sv_df.to_csv(sv_path, index=False)
    print(f"[TreeSHAP-E2E] Raw SHAP values -> {sv_path}")

    # Metadata
    metadata = {
        "serial_number": serial,
        "method": "Two-Stage Chained TreeSHAP (E2E through meta-learner)",
        "best_method": best_method,
        "is_weighted_avg": is_wa,
        "use_interactions": use_inter,
        "n_test_samples": n_samples,
        "elapsed_seconds": round(elapsed, 1),
        "top_features": importance_df.head(15)[
            ['Rank', 'Feature', 'Mean_Abs_SHAP']
        ].to_dict('records'),
    }
    meta_path = os.path.join(output_dir, f"shap_metadata_{serial}.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4, cls=NumpyEncoder)
    print(f"[TreeSHAP-E2E] Metadata -> {meta_path}")

    print("\n" + "=" * 70)
    print(f"  TreeSHAP E2E COMPLETE — {serial}")
    print(f"  Method: {'Weighted Avg' if is_wa else 'Two-Stage Chain (' + best_method + ')'}")
    print("=" * 70)

    return ensemble_shap_filtered, filtered_feat, importance_df


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        mp = sys.argv[1]
        od = sys.argv[2] if len(sys.argv) >= 3 else None
        main(mp, output_dir=od)
    else:
        print("Usage: python -m runtime.externer_assessor_shap_e2e_tree <model.pkl> [output_dir]")
