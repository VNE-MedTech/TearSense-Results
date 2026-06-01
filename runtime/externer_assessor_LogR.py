import argparse
import joblib
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, confusion_matrix,
    f1_score, roc_curve, RocCurveDisplay
)
from sklearn.calibration import CalibrationDisplay


# ══════════════════════════════════════════════════════════════════════════════
# FEATURES TO EXCLUDE FROM LR
# Add engineered feature names here to exclude them from the LR baseline.
# These are removed before training so LR only sees raw clinical features.
# ══════════════════════════════════════════════════════════════════════════════
EXCLUDE_FEATURES = [
    'Logit_Retear_Risk',
    'Tear_AP_ML_Ratio',
    'Tear_Area_cm2',
    'Log_Tear_Area',
    'ER_IR_Ratio',
    'ROM_Deficit_Score',
    'InsuranceType_NAN',
    'tear_characteristics_Full_NAN',

]


# ══════════════════════════════════════════════════════════════════════════════
# JSON ENCODER
# ══════════════════════════════════════════════════════════════════════════════

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def find_youden_threshold(y_true, y_probs):
    """Find threshold maximizing Youden's J statistic (Sensitivity + Specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


def calculate_net_benefit(y_true, y_prob, threshold):
    """Calculate net benefit at a specific threshold for DCA."""
    if threshold <= 0 or threshold >= 1:
        return 0.0
    tp = np.sum((y_prob >= threshold) & (y_true == 1))
    fp = np.sum((y_prob >= threshold) & (y_true == 0))
    n = len(y_true)
    weight = threshold / (1 - threshold)
    return float((tp / n) - (fp / n) * weight)


def calculate_net_benefit_curve(y_true, y_probs, thresholds):
    """Calculate net benefit curve for Decision Curve Analysis."""
    net_benefits = []
    for t in thresholds:
        if t >= 1.0:
            t = 0.999
        net_benefits.append(calculate_net_benefit(y_true, y_probs, t))
    return net_benefits


def get_calibration_slope_intercept(y_true, y_prob):
    """Calculate calibration slope and intercept via logistic regression on log-odds."""
    epsilon = 1e-10
    y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
    log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))

    clf = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
    clf.fit(log_odds.reshape(-1, 1), y_true)

    return float(clf.coef_[0][0]), float(clf.intercept_[0])


def compute_all_metrics(y_true, probs, threshold, model_name="Model"):
    """Compute comprehensive clinical metrics for a model."""
    auc = roc_auc_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)

    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, preds, zero_division=0)

    net_benefit = calculate_net_benefit(y_true, probs, threshold)
    cal_slope, cal_intercept = get_calibration_slope_intercept(y_true, probs)

    return {
        "model": model_name,
        "AUC": float(auc),
        "Brier_Score": float(brier),
        "F1_Score": float(f1),
        "Sensitivity_Recall": float(sens),
        "Specificity": float(spec),
        "PPV_Precision": float(ppv),
        "NPV": float(npv),
        "Net_Benefit": float(net_benefit),
        "Calibration_Slope": float(cal_slope),
        "Calibration_Intercept": float(cal_intercept),
        "Threshold_Used": float(threshold),
        "Confusion_Matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)}
    }


def bootstrap_metrics(y_true, y_probs, threshold, n_bootstraps=1000, seed=42):
    """Bootstrap 95% Confidence Intervals for all metrics."""
    rng = np.random.RandomState(seed)

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    indices = np.arange(len(y_true))

    boot_stats = {k: [] for k in [
        "AUC", "Brier_Score", "F1_Score", "Sensitivity_Recall",
        "Specificity", "PPV_Precision", "NPV", "Net_Benefit", "Calibration_Slope"
    ]}

    for _ in range(n_bootstraps):
        idx = rng.choice(indices, size=len(indices), replace=True)
        y_true_boot = y_true[idx]
        y_probs_boot = y_probs[idx]

        if len(np.unique(y_true_boot)) < 2:
            continue

        preds_boot = (y_probs_boot >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_boot, preds_boot).ravel()

        boot_stats["AUC"].append(roc_auc_score(y_true_boot, y_probs_boot))
        boot_stats["Brier_Score"].append(brier_score_loss(y_true_boot, y_probs_boot))
        boot_stats["F1_Score"].append(f1_score(y_true_boot, preds_boot, zero_division=0))

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        boot_stats["Sensitivity_Recall"].append(sens)
        boot_stats["Specificity"].append(spec)
        boot_stats["PPV_Precision"].append(ppv)
        boot_stats["NPV"].append(npv)

        boot_stats["Net_Benefit"].append(
            calculate_net_benefit(y_true_boot, y_probs_boot, threshold)
        )

        try:
            slope, _ = get_calibration_slope_intercept(y_true_boot, y_probs_boot)
            boot_stats["Calibration_Slope"].append(slope)
        except:
            pass

    ci_results = {}
    for key, values in boot_stats.items():
        if len(values) > 0:
            ci_results[key] = (np.percentile(values, 2.5), np.percentile(values, 97.5))
        else:
            ci_results[key] = (0.0, 0.0)

    return ci_results


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct_data_from_csv(feature_names, cat_cols, test_size=0.2, feature_engineering=False):
    """
    Reconstruct training data from training.csv using the same logic as core.py.
    Uses random_state=42 to get the exact same train/test split.
    """
    print("[INFO] Reconstructing data from training.csv...")

    if not os.path.exists("training.csv"):
        print("[ERROR] training.csv not found. Cannot reconstruct training data.")
        return None, None, None, None

    df = pd.read_csv("training.csv")
    df = df.replace("NaN", np.nan)

    # Filter to only the columns we need (feature_names + Re-Tear for target)
    all_needed_cols = list(feature_names) + ['Re-Tear'] if 'Re-Tear' not in feature_names else list(feature_names)
    existing_cols = [c for c in all_needed_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Create target
    if "Re-Tear" not in df.columns:
        print("[ERROR] 'Re-Tear' column not found in training.csv")
        return None, None, None, None

    y_raw = pd.to_numeric(df["Re-Tear"], errors="coerce")
    y_bin = (y_raw > 0).astype(int)
    valid_mask = ~y_raw.isna()
    df = df.loc[valid_mask].copy()
    y = y_bin.loc[valid_mask].copy()

    # Drop target column from features
    df = df.drop(columns=['Re-Tear'], errors='ignore')

    # Handle categorical columns
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).astype("category")
            if "Missing" not in df[col].cat.categories:
                df[col] = df[col].cat.add_categories("Missing")
            df[col] = df[col].fillna("Missing")

    # Handle numeric columns
    num_cols = [c for c in df.columns if c not in cat_cols]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')


    # Ensure we only have the features that match feature_names
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"[WARN] Missing features after reconstruction: {missing_features}")

    # Filter to only feature_names columns that exist
    available_features = [f for f in feature_names if f in df.columns]
    X = df[available_features].copy()

    # Split with the exact same parameters as core.py
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print(f"[INFO] Reconstructed - Train: {len(y_train)}, Test: {len(y_test)}")
    return X_train, y_train, X_test, y_test


def prepare_data_for_lr(X, cat_cols):
    """
    Prepare data for Logistic Regression:
    - One-hot encode categoricals (uppercased for consistency)
    - Handle missing values with median imputation
    Returns: (X_encoded, train_medians_dict)
        train_medians_dict is populated only when fitting; pass None for transform-only.
    """
    X_encoded = X.copy()

    # Handle categoricals: clean and uppercase
    for col in cat_cols:
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].fillna('Missing').astype(str).str.upper()

    # One-hot encode
    if cat_cols:
        existing_cats = [c for c in cat_cols if c in X_encoded.columns]
        if existing_cats:
            X_encoded = pd.get_dummies(X_encoded, columns=existing_cats,
                                       drop_first=True, dummy_na=False)

    # Fill numeric NaN with median and record medians
    train_medians = {}
    for col in X_encoded.columns:
        if X_encoded[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            med = X_encoded[col].median()
            train_medians[col] = med
            X_encoded[col] = X_encoded[col].fillna(med)

    return X_encoded, train_medians


def prepare_data_for_lr_transform(X, cat_cols, train_medians):
    """Prepare TEST data for LR using TRAIN medians (prevents data leakage)."""
    X_encoded = X.copy()

    for col in cat_cols:
        if col in X_encoded.columns:
            X_encoded[col] = X_encoded[col].fillna('Missing').astype(str).str.upper()

    if cat_cols:
        existing_cats = [c for c in cat_cols if c in X_encoded.columns]
        if existing_cats:
            X_encoded = pd.get_dummies(X_encoded, columns=existing_cats,
                                       drop_first=True, dummy_na=False)

    for col in X_encoded.columns:
        if X_encoded[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            med = train_medians.get(col, 0.0)
            X_encoded[col] = X_encoded[col].fillna(med)

    return X_encoded


def align_columns(X_train, X_test):
    """Ensure train and test have same columns after one-hot encoding."""
    all_cols = list(set(X_train.columns) | set(X_test.columns))

    for col in all_cols:
        if col not in X_train.columns:
            X_train[col] = 0
        if col not in X_test.columns:
            X_test[col] = 0

    sorted_cols = sorted(all_cols)
    return X_train[sorted_cols], X_test[sorted_cols]


# ══════════════════════════════════════════════════════════════════════════════
# FORMULA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_formula(lr_model, feature_names):
    """
    Generate the full human-readable logistic regression formula
    including ALL terms.

    P(retear) = 1 / (1 + exp(-z))
    where z = intercept + β1*x1 + β2*x2 + ...
    """
    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]

    # Create coefficient dataframe
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Abs_Coefficient': np.abs(coefs),
        'Odds_Ratio': np.exp(coefs)
    }).sort_values('Abs_Coefficient', ascending=False)

    # Start formula with intercept
    formula_parts = [f"{intercept:.4f}"]

    # Iterate through ALL rows in the dataframe (no slicing)
    for _, row in coef_df.iterrows():
        sign = "+" if row['Coefficient'] >= 0 else "-"
        formula_parts.append(f"{sign} {abs(row['Coefficient']):.4f} × {row['Feature']}")

    # Join all parts with newlines for readability
    formula_str = "z = " + "\n    ".join(formula_parts)
    formula_str += "\n\nP(retear) = 1 / (1 + exp(-z))"

    return formula_str, coef_df


# ══════════════════════════════════════════════════════════════════════════════
# CORE LR TRAINING — SINGLE SOURCE OF TRUTH
# ══════════════════════════════════════════════════════════════════════════════

def train_lr(X_train, y_train, X_test, cat_cols, feature_names, exclude_features=None):
    """
    Train a Logistic Regression model and return everything needed
    by combiner.py, external_assessor.py, and standalone analysis.

    This is the SINGLE place where LR hyperparameters are defined.
    All other modules import and call this function.

    Parameters
    ----------
    X_train : pd.DataFrame   — raw training features
    y_train : array-like      — binary training labels
    X_test  : pd.DataFrame    — raw test features
    cat_cols : list[str]      — categorical column names
    feature_names : list[str] — feature column names to use
    exclude_features : list[str] or None
        Features to exclude from LR. If None, uses module-level EXCLUDE_FEATURES.
        Pass an empty list [] to exclude nothing.

    Returns
    -------
    dict with keys:
        'probs_train'   : np.ndarray — predicted probabilities on training set
        'probs_test'    : np.ndarray — predicted probabilities on test set
        'model'         : LogisticRegression — fitted sklearn model
        'scaler'        : StandardScaler — fitted scaler
        'feature_cols'  : list[str] — final encoded feature column names
        'coef_df'       : pd.DataFrame — coefficients with odds ratios
        'formula'       : str — human-readable formula string
        'train_medians' : dict — median values used for imputation
    """
    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)

    # ── Exclude engineered / unwanted features ──
    if exclude_features is None:
        exclude_features = EXCLUDE_FEATURES
    if exclude_features:
        excluded = [f for f in exclude_features if f in feature_names]
        if excluded:
            print(f"[LR] Excluding {len(excluded)} features: {excluded}")
        feature_names = [f for f in feature_names if f not in exclude_features]

    # Subset to available features
    available = [f for f in feature_names if f in X_train.columns]
    if len(available) < len(feature_names):
        missing = set(feature_names) - set(available)
        print(f"[LR] Warning: {len(missing)} features missing: {list(missing)[:5]}...")

    X_train_subset = X_train[available].copy()
    X_test_subset = X_test[available].copy()

    # ── Drop columns that are entirely NaN ──
    nan_cols = X_train_subset.columns[X_train_subset.isna().all()].tolist()
    if nan_cols:
        print(f"[LR] Dropping {len(nan_cols)} all-NaN columns: {nan_cols}")
        X_train_subset = X_train_subset.drop(columns=nan_cols)
        X_test_subset = X_test_subset.drop(columns=nan_cols)

    # Encode training data (get medians for no-leak test imputation)
    X_train_enc, train_medians = prepare_data_for_lr(X_train_subset, cat_cols)
    X_test_enc = prepare_data_for_lr_transform(X_test_subset, cat_cols, train_medians)

    # Align columns (one-hot encoding may produce different columns)
    X_train_enc, X_test_enc = align_columns(X_train_enc, X_test_enc)

    print(f"[LR] Encoded features: {X_train_enc.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    # ═══════════════════════════════════════════════════════════════════════
    # LR HYPERPARAMETERS — SINGLE DEFINITION
    # Change these here; they propagate everywhere automatically.
    # ═══════════════════════════════════════════════════════════════════════
    lr_model = LogisticRegression(
        penalty='l2',
        C=1, # default is 1
        solver='lbfgs',
        max_iter=1000,
        class_weight=None
    )

    # lr_model = LogisticRegression()


    lr_model.fit(X_train_scaled, y_train_arr)

    probs_train = lr_model.predict_proba(X_train_scaled)[:, 1]
    probs_test = lr_model.predict_proba(X_test_scaled)[:, 1]

    print(f"[LR] Training AUC: {roc_auc_score(y_train_arr, probs_train):.4f}")

    # Generate formula and coefficients
    formula_str, coef_df = generate_formula(lr_model, list(X_train_enc.columns))

    return {
        'probs_train': probs_train,
        'probs_test': probs_test,
        'model': lr_model,
        'scaler': scaler,
        'feature_cols': list(X_train_enc.columns),
        'coef_df': coef_df,
        'formula': formula_str,
        'train_medians': train_medians,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: EXTRACT TRAIN/TEST DATA FROM BUNDLE
# ══════════════════════════════════════════════════════════════════════════════

def get_train_test_from_bundle(bundle):
    """
    Extract or reconstruct train/test data from a model bundle (.pkl).
    Returns (X_train, y_train, X_test, y_test, cat_cols, feature_names, feature_engineering)
    """
    X_test = bundle.get('X_test_exact')
    y_test = bundle.get('y_test_exact')
    X_train = bundle.get('X_train_all')
    y_train = bundle.get('y_train_all')
    cat_cols = bundle.get('cat_cols', [])
    feature_names = bundle.get('feature_names', [])
    feature_engineering = bundle.get('feature_engineering', True)

    if X_train is None or y_train is None:
        print("[LR] Training data not in PKL. Reconstructing from training.csv...")
        X_train, y_train, X_test_recon, y_test_recon = reconstruct_data_from_csv(
            feature_names, cat_cols,
            test_size=0.2,
            feature_engineering=False
        )
        if X_train is None:
            raise ValueError("Could not reconstruct training data. Ensure training.csv exists.")
        if X_test is None:
            X_test = X_test_recon
            y_test = y_test_recon

    if y_test is None:
        raise ValueError("No test data available (y_test_exact missing from bundle)")

    return X_train, y_train, X_test, y_test, cat_cols, feature_names, feature_engineering


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(model_path, tearsense_probs=None):
    print("=" * 60)
    print("LOGISTIC REGRESSION BASELINE COMPARISON")
    print("=" * 60)

    # ─────────────────────────────────────────
    # 1. LOAD PKL
    # ─────────────────────────────────────────
    print(f"\n[INFO] Loading model bundle: {model_path}")
    pkg = joblib.load(model_path)

    serial_number = pkg.get('serial_number', 'unknown')
    output_dir = os.path.join("baseline_comparison", serial_number)
    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────
    # 2. EXTRACT DATA
    # ─────────────────────────────────────────
    X_train, y_train, X_test, y_test, cat_cols, feature_names, _ = get_train_test_from_bundle(pkg)

    y_train_arr = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
    y_test_arr = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    print(f"[INFO] Training samples: {len(y_train_arr)}")
    print(f"[INFO] Test samples: {len(y_test_arr)}")
    print(f"[INFO] Features: {len(feature_names)}")
    print(f"[INFO] Categorical columns: {len(cat_cols)}")

    # ─────────────────────────────────────────
    # 3. TRAIN LR (single source of truth)
    # ─────────────────────────────────────────
    print("\n[INFO] Training Logistic Regression...")

    lr_result = train_lr(X_train, y_train_arr, X_test, cat_cols, feature_names)

    lr_probs_train = lr_result['probs_train']
    lr_probs_test = lr_result['probs_test']
    lr_model = lr_result['model']
    coef_df = lr_result['coef_df']
    formula_str = lr_result['formula']

    # ─────────────────────────────────────────
    # 4. COMPUTE THRESHOLD (Youden's J on LR test probs)
    # ─────────────────────────────────────────
    lr_threshold = find_youden_threshold(y_test_arr, lr_probs_test)
    print(f"[INFO] LR Youden's J Threshold: {lr_threshold:.4f}")

    # Also get TearSense threshold for comparison
    ts_threshold = pkg.get('optimal_threshold', 0.5)

    # ─────────────────────────────────────────
    # 5. COMPUTE METRICS
    # ─────────────────────────────────────────
    print("\n[INFO] Computing metrics...")

    lr_metrics = compute_all_metrics(y_test_arr, lr_probs_test, lr_threshold, "Logistic Regression")
    lr_metrics['n_features'] = int(len(lr_result['feature_cols']))
    lr_metrics['n_train'] = int(len(y_train_arr))
    lr_metrics['n_test'] = int(len(y_test_arr))

    # Save coefficients
    coef_path = os.path.join(output_dir, "lr_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"[EXPORT] Coefficients saved to: {coef_path}")

    # ─────────────────────────────────────────
    # 6. LOAD TEARSENSE METRICS FOR COMPARISON
    # ─────────────────────────────────────────
    ts_metrics_stored = pkg.get('metrics', {})
    ts_auc = ts_metrics_stored.get('test_auc', 0)
    ts_brier = ts_metrics_stored.get('test_brier', 0)

    # ─────────────────────────────────────────
    # 7. COMPARISON TABLE
    # ─────────────────────────────────────────
    comparison = {
        "serial_number": serial_number,
        "logistic_regression": lr_metrics,
        "tearsense": {
            "AUC": ts_auc,
            "Brier_Score": ts_brier,
            "Threshold_Used": ts_threshold,
            "Note": "Full TearSense metrics available in external_assessor output"
        },
        "comparison": {
            "AUC_difference": ts_auc - lr_metrics['AUC'] if ts_auc else None,
            "Brier_difference": lr_metrics['Brier_Score'] - ts_brier if ts_brier else None,
            "TearSense_better_AUC": ts_auc > lr_metrics['AUC'] if ts_auc else None
        }
    }

    # Save comparison
    comparison_path = os.path.join(output_dir, "comparison_table.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=4, cls=NumpyEncoder)

    # Save LR metrics
    lr_metrics_path = os.path.join(output_dir, "lr_metrics.json")
    with open(lr_metrics_path, 'w') as f:
        json.dump(lr_metrics, f, indent=4, cls=NumpyEncoder)
    print(f"[EXPORT] LR metrics saved to: {lr_metrics_path}")

    # ─────────────────────────────────────────
    # 8. PLOTS
    # ─────────────────────────────────────────
    print("\n[INFO] Generating plots...")

    # ROC Curve
    plt.figure(figsize=(8, 8))
    RocCurveDisplay.from_predictions(
        y_test_arr, lr_probs_test,
        name=f"Logistic Regression (AUC={lr_metrics['AUC']:.3f})",
        color="blue"
    )
    if ts_auc:
        plt.title(f"ROC Comparison\nLR AUC: {lr_metrics['AUC']:.3f} | TearSense AUC: {ts_auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Calibration Curve
    plt.figure(figsize=(8, 8))
    CalibrationDisplay.from_predictions(
        y_test_arr, lr_probs_test,
        n_bins=10, name="Logistic Regression"
    )
    plt.title(f"Calibration Curve (Slope: {lr_metrics['Calibration_Slope']:.2f}, Brier: {lr_metrics['Brier_Score']:.3f})")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_calibration.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # DCA
    plt.figure(figsize=(10, 6))
    thresholds_dca = np.linspace(0.01, 0.99, 100)
    nb_lr = calculate_net_benefit_curve(y_test_arr, lr_probs_test, thresholds_dca)
    prevalence = np.mean(y_test_arr)
    nb_all = prevalence - (1 - prevalence) * (thresholds_dca / (1 - thresholds_dca))
    nb_none = np.zeros_like(thresholds_dca)

    plt.plot(thresholds_dca, nb_lr, color='blue', lw=2, label='Logistic Regression')
    plt.plot(thresholds_dca, nb_all, color='gray', linestyle='--', label='Treat All')
    plt.plot(thresholds_dca, nb_none, color='black', linestyle=':', label='Treat None')

    # Mark optimal threshold
    nb_at_opt = lr_metrics['Net_Benefit']
    plt.scatter(lr_threshold, nb_at_opt, color='blue', s=100, zorder=5,
                label=f"LR Threshold ({lr_threshold:.2f})")

    plt.ylim(-0.05, max(prevalence, max(nb_lr)) + 0.05)
    plt.xlim(0, 1.0)
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis - Logistic Regression")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "lr_dca.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # ─────────────────────────────────────────
    # 9. PRINT REPORT
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    print(f"\nModel: Logistic Regression (L2 regularization)")
    print(f"Training N: {lr_metrics['n_train']}")
    print(f"Test N: {lr_metrics['n_test']}")
    print(f"Features: {lr_metrics['n_features']}")
    print(f"Threshold (Youden's J): {lr_threshold:.4f}")
    print(f"\n{'-'*60}")
    print(f"{'Metric':<25} | {'Value':>10}")
    print(f"{'-'*60}")
    print(f"{'AUC':<25} | {lr_metrics['AUC']:>10.4f}")
    print(f"{'Brier Score':<25} | {lr_metrics['Brier_Score']:>10.4f}")
    print(f"{'Calibration Slope':<25} | {lr_metrics['Calibration_Slope']:>10.4f}")
    print(f"{'Net Benefit':<25} | {lr_metrics['Net_Benefit']:>10.4f}")
    print(f"{'Sensitivity':<25} | {lr_metrics['Sensitivity_Recall']:>10.4f}")
    print(f"{'Specificity':<25} | {lr_metrics['Specificity']:>10.4f}")
    print(f"{'PPV':<25} | {lr_metrics['PPV_Precision']:>10.4f}")
    print(f"{'NPV':<25} | {lr_metrics['NPV']:>10.4f}")
    print(f"{'F1 Score':<25} | {lr_metrics['F1_Score']:>10.4f}")
    print(f"{'-'*60}")

    cm = lr_metrics['Confusion_Matrix']
    print(f"\nConfusion Matrix: TP={cm['TP']}, TN={cm['TN']}, FP={cm['FP']}, FN={cm['FN']}")

    # Print comparison if available
    if ts_auc:
        print(f"\n{'='*60}")
        print("COMPARISON: Logistic Regression vs TearSense")
        print(f"{'='*60}")
        print(f"{'Metric':<25} | {'LR':>10} | {'TearSense':>10} | {'Δ':>10}")
        print(f"{'-'*60}")
        print(f"{'AUC':<25} | {lr_metrics['AUC']:>10.4f} | {ts_auc:>10.4f} | {ts_auc - lr_metrics['AUC']:>+10.4f}")
        if ts_brier:
            print(f"{'Brier':<25} | {lr_metrics['Brier_Score']:>10.4f} | {ts_brier:>10.4f} | {ts_brier - lr_metrics['Brier_Score']:>+10.4f}")
        print(f"{'-'*60}")

        if ts_auc > lr_metrics['AUC']:
            print(f"\n✓ TearSense outperforms LR by {(ts_auc - lr_metrics['AUC'])*100:.2f}% AUC")
        else:
            print(f"\n✗ LR outperforms TearSense by {(lr_metrics['AUC'] - ts_auc)*100:.2f}% AUC")

    # Print formula
    print(f"\n{'='*60}")
    print("LOGISTIC REGRESSION FORMULA")
    print(f"{'='*60}")
    print(formula_str)

    print(f"\n{'='*60}")
    print(f"[EXPORT] All outputs saved to: {output_dir}/")
    print("[DONE]")

    return lr_metrics, comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logistic Regression Baseline Comparison")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pkl model file")
    args = parser.parse_args()
    main(args.model_path)
