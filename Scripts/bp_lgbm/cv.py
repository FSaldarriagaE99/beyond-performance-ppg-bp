######## CONFIG SCRIPT ###############################################################
#                                                                                    #
# This script handles the model and the experiment config scripts                    #
# of the different tree based models that I am using.                                #
#                                                                                    #
######################################################################################

# cv.py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

# ----------------------------
# CV Computing Functions
# ----------------------------
def patient_wise_cv(pipeline, X, y, groups, metric_fn, n_splits=5):
    """
    Run patient-wise cross-validation with GroupKFold.

    Args:
        pipeline: sklearn Pipeline (scaler + model)
        X (pd.DataFrame or np.ndarray): Features
        y (pd.Series or np.ndarray): Target values
        groups (array-like): Patient IDs for grouping
        metric_fn: function(y_true, y_pred) -> dict or float
        n_splits: number of folds

    Returns:
        results: list of per-fold dicts (fold info + metrics)
        summary: dict of mean and std for each metric across folds
    """
    cv = GroupKFold(n_splits=n_splits)
    results = []
    all_metrics  = {}
    total_signals = len(y)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        train_patients = np.unique(groups[train_idx])
        val_patients = np.unique(groups[val_idx])

        # Fit pipeline (scaler + model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        # Compute metrics
        metrics = metric_fn(y_val, y_pred)
        if isinstance(metrics, float):
            metrics = {"score": metrics}

        # Save metrics for raw distribution
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append((v, len(val_idx)))

        # Store fold result
        fold_result = {
            "fold": fold,
            "n_train_signals": len(train_idx),
            "n_val_signals": len(val_idx),
            "n_train_patients": len(train_patients),
            "n_val_patients": len(val_patients),
            "train_patients": train_patients.tolist(),
            "val_patients": val_patients.tolist(),
            "metrics": metrics,
        }
        results.append(fold_result)

    # Compute weighted averages and stds
    summary = {}
    if not all_metrics:
        print("⚠️ Warning: no metrics were collected. Did you pass a valid metric_fn?")
    else:
        for k, values in all_metrics.items():
            vals = np.array([v for v, _ in values])
            weights = np.array([w for _, w in values]) / total_signals

            weighted_avg = np.sum(vals * weights)
            std = np.std(vals, ddof=1)

            summary[f"{k}_weighted_mean"] = weighted_avg
            summary[f"{k}_std"] = std

    return results, summary

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def sample_wise_cv(pipeline, X, y, metric_fn=None, n_splits=5, random_state=42):
    """
    Run sample-wise cross-validation (ignores patient grouping).

    Args:
        pipeline: sklearn Pipeline (scaler + model)
        X (pd.DataFrame or np.ndarray): Features
        y (pd.Series or np.ndarray): Target values
        metric_fn: function(y_true, y_pred) -> dict or float
        n_splits: number of folds
        random_state: random seed for reproducibility

    Returns:
        results: list of per-fold dicts (fold info + metrics)
        weighted_metrics: dict of weighted averages across folds
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []
    all_metrics = {}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        # Compute metrics
        metrics = metric_fn(y_val, y_pred) if metric_fn else {}
        if isinstance(metrics, float):
            metrics = {"score": metrics}

        # Update weighted averages
        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

        results.append({
            "fold": fold,
            "n_train_signals": len(train_idx),
            "n_val_signals": len(val_idx),
            "metrics": metrics,
        })

    # Compute mean + std across folds
    summary = {f"{k}_mean": np.mean(v) for k, v in all_metrics.items()}
    summary.update({f"{k}_std": np.std(v, ddof=1) for k, v in all_metrics.items()})

    return results, summary

# ----------------------------
# Reporting Functions
# ----------------------------
def print_patient_wise_cv(results, weighted_metrics):
    # Print per-fold
    for fold in results:
        print(f"\nFold {fold['fold']}:")
        print(f"  Validation patients: {fold['val_patients']}")
        print(f"  n_val_signals: {fold['n_val_signals']}, n_val_patients: {fold['n_val_patients']}")
        for k, v in fold["metrics"].items():
            print(f"    {k}: {v:.4f}")

    # Print weighted average
    print("\n=== Weighted Average Across Folds ===")
    for k, v in weighted_metrics.items():
        print(f"{k}: {v:.4f}")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def print_sample_wise_cv(results, summary):
    # Print per-fold
    for fold in results:
        print(f"\nFold {fold['fold']}:")
        print(f"  n_val_signals: {fold['n_val_signals']}")
        for k, v in fold["metrics"].items():
            print(f"    {k}: {v:.4f}")

    # Print overall summary
    print("\n=== Mean ± SD Across Folds ===")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")