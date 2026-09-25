######## EVAL SCRIPT #################################################################
#                                                                                    #
# This script handles the model eval with all of the apropriate metrics              #
# for BP estimation evaluation according to Elgendi (2024                            #
#                                                                                    #
######################################################################################
"""
Citation:

Elgendi, M., Haugg, F., Fletcher, R.R. et al. 
Recommendations for evaluating photoplethysmography-based algorithms for blood pressure assessment. 
Commun Med 4, 140 (2024). https://doi.org/10.1038/s43856-024-00555-2
"""

# Metrics: MAE (+SD), ME (+SD) - or SDE, RMSE, MSE, R2, absolute errors, Bland–Altman plot (7 metrics)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, List
from pathlib import Path

# ----------------------------
# Evaluation
# ----------------------------
def bland_altman_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str = "bland_altman.png") -> Dict[str, float]:
    """
    Bland–Altman: difference vs mean, with bias and 95% LoA.
    Returns bias and LoA stats; saves plot to file.
    """
    diffs = y_pred - y_true
    means = (y_pred + y_true) / 2.0

    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    loa_low = bias - 1.96 * sd
    loa_high = bias + 1.96 * sd

    plt.figure(figsize=(7, 6), dpi=140)
    plt.scatter(means, diffs, alpha=0.4, s=12, color="steelblue", edgecolor="none")

    # Bias line in black, thicker
    plt.axhline(bias, color="black", linestyle="--", linewidth=2, label=f"Bias = {bias:.2f}")

    # LoA lines in red, thicker
    plt.axhline(loa_low, color="red", linestyle=":", linewidth=2, label=f"LoA low = {loa_low:.2f}")
    plt.axhline(loa_high, color="red", linestyle=":", linewidth=2, label=f"LoA high = {loa_high:.2f}")

    # Labels with formulas in parentheses
    plt.xlabel("Mean of prediction and reference ( (ŷ + y) / 2 )", fontsize=12)
    plt.ylabel("Prediction − Reference ( ŷ − y )", fontsize=12)

    # Title larger and bold
    plt.title("Bland–Altman Plot", fontsize=16, weight="bold")

    # Grid for readability
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(loc="best", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return {
        "bias_ME": float(bias),
        "sd_diff": float(sd),
        "loa_low": float(loa_low),
        "loa_high": float(loa_high),
        "plot_path": path,
    }

def r2_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str = "r2_plot.png"):
    """
    Scatter plot of predicted vs. true values with R² score.
    Adds continuous identity line, shaded dispersion band, and saves to file.
    """

    r2 = r2_score(y_true, y_pred)

    # Compute spread (std of residuals)
    residuals = y_pred - y_true
    spread = np.std(residuals, ddof=1)

    plt.figure(figsize=(7, 6), dpi=140)
    plt.scatter(y_true, y_pred, alpha=0.4, s=12, color="steelblue", edgecolor="none")

    # Common limits
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    x_vals = np.linspace(min_val, max_val, 100)

    # Identity line (continuous black)
    plt.plot(x_vals, x_vals, "k-", linewidth=2, label="Identity (y = x)")

    # Shadow band ±1 SD of residuals
    plt.fill_between(x_vals, x_vals - spread, x_vals + spread,
                     color="gray", alpha=0.2, label=f"±1 SD ({spread:.2f})")

    # Enforce square axes
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Labels (not bold)
    plt.xlabel("True values (y)", fontsize=12)
    plt.ylabel("Predicted values (ŷ)", fontsize=12)
    plt.title("Predicted vs True", fontsize=16, weight="bold")

    # Grid and legend
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend([f"R² = {r2:.3f}", f"±1 SD = {spread:.2f}"], loc="best", frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return r2


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics without generating any plots.

    Mirrors the scalar outputs of evaluate(). Use this inside bootstrap
    loops where generating plots for every resample is undesirable.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted values.

    Returns
    -------
    dict of scalar metrics (no plot paths).
    """
    errors   = y_pred - y_true
    abs_errors = np.abs(errors)
    bias = float(np.mean(errors))
    sd   = float(np.std(errors, ddof=1))

    return {
        "MAE":        float(mean_absolute_error(y_true, y_pred)),
        "MAE_SD":     float(np.std(abs_errors, ddof=1)),
        "ME":         bias,
        "SDE":        sd,
        "MSE":        float(mean_squared_error(y_true, y_pred)),
        "RMSE":       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":         float(r2_score(y_true, y_pred)),
        "BA_bias_ME": bias,
        "BA_sd_diff": sd,
        "BA_loa_low":  float(bias - 1.96 * sd),
        "BA_loa_high": float(bias + 1.96 * sd),
    }


def compute_metrics_macro_by_subject(y_true: np.ndarray, y_pred: np.ndarray,
                                      subject_ids) -> Dict[str, float]:
    """
    Macro-average metrics: compute_metrics() is applied WITHIN each subject
    first, then those per-subject values are averaged ACROSS subjects
    (unweighted - each subject counts once regardless of how many signals
    they contribute). This differs from the usual pooled/micro-average
    (compute_metrics() called once over all signals), which implicitly
    weights subjects by their signal count.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Ground-truth and predicted values, one row per signal.
    subject_ids : array-like
        Subject id per signal, same length/order as y_true/y_pred.

    Returns
    -------
    dict with "{metric}_macro_mean" and "{metric}_macro_std" (std across
    subjects, ddof=1) for every metric compute_metrics() returns, plus
    "n_subjects".
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "Subject": np.asarray(subject_ids),
    })

    per_subject = pd.DataFrame([
        compute_metrics(g["y_true"].to_numpy(), g["y_pred"].to_numpy())
        for _, g in df.groupby("Subject")
    ])

    macro_mean = per_subject.mean()
    macro_std = per_subject.std(ddof=1)

    result = {f"{col}_macro_mean": float(macro_mean[col]) for col in per_subject.columns}
    result.update({f"{col}_macro_std": float(macro_std[col]) for col in per_subject.columns})
    result["n_subjects"] = int(df["Subject"].nunique())
    return result


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, R2_path: str, BA_path: str) -> Dict[str, float]:
    """
    Compute requested metrics:
    - MAE (+SD of |error|)
    - ME
    - SDE (SD of ME)
    - RMSE, MSE
    - R2
    - Absolute errors (returned as array for further analysis)
    - Bland–Altman stats (also saves a plot)
    """
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = mean_absolute_error(y_true, y_pred)
    mae_sd = float(np.std(abs_errors, ddof=1))

    me = float(np.mean(errors))
    sde = float(np.std(errors, ddof=1))

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    r2 = r2_plot(y_true, y_pred, path=Path(R2_path))

    ba_stats = bland_altman_plot(y_true, y_pred, path=Path(BA_path))

    metrics = {
        "MAE": float(mae),
        "MAE_SD": float(mae_sd),
        "ME": me,
        "SDE": sde,
        "MSE": float(mse),
        "RMSE": rmse,
        "R2": float(r2),
        "AbsError_mean": float(np.mean(abs_errors)),
        "AbsError_std": float(np.std(abs_errors, ddof=1)),
        "AbsError_min": float(np.min(abs_errors)),
        "AbsError_max": float(np.max(abs_errors)),
        # Bland–Altman
        "BA_bias_ME": ba_stats["bias_ME"],
        "BA_sd_diff": ba_stats["sd_diff"],
        "BA_loa_low": ba_stats["loa_low"],
        "BA_loa_high": ba_stats["loa_high"],
        "BA_plot_path": ba_stats["plot_path"],
    }

    return metrics

def allocate_bootstrap_resamples(
    groups,
    n_resamples: int = 1000,
    random_state: int = 42
) -> list:
    """
    Pre-allocate subject ID lists for bootstrap resampling (with replacement).

    Sampling is subject-level to preserve within-subject signal correlation.
    Each resample contains the same number of subjects as the original set,
    drawn with replacement (so some subjects may appear multiple times).

    Parameters
    ----------
    groups : array-like of shape (n_samples,)
        Subject IDs aligned with the dataset rows (e.g. X["Subject"]).
    n_resamples : int, default=1000
        Number of bootstrap resamples to pre-allocate.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    resamples : list of np.ndarray, length n_resamples
        Each entry is an array of subject IDs sampled with replacement,
        of length equal to the number of unique subjects.
    """
    rng = np.random.default_rng(random_state)
    unique_subjects = np.unique(groups)
    n_subjects = len(unique_subjects)

    return [
        rng.choice(unique_subjects, size=n_subjects, replace=True)
        for _ in range(n_resamples)
    ]


def build_subject_index(df: pd.DataFrame, subject_col: str = "Subject") -> Dict:
    """
    Precompute a subject → row-label index mapping for fast bootstrap sampling.

    Call this once on the test set before the bootstrap loop; pass the result
    to build_bootstrap_sample on every iteration.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to index (typically the test set).
    subject_col : str, default="Subject"
        Column name identifying subjects.

    Returns
    -------
    dict mapping subject_id → np.ndarray of df index labels
    """
    return {
        subject_id: idx.to_numpy()
        for subject_id, idx in df.groupby(subject_col).groups.items()
    }


def build_bootstrap_sample(
    df: pd.DataFrame,
    resample,
    subject_index: Dict,
) -> pd.DataFrame:
    """
    Build one bootstrap sample using a precomputed subject index.

    Subjects that appear multiple times in `resample` have their rows
    duplicated accordingly, preserving within-subject signal structure.

    Parameters
    ----------
    df : pd.DataFrame
        Full test set (same one passed to build_subject_index).
    resample : array-like of subject IDs
        One entry from allocate_bootstrap_resamples().
    subject_index : dict
        Output of build_subject_index — subject_id → row label array.

    Returns
    -------
    pd.DataFrame
        Resampled dataset with reset index.
    """
    idx = np.concatenate([subject_index[s] for s in resample])
    return df.loc[idx].reset_index(drop=True)


def compute_bootstrap_ci(
    distribution: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals from a distribution of metric values.

    Input-agnostic: works on any DataFrame where each row is one bootstrap
    resample and each column is a metric — whether those are raw evaluation
    metrics or paired differences between models.

    Parameters
    ----------
    distribution : pd.DataFrame
        Shape (n_resamples, n_metrics). Typically loaded from a
        Distribution_{target}.csv saved by Experiment_Bootstrap.py.
    alpha : float, default=0.05
        Significance level. 0.05 → 95% CI.

    Returns
    -------
    pd.DataFrame with columns:
        metric, ci_lower, ci_upper, bootstrap_mean, bootstrap_std
    """
    rows = []
    for col in distribution.columns:
        values = distribution[col].values
        rows.append({
            "metric":         col,
            "ci_lower":       float(np.percentile(values, 100 * alpha / 2)),
            "ci_upper":       float(np.percentile(values, 100 * (1 - alpha / 2))),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_std":  float(np.std(values, ddof=1)),
        })
    return pd.DataFrame(rows)


def save_bootstrap_ci(
    ci_df: pd.DataFrame,
    path,
    point_metrics: dict = None,
) -> None:
    """
    Save a CI DataFrame to CSV, optionally inserting point estimates.

    Parameters
    ----------
    ci_df : pd.DataFrame
        Output of compute_bootstrap_ci.
    path : str or Path
        Destination CSV path. Parent directory is created if needed.
    point_metrics : dict, optional
        {metric_name: scalar_value} from evaluation on the full test set
        (no resampling). Pass None to omit the point_estimate column —
        useful for paired-difference CIs where there is no single point
        estimate.
    """
    ci_df = ci_df.copy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if point_metrics is not None:
        ci_df.insert(1, "point_estimate", ci_df["metric"].map(point_metrics))

    ci_df.to_csv(path, index=False)
    print(f"Saved: {path}")


def save_results_dict(res_dict: dict, save_path: str, subset_col: str = "subset"):
    """
    Convert a dictionary of dictionaries into a DataFrame and save as CSV.

    Args:
        res_dict (dict): e.g., {"val": val_dict, "test": test_dict, ...}
        save_path (str): path to save the resulting CSV
        subset_col (str): name of the column for the outer dictionary keys (default="subset")
    """
    # Convert nested dict into a DataFrame
    df = pd.DataFrame.from_dict(res_dict, orient="index")

    # Add a column with the subset/experiment name
    df.reset_index(inplace=True)
    df.rename(columns={"index": subset_col}, inplace=True)

    # Save to CSV
    df.to_csv(save_path, index=False)

    print(f"Results saved to {save_path}")
    return df