############ BOOTSTRAP CI COMPUTATION #################################
#                                                                      #
# Loads the bootstrap artifacts produced by Experiment_Bootstrap.py   #
# and computes confidence intervals.                                   #
#                                                                      #
# Cheap to rerun — no model training, no bootstrap loop.              #
#                                                                      #
# Produces two types of output:                                        #
#                                                                      #
# 1. Standalone CIs (Bootstrap/{model}/CI_{target}.csv)               #
#    For each of the three models: ppg, demo, ppg_demo.               #
#    Includes point_estimate column (metrics on the full test set).   #
#                                                                      #
# 2. Paired-difference CIs (Bootstrap/paired/{name}/CI_{target}.csv)  #
#    Computed as dist_A − dist_B column-wise (same bootstrap draws,   #
#    so the pairing is exact). Negative values favour model A.        #
#    Includes point_estimate column (difference of point estimates).  #
#                                                                      #
#    ppg_minus_demo            : PPG − Demo     (is PPG better than demo?)  #
#    ppg_minus_ppg_demographics: PPG − PPG+Demo (cost of adding demographics)#
########################################################################

import pandas as pd

import eval
from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS    = ["SBP", "DBP", "MAP"]
MODEL_KEYS = ["ppg", "demo", "ppg_demo"]
ALPHA      = 0.05   # 0.05 → 95% CI

PAIRED_COMPARISONS = [
    # (name,                        model_A,    model_B)
    ("ppg_minus_demo",               "ppg",      "demo"),
    ("ppg_minus_ppg_demographics",   "ppg",      "ppg_demo"),
]


def load_distribution(res_root, model_key, target):
    return pd.read_csv(res_root / model_key / f"Distribution_{target}.csv")


def load_point_metrics(res_root, model_key, target):
    pt_df = pd.read_csv(res_root / model_key / f"PointEstimates_{target}.csv")
    return dict(zip(pt_df["metric"], pt_df["value"]))


if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap"

    # =========================================================
    # 1. Standalone CIs for each model
    # =========================================================
    print("=== Standalone CIs ===")
    for model_key in MODEL_KEYS:
        for target in TARGETS:
            dist_df      = load_distribution(res_root, model_key, target)
            point_metrics = load_point_metrics(res_root, model_key, target)

            ci_df = eval.compute_bootstrap_ci(dist_df, alpha=ALPHA)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_root / model_key / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    # =========================================================
    # 2. Paired-difference CIs
    # =========================================================
    print("\n=== Paired-difference CIs ===")
    for comp_name, model_a, model_b in PAIRED_COMPARISONS:
        print(f"\n--- {comp_name} ({model_a} − {model_b}) ---")
        save_dir = res_root / "paired" / comp_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for target in TARGETS:
            dist_a = load_distribution(res_root, model_a, target)
            dist_b = load_distribution(res_root, model_b, target)

            # Column-wise difference — valid because both distributions
            # were drawn from the same pre-allocated bootstrap resamples
            diff_df = dist_a - dist_b

            # Point estimate = difference of full-test-set metrics
            pt_a = load_point_metrics(res_root, model_a, target)
            pt_b = load_point_metrics(res_root, model_b, target)
            point_diff = {metric: pt_a[metric] - pt_b[metric] for metric in pt_a}

            ci_df = eval.compute_bootstrap_ci(diff_df, alpha=ALPHA)
            eval.save_bootstrap_ci(
                ci_df,
                path=save_dir / f"CI_{target}.csv",
                point_metrics=point_diff,
            )

    print("\nDone.")
