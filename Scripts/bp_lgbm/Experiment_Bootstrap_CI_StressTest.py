############ BOOTSTRAP CI — STRESS TEST #######################################
#                                                                              #
# Computes standalone CIs from the Bootstrap_StressTest artifacts.             #
# Paired-difference CIs are skipped (commented out).                           #
###############################################################################

import pandas as pd

import eval
from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS    = ["SBP", "DBP", "MAP"]
MODEL_KEYS = ["ppg", "demo", "ppg_demo"]
ALPHA      = 0.05


def load_distribution(res_root, model_key, target):
    return pd.read_csv(res_root / model_key / f"Distribution_{target}.csv")


def load_point_metrics(res_root, model_key, target):
    pt_df = pd.read_csv(res_root / model_key / f"PointEstimates_{target}.csv")
    return dict(zip(pt_df["metric"], pt_df["value"]))


if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap_StressTest"

    # =========================================================
    # 1. Standalone CIs for each model
    # =========================================================
    print("=== Standalone CIs ===")
    for model_key in MODEL_KEYS:
        for target in TARGETS:
            dist_df       = load_distribution(res_root, model_key, target)
            point_metrics = load_point_metrics(res_root, model_key, target)

            ci_df = eval.compute_bootstrap_ci(dist_df, alpha=ALPHA)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_root / model_key / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    # =========================================================
    # 2. Paired-difference CIs — skipped for speed
    # =========================================================
    # PAIRED_COMPARISONS = [
    #     ("ppg_minus_demo",               "ppg",  "demo"),
    #     ("ppg_minus_ppg_demographics",   "ppg",  "ppg_demo"),
    # ]
    # for comp_name, model_a, model_b in PAIRED_COMPARISONS:
    #     save_dir = res_root / "paired" / comp_name
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     for target in TARGETS:
    #         dist_a = load_distribution(res_root, model_a, target)
    #         dist_b = load_distribution(res_root, model_b, target)
    #         diff_df = dist_a - dist_b
    #         pt_a = load_point_metrics(res_root, model_a, target)
    #         pt_b = load_point_metrics(res_root, model_b, target)
    #         point_diff = {metric: pt_a[metric] - pt_b[metric] for metric in pt_a}
    #         ci_df = eval.compute_bootstrap_ci(diff_df, alpha=ALPHA)
    #         eval.save_bootstrap_ci(ci_df, path=save_dir / f"CI_{target}.csv", point_metrics=point_diff)

    print("\nDone.")
