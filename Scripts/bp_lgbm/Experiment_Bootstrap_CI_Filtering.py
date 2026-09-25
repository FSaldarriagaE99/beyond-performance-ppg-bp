############ BOOTSTRAP CI — FILTERING COMPARISON ##############################
#                                                                              #
# Computes standalone and paired-difference CIs from Bootstrap_Filtering.      #
#                                                                              #
# Standalone CIs:                                                              #
#   Bootstrap_Filtering/{model}/CI_{target}.csv                                #
#                                                                              #
# Paired-difference CI (filtered_minus_original):                              #
#   Bootstrap_Filtering/paired/filtered_minus_original/CI_{target}.csv        #
#   Positive values → ppg_filtered better than ppg_original                   #
###############################################################################

import pandas as pd

import eval
from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS    = ["SBP", "DBP", "MAP"]
MODEL_KEYS = ["ppg_original", "ppg_filtered"]
ALPHA      = 0.05


def load_distribution(res_root, model_key, target):
    return pd.read_csv(res_root / model_key / f"Distribution_{target}.csv")


def load_point_metrics(res_root, model_key, target):
    pt_df = pd.read_csv(res_root / model_key / f"PointEstimates_{target}.csv")
    return dict(zip(pt_df["metric"], pt_df["value"]))


if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap_Filtering"

    # =========================================================
    # 1. Standalone CIs
    # =========================================================
    print("=== Standalone CIs ===")
    for model_key in MODEL_KEYS:
        for target in TARGETS:
            dist_df       = load_distribution(res_root, model_key, target)
            point_metrics = load_point_metrics(res_root, model_key, target)
            ci_df         = eval.compute_bootstrap_ci(dist_df, alpha=ALPHA)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_root / model_key / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    # =========================================================
    # 2. Paired-difference CI: ppg_filtered − ppg_original
    # =========================================================
    print("\n=== Paired-difference CI: filtered − original ===")
    comp_name = "filtered_minus_original"
    save_dir  = res_root / "paired" / comp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    for target in TARGETS:
        dist_filtered = load_distribution(res_root, "ppg_filtered", target)
        dist_original = load_distribution(res_root, "ppg_original", target)

        diff_df = dist_filtered - dist_original

        pt_filtered = load_point_metrics(res_root, "ppg_filtered", target)
        pt_original = load_point_metrics(res_root, "ppg_original", target)
        point_diff  = {m: pt_filtered[m] - pt_original[m] for m in pt_filtered}

        ci_df = eval.compute_bootstrap_ci(diff_df, alpha=ALPHA)
        eval.save_bootstrap_ci(
            ci_df,
            path=save_dir / f"CI_{target}.csv",
            point_metrics=point_diff,
        )

    print("\nDone.")
