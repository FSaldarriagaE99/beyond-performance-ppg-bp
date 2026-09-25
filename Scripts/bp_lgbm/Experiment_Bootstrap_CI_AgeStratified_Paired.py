############ BOOTSTRAP PAIRED CI — AGE>=60 vs FULL MODEL (SAME TEST SET) #####
#                                                                              #
# Compares two PPG-only models both evaluated on the SAME Age>=60 test        #
# subjects (see Experiment_Bootstrap_AgeStratified_Paired.py):                #
#   age_gte60  : trained on Age>=60-only training data                        #
#   full_model : trained on the full (all-ages) training set                  #
#                                                                              #
# Both arms share one bootstrap resample list (allocated once, reused for     #
# both), so row i of both Distribution_{target}.csv files comes from the      #
# same resampled subjects — a genuinely paired design. Same pattern as        #
# Experiment_Bootstrap_CI_Filtering.py / Experiment_Bootstrap_CI_StressTest_  #
# Paired.py: reconstructs the paired diff row-wise and computes a bootstrap   #
# CI across ALL metrics (not just R2 — see Experiment_MDE_Equivalence_        #
# AgeStratified_Paired.py for the R2-specific MDE/equivalence verdict).       #
#                                                                              #
# Paired difference: age_gte60 − full_model                                   #
#   Positive values -> age-specific model better than the full-population     #
#   model, on the Age>=60 test subjects.                                      #
#                                                                              #
# Saves to:                                                                   #
#   Bootstrap_AgeStratified_Paired/paired/age_gte60_minus_full_model/         #
#     CI_{target}.csv                                                        #
###############################################################################

import pandas as pd

import eval
from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS = ["SBP", "DBP", "MAP"]
ALPHA   = 0.05

if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap_AgeStratified_Paired"
    save_dir = res_root / "paired" / "age_gte60_minus_full_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paired CI: age_gte60 - full_model (same Age>=60 test subjects) ===")
    for target in TARGETS:
        dist_gte60 = pd.read_csv(res_root / "age_gte60"  / f"Distribution_{target}.csv")
        dist_full  = pd.read_csv(res_root / "full_model" / f"Distribution_{target}.csv")

        if len(dist_gte60) != len(dist_full):
            raise RuntimeError(
                f"{target}: row-count mismatch between age_gte60 "
                f"({len(dist_gte60)}) and full_model ({len(dist_full)}) "
                f"distributions — files are not paired by resample index."
            )

        diff_df = dist_gte60 - dist_full

        pt_gte60 = dict(zip(
            pd.read_csv(res_root / "age_gte60" / f"PointEstimates_{target}.csv")["metric"],
            pd.read_csv(res_root / "age_gte60" / f"PointEstimates_{target}.csv")["value"],
        ))
        pt_full = dict(zip(
            pd.read_csv(res_root / "full_model" / f"PointEstimates_{target}.csv")["metric"],
            pd.read_csv(res_root / "full_model" / f"PointEstimates_{target}.csv")["value"],
        ))
        point_diff = {m: pt_gte60[m] - pt_full[m] for m in pt_gte60}

        ci_df = eval.compute_bootstrap_ci(diff_df, alpha=ALPHA)
        eval.save_bootstrap_ci(
            ci_df,
            path=save_dir / f"CI_{target}.csv",
            point_metrics=point_diff,
        )

    print("\nDone.")
