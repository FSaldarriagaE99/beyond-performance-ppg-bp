############ BOOTSTRAP PAIRED CI — STRESS TEST vs GS CONFIG (PPG only) #######
#                                                                              #
# Compares the PPG-only model across two configurations:                       #
#   ppg_stresstest : trained with stress_test_params  (Bootstrap_StressTest/) #
#   ppg_gs         : trained with GS config           (Bootstrap/)            #
#                                                                              #
# Both evaluated on the same original test set — distributions already saved. #
# Paired difference: stresstest − gs                                           #
# Positive values → stress test config is better.                              #
#                                                                              #
# Saves to:                                                                    #
#   Bootstrap_StressTest/paired/ppg_stresstest_minus_ppg_gs/CI_{target}.csv   #
###############################################################################

import pandas as pd

import eval
from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS = ["SBP", "DBP", "MAP"]
ALPHA   = 0.05

if __name__ == "__main__":

    root_gs          = PERFORMANCE_RESULTS_PAPER / "Bootstrap"
    root_stress      = PERFORMANCE_RESULTS_PAPER / "Bootstrap_StressTest"
    save_dir         = root_stress / "paired" / "ppg_stresstest_minus_ppg_gs"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=== Paired CI: PPG stress test − PPG GS ===")
    for target in TARGETS:
        dist_stress = pd.read_csv(root_stress / "ppg" / f"Distribution_{target}.csv")
        dist_gs     = pd.read_csv(root_gs     / "ppg" / f"Distribution_{target}.csv")

        diff_df = dist_stress - dist_gs

        pt_stress = dict(zip(
            pd.read_csv(root_stress / "ppg" / f"PointEstimates_{target}.csv")["metric"],
            pd.read_csv(root_stress / "ppg" / f"PointEstimates_{target}.csv")["value"],
        ))
        pt_gs = dict(zip(
            pd.read_csv(root_gs / "ppg" / f"PointEstimates_{target}.csv")["metric"],
            pd.read_csv(root_gs / "ppg" / f"PointEstimates_{target}.csv")["value"],
        ))
        point_diff = {m: pt_stress[m] - pt_gs[m] for m in pt_stress}

        ci_df = eval.compute_bootstrap_ci(diff_df, alpha=ALPHA)
        eval.save_bootstrap_ci(
            ci_df,
            path=save_dir / f"CI_{target}.csv",
            point_metrics=point_diff,
        )

    print("\nDone.")
