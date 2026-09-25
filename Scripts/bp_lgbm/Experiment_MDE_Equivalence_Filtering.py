############ MDE + EQUIVALENCE — FILTERING CONTRAST ###########################
#                                                                              #
# Post-processing only: no model training, no re-inference. Computes minimum  #
# detectable effect (MDE) and an equivalence test for the null-result         #
# filtering contrast (filtered vs unfiltered training set), reported in the   #
# manuscript as a non-significant difference in test R2.                     #
#                                                                              #
# There is no pre-saved paired-difference distribution file. The paired diff  #
# is reconstructed here from the two per-model Distribution_{target}.csv      #
# files under Bootstrap_Filtering/, which are row-aligned by construction     #
# (Experiment_Bootstrap_Filtering.py allocates one `resamples` list and       #
# reuses it for both models in the same loop).                                #
#                                                                              #
# Statistics (CI / MDE / equivalence) come from bootstrap_stats.py, shared    #
# with Sweep_Threshold_ML.py.                                                 #
#                                                                              #
# Saves:                                                                      #
#   Bootstrap_Filtering/MDE_Equivalence_Filtering.csv                         #
###############################################################################

import numpy as np
import pandas as pd

from local_paths import PERFORMANCE_RESULTS_PAPER
from bootstrap_stats import paired_diff_stats

TARGETS  = ["SBP", "DBP", "MAP"]
CONTRAST = "filtered_vs_unfiltered"


def load_point_r2(res_dir, target):
    pt = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
    return float(pt.loc[pt["metric"] == "R2", "value"].iloc[0])


if __name__ == "__main__":

    res_root = PERFORMANCE_RESULTS_PAPER / "Bootstrap_Filtering"

    rows = []
    for target in TARGETS:
        dist_filtered = pd.read_csv(res_root / "ppg_filtered" / f"Distribution_{target}.csv")
        dist_original = pd.read_csv(res_root / "ppg_original" / f"Distribution_{target}.csv")

        if len(dist_filtered) != len(dist_original):
            raise RuntimeError(
                f"{target}: row-count mismatch between ppg_filtered "
                f"({len(dist_filtered)}) and ppg_original ({len(dist_original)}) "
                f"distributions — files are not paired by resample index."
            )

        # Paired difference: same resample index -> same subject draw for both
        # models (allocated once and reused across both loops upstream).
        diff_pp = (dist_filtered["R2"] - dist_original["R2"]).to_numpy() * 100.0

        observed_diff_pp = (
            load_point_r2(res_root / "ppg_filtered", target)
            - load_point_r2(res_root / "ppg_original", target)
        ) * 100.0

        stats = paired_diff_stats(diff_pp, observed_diff_pp)

        rows.append({
            "contrast": CONTRAST,
            "target":   target,
            **stats,
        })

    result_df = pd.DataFrame(rows)

    out_path = res_root / "MDE_Equivalence_Filtering.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n=== MDE / Equivalence - filtering contrast (pp of R2) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(result_df.to_string(index=False))
