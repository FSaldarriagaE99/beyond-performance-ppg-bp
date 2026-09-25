############ MDE + EQUIVALENCE — AGE >=60 vs FULL, PAIRED (SAME TEST SET) ####
#                                                                             #
# Post-processing only: no model training, no re-inference. Computes the     #
# minimum detectable effect (MDE) and an equivalence test for the Age>=60    #
# vs Full contrast produced by Experiment_Bootstrap_AgeStratified_Paired.py. #
#                                                                             #
# Unlike Experiment_MDE_Equivalence_AgeStratified.py (the original, which    #
# bootstraps two independent, unequal-sized pools and has to fall back to a  #
# Wald/quadrature CI), this contrast IS paired: both arms are evaluated on   #
# the literal same Age>=60 test subjects, driven by one shared bootstrap     #
# resample list, so row i of both Distribution_{target}.csv files comes      #
# from the same resampled subjects. Reconstructs the paired diff by loading  #
# both files and subtracting row-wise — same pattern as                     #
# Experiment_MDE_Equivalence_Filtering.py.                                   #
#                                                                             #
# Saves:                                                                     #
#   Bootstrap_AgeStratified_Paired/                                          #
#     MDE_Equivalence_AgeStratified_Paired.csv                               #
###############################################################################

import numpy as np
import pandas as pd

from local_paths import PERFORMANCE_RESULTS_PAPER

TARGETS  = ["SBP", "DBP", "MAP"]
CONTRAST = "age_gte60_vs_full_model_paired_same_test"

Z_MDE       = 2.8    # (1.96 + 0.84), two-sided alpha=0.05, 80% power
EQUIV_DELTA = 2.0    # equivalence margin, pp of R2, declared a priori

OBSERVED_DIFF_TOL_PP = 1.0  # loose structural sanity check, not a manuscript match


def load_point_r2(res_dir, target):
    pt = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
    return float(pt.loc[pt["metric"] == "R2", "value"].iloc[0])


if __name__ == "__main__":

    res_root  = PERFORMANCE_RESULTS_PAPER / "Bootstrap_AgeStratified_Paired"
    dir_full  = res_root / "full_model"
    dir_gte60 = res_root / "age_gte60"

    rows = []
    for target in TARGETS:
        dist_full  = pd.read_csv(dir_full  / f"Distribution_{target}.csv")
        dist_gte60 = pd.read_csv(dir_gte60 / f"Distribution_{target}.csv")

        if len(dist_full) != len(dist_gte60):
            raise RuntimeError(
                f"{target}: row-count mismatch between full_model "
                f"({len(dist_full)}) and age_gte60 ({len(dist_gte60)}) "
                f"distributions — files are not paired by resample index."
            )

        # Paired difference: both arms evaluated on the same shared bootstrap
        # resamples of the same test subjects, so row i of both distributions
        # comes from the same underlying draw.
        diff_pp = (dist_gte60["R2"] - dist_full["R2"]).to_numpy() * 100.0

        observed_diff_pp = (
            load_point_r2(dir_gte60, target) - load_point_r2(dir_full, target)
        ) * 100.0

        sd_pp  = float(np.std(diff_pp, ddof=1))
        mde_pp = Z_MDE * sd_pp

        ci95_lower_pp, ci95_upper_pp = np.percentile(diff_pp, [2.5, 97.5])
        ci90_lower_pp, ci90_upper_pp = np.percentile(diff_pp, [5, 95])

        equivalent = bool((ci90_lower_pp > -EQUIV_DELTA) and (ci90_upper_pp < EQUIV_DELTA))

        rows.append({
            "contrast":         CONTRAST,
            "target":           target,
            "n_resamples":      len(diff_pp),
            "observed_diff_pp": observed_diff_pp,
            "mean_diff_pp":     float(np.mean(diff_pp)),
            "sd_pp":            sd_pp,
            "mde_pp":           mde_pp,
            "ci95_lower_pp":    float(ci95_lower_pp),
            "ci95_upper_pp":    float(ci95_upper_pp),
            "ci90_lower_pp":    float(ci90_lower_pp),
            "ci90_upper_pp":    float(ci90_upper_pp),
            "equivalent":       equivalent,
        })

    result_df = pd.DataFrame(rows)

    # =========================================================
    # Sanity checks — stop before saving if any fail
    # =========================================================
    for row in rows:
        target = row["target"]

        ci95_width = row["ci95_upper_pp"] - row["ci95_lower_pp"]
        ci90_width = row["ci90_upper_pp"] - row["ci90_lower_pp"]
        if not (ci90_width < ci95_width):
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): 90% CI width ({ci90_width:.3f} pp) "
                f"is not strictly narrower than 95% CI width ({ci95_width:.3f} pp)."
            )

        if not (row["mde_pp"] > 0):
            raise RuntimeError(f"SANITY CHECK FAILED ({target}): MDE is not positive.")

        # Point-estimate diff and bootstrap-mean diff should be in rough
        # agreement (both estimate the same quantity from the same models).
        if abs(row["observed_diff_pp"] - row["mean_diff_pp"]) > OBSERVED_DIFF_TOL_PP:
            raise RuntimeError(
                f"SANITY CHECK FAILED ({target}): point-estimate diff "
                f"({row['observed_diff_pp']:.2f} pp) and bootstrap-mean diff "
                f"({row['mean_diff_pp']:.2f} pp) disagree by more than "
                f"{OBSERVED_DIFF_TOL_PP} pp — check pairing is correct."
            )

    print("Sanity checks passed: 90% CIs are narrower than 95% CIs, all MDEs are "
          "positive, point-estimate and bootstrap-mean diffs agree.\n")

    # =========================================================
    # Save + print
    # =========================================================
    out_path = res_root / "MDE_Equivalence_AgeStratified_Paired.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\n=== MDE / Equivalence - age>=60 vs full model, same test set (pp of R2) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(result_df.to_string(index=False))
