############ ABLATION: TEST-SET BOOTSTRAP CI ##############################
#                                                                        #
# Computes confidence intervals from the test-set BOOTSTRAP              #
# distributions (test-evaluation resampling noise, NOT the training-    #
# draw variability the figures' error bars show) produced by            #
# Abblation_SampleSize_Uncertainty.py and                               #
# Abblation_SubjectSize_Uncertainty.py                                  #
# (TestBootstrap_Distribution_{SBP,DBP,MAP}.csv, 1,000 resamples each   #
# level). Point estimate = the canonical (draw_seed==42) Test row from  #
# the matching TrainDraws_Distribution.csv — the same model the        #
# bootstrap ran on.                                                     #
#                                                                        #
# Cheap to rerun — no model training, no bootstrap loop, no randomness. #
# Mirrors Experiment_Bootstrap_CI.py's pattern, reusing                 #
# eval.compute_bootstrap_ci / eval.save_bootstrap_ci.                   #
#                                                                        #
# Output:                                                                #
#   Per level (alongside the existing distribution files):              #
#     TestBootstrap_CI_{SBP,DBP,MAP}.csv   — all metrics, one row each  #
#   Per ablation type, one combined summary table across all            #
#   variants/levels/targets, R2 only (the metric the figures plot):     #
#     {Sample,Subject}_Size_Uncertainty/Table_TestBootstrap_CI_R2.csv   #
###########################################################################

import pandas as pd

import eval
from local_paths import ABBLATION_RESULTS_PAPER

TARGETS  = ["SBP", "DBP", "MAP"]
VARIANTS = ["Generalisable", "FitBiased"]
ALPHA    = 0.05  # 0.05 -> 95% CI

ABLATIONS = {
    "Sample_Size_Uncertainty": {
        "levels":       [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00],
        "level_dir":    lambda p: f"Sample_Size_{p:.2f}",
        "level_label":  lambda p: f"{p:.2f}",
    },
    "Subject_Size_Uncertainty": {
        "levels":       [25, 50, 100, 200, 400, 600, 800, 1000, 1293],
        "level_dir":    lambda n: f"Subject_Size_{n}",
        "level_label":  lambda n: str(n),
    },
}


def load_point_row(level_path, target):
    df = pd.read_csv(level_path / "TrainDraws_Distribution.csv")
    row = df.loc[(df["target"] == target) & (df["subset"] == "Test") & df["used_for_bootstrap"]]
    return row.iloc[0]


if __name__ == "__main__":
    for ablation_name, spec in ABLATIONS.items():
        base_path = ABBLATION_RESULTS_PAPER / ablation_name
        print(f"\n=== {ablation_name} ===")

        summary_rows = []

        for variant in VARIANTS:
            for level in spec["levels"]:
                level_path = base_path / variant / spec["level_dir"](level)
                if not level_path.exists():
                    print(f"  MISSING: {level_path}")
                    continue

                for target in TARGETS:
                    dist_df = pd.read_csv(level_path / f"TestBootstrap_Distribution_{target}.csv")

                    point_row = load_point_row(level_path, target)
                    point_metrics = {m: point_row[m] for m in dist_df.columns}

                    ci_df = eval.compute_bootstrap_ci(dist_df, alpha=ALPHA)
                    eval.save_bootstrap_ci(
                        ci_df,
                        path=level_path / f"TestBootstrap_CI_{target}.csv",
                        point_metrics=point_metrics,
                    )

                    r2_row = ci_df.loc[ci_df["metric"] == "R2"].iloc[0]
                    summary_rows.append({
                        "variant":           variant,
                        "level":             spec["level_label"](level),
                        "target":            target,
                        "point_estimate_R2": point_metrics["R2"],
                        "ci_lower_R2":       r2_row["ci_lower"],
                        "ci_upper_R2":       r2_row["ci_upper"],
                        "bootstrap_mean_R2": r2_row["bootstrap_mean"],
                        "bootstrap_std_R2":  r2_row["bootstrap_std"],
                    })

        # One combined R2 summary table for this ablation type
        summary_df = pd.DataFrame(summary_rows)
        table_path = base_path / "Table_TestBootstrap_CI_R2.csv"
        summary_df.to_csv(table_path, index=False)
        print(f"\nSaved summary table: {table_path}")
        print(summary_df.to_string(index=False))

    print("\nDone.")
