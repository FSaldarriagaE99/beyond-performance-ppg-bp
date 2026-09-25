############ VISUALIZATION SHAPLEY VALUES #####################
#                                                             #
# Here, all the results from shap will be visualized          #
#                                                             #
###############################################################

import pandas as pd
import shap_analysis as sa
from local_paths import SHAP_RESULTS_PAPER
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

    # =========================================================
    # Parameters
    # =========================================================
    targets = ["SBP", "DBP", "MAP"]
    root_path = SHAP_RESULTS_PAPER / "Stratified_1variable_Stress_Test"
    top_k = 10

    # =========================================================
    # Iterate over variable and strata folders
    # =========================================================
    for variable_dir in root_path.iterdir():
        if not variable_dir.is_dir():
            continue  # Skip any stray files
        variable = variable_dir.name
        print(f"\n📂 Processing variable: {variable}")

        for stratum_dir in variable_dir.iterdir():
            if not stratum_dir.is_dir():
                continue
            stratum_label = stratum_dir.name
            print(f"   ▶️  {stratum_label}")

            for target in targets:
                try:
                    # --- Load results
                    df_rank = pd.read_csv(stratum_dir / f"{target}_rank_matrix.csv")
                    df_abs = pd.read_csv(stratum_dir / f"{target}_abs_shap_matrix.csv")
                    df_raw = pd.read_csv(stratum_dir / f"{target}_raw_shap_matrix.csv")
                    feature_names = df_rank.columns.to_list()

                    # --- RANK plots
                    sa.plot_rank_boxplot(
                        rank_matrix=df_rank,
                        feature_names=feature_names,
                        top_k=top_k,
                        show=False,
                        sort_by="hybrid",
                        alpha=1,
                        save_path=stratum_dir / f"top_{top_k}_rank_features_{target}.png"
                    )

                    sa.plot_mean_std_scatter(
                        rank_matrix=df_rank,
                        feature_names=feature_names,
                        show=False,
                        save_path=stratum_dir / f"feature_rank_dispersion_{target}.png"
                    )

                    # --- ABS SHAP plots
                    sa.plot_shap_magnitude(
                        shap_abs_matrix=df_abs,
                        show=False,
                        save_path=stratum_dir / f"mean_feature_shap_contribution_{target}.png"
                    )

                    # --- RAW SHAP plots
                    sa.plot_shap_directionality(
                        shap_signed_matrix=df_raw,
                        show=False,
                        save_path=stratum_dir / f"feature_direction_{target}.png"
                    )

                    print(f"      ✅ Finished {target}")

                except FileNotFoundError as e:
                    print(f"      ⚠️ Missing files for {target} in {stratum_label}: {e.filename}")
                except Exception as e:
                    print(f"      ❌ Error for {target} in {stratum_label}: {e}")