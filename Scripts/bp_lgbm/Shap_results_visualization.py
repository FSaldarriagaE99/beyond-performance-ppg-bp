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
    # Load all the needed shapley results
    #==========================================================
    targets = ["SBP", "DBP", "MAP"]
    """
    res_path_80 = SHAP_RESULTS_PAPER/ "Clean_80_DS"
    """
    res_path_orig = SHAP_RESULTS_PAPER/ "StressTest_Original"

    for target in targets:
        #------ RANK PLOTS --------------------------------------
        top_k = 10
        df_rank_matrix_orig = pd.read_csv(res_path_orig/f"{target}_rank_matrix.csv")
        """
        df_rank_matrix_80 = pd.read_csv(res_path_80/f"{target}_rank_matrix.csv")
        """
        feature_names = df_rank_matrix_orig.columns.to_list()
        """
        sa.plot_rank_boxplot(rank_matrix=df_rank_matrix_80,
                            feature_names= feature_names,
                            top_k=top_k,
                            show=False,
                            sort_by="hybrid",
                            alpha=1,
                            save_path=res_path_80/f"top_{top_k}_rank_features_{target}.png")
        """
        sa.plot_rank_boxplot(rank_matrix=df_rank_matrix_orig,
                            feature_names= feature_names,
                            top_k=top_k,
                            show=False,
                            sort_by="hybrid",
                            alpha=1,
                            save_path=res_path_orig/f"top_{top_k}_rank_features_{target}.png")
        """
        sa.plot_mean_std_scatter(rank_matrix=df_rank_matrix_80,
                                feature_names=feature_names,
                                show=False,
                                save_path=res_path_80/f"feature_rank_dispersion_{target}.png")
        """
        sa.plot_mean_std_scatter(rank_matrix=df_rank_matrix_orig,
                                feature_names=feature_names,
                                show=False,
                                save_path=res_path_orig/f"feature_rank_dispersion_{target}.png")
        """
        sa.plot_mean_std_scatter(rank_matrix=df_rank_matrix_orig,
                                rank_matrix_2=df_rank_matrix_80,
                                labels= ("Original dataset", r"80 % confidence"),
                                feature_names=feature_names,
                                show=False,
                                save_path=SHAP_RESULTS_PAPER/f"feature_rank_dispersion_{target}_comparison.png")
        """
        #------ ABS SHAP PLOTS ------------------------------------
        """
        abs_df_80 = pd.read_csv(res_path_80/f"{target}_abs_shap_matrix.csv")
        """
        abs_df_orig = pd.read_csv(res_path_orig/f"{target}_abs_shap_matrix.csv")
        """
        sa.plot_shap_magnitude(shap_abs_matrix= abs_df_80,
                                    show=False,
                                    save_path=res_path_80/f"mean_feature_shap_contribution_{target}.png")
        """
        sa.plot_shap_magnitude(shap_abs_matrix= abs_df_orig,
                                    show=False,
                                    save_path=res_path_orig/f"mean_feature_shap_contribution_{target}.png")
        """
        sa.plot_shap_magnitude(shap_abs_matrix= abs_df_orig,
                               shap_abs_matrix_2= abs_df_80,
                               labels=("Original dataset", r"80 % confidence"),
                               show=False,
                               save_path=SHAP_RESULTS_PAPER/f"mean_feature_shap_contribution_{target}_comparison.png")
        """
        #------ RAW SHAP PLOTS ------------------------------------
        """
        raw_df_80 = pd.read_csv(res_path_80/f"{target}_raw_shap_matrix.csv")
        sa.plot_shap_directionality(shap_signed_matrix=raw_df_80,
                                    show=False,
                                    save_path=res_path_80/f"feature_direction_{target}.png")
        """
        raw_df_orig = pd.read_csv(res_path_orig/f"{target}_raw_shap_matrix.csv")
        sa.plot_shap_directionality(shap_signed_matrix=raw_df_orig,
                                    show=False,
                                    save_path=res_path_orig/f"feature_direction_{target}.png")