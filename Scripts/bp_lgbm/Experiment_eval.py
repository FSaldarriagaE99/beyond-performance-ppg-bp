############ EXPERIMENT SHAPLEY VALUES ########################
#                                                             #
# Here, the mutual information experiment will be performed   #
#                                                             #
###############################################################
import preprocessing
import eval
from pathlib import Path
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import ExperimentConfig, load_config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from models import build_lgbm

if __name__ == "__main__":

    # =========================================================
    # Paths
    #==========================================================
    train_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    train_clean_path_90 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset_90.h5"
    train_clean_path_80 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_Train_Subset_80.h5"

    test_clean_path_90 = PULSE_DB_SUP_DIR / "Clean_Features_VitalDB_CalFree_Test_Subset_90.h5"
    test_clean_path_80 = PULSE_DB_SUP_DIR/ "Clean_Features_VitalDB_CalFree_Test_Subset_80.h5"
    test_original_path = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    
    # =========================================================
    # Load Data
    #==========================================================
    feature_names = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", 
                 "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75", 
                 "AUCpi", "IPA",  "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", 
                 "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", 
                 "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt", 
                 "FullSkew", "sdPRV", "IQR_PRV"]
    df_train = load_PulseDB_sup_ds(train_clean_path_80, feature_names=feature_names)
    df_test = load_PulseDB_sup_ds(test_original_path, feature_names=feature_names)
    df_test_90 = load_PulseDB_sup_ds(test_clean_path_90, feature_names=feature_names)
    df_test_80 = load_PulseDB_sup_ds(test_clean_path_80, feature_names=feature_names)
    print(len(df_train["Subject"].unique()))
    # =========================================================
    # Check NaNs and fill them
    #==========================================================
    """
    print("************************************************************")
    print("Check NaNs")
    print("************************************************************")
    
    # Check if any NaN at all
    print("Is there any NaN in: df_features_mean?")
    print(df_train.isna().any().any())
    # Count total number of NaNs
    print(df_train.isna().sum())
    """
    # Fill X NaNs
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col= "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col= "Subject")
    """
    print("Original testing set has NaN?:", df_test.isna().any().any())
    print("90 % clean testing set has NaN?:", df_test_90.isna().any().any())
    print("80 % clean testing set has NaN?:", df_test_80.isna().any().any())
    
    """
    # =========================================================
    # Gender as categorical
    # =========================================================
    df_train["Gender"] = df_train["Gender"].astype("category")
    df_test["Gender"] = df_test["Gender"].astype("category")
    df_test_90["Gender"] = df_test_90["Gender"].astype("category")
    df_test_80["Gender"] = df_test_80["Gender"].astype("category")

    # =========================================================
    # Initialize model
    #==========================================================
    print(df_train.info())
    """
    df_train = preprocessing.downsample_per_patient(df_train, patient_col="Subject", proportion = 0.1)
    print(df_train.info())
    """

    # drop ID columns before training
    id_cols = ["Age","Gender", "Height", "Weight", "BMI", "SF"]
    targets = ["SBP", "DBP", "MAP"]
    id_cols.extend(targets)
    groups = df_train["Subject"]
    X_train = df_train.drop(columns=id_cols)
    X_test = df_test.drop(columns=id_cols)
    X_test_80 = df_test_80.drop(columns=id_cols)
    X_test_90 = df_test_90.drop(columns=id_cols)

    Y_train = df_train[targets]
    Y_test = df_test[targets]
    Y_test_80 = df_test_80[targets]
    Y_test_90 = df_test_90[targets]

    # build the model
    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)
    lgbm = build_lgbm(cfg)

    # Splittin sample wise
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=cfg.random_state, shuffle=True,) #stratify=X_train["Subject"]) # Cant stratify in the cleaned as dataset is quite imbalanced
    """
    counts = X_val["Subject"].value_counts()
    print(counts)
    """
    #-----------------------------------------
    # Just for diagnosing the model
    #-----------------------------------------
    #_, X_sub_train, _, Y_sub_train = train_test_split(X_train, Y_train, test_size=0.1, random_state=cfg.random_state, shuffle=True, stratify=X_train["Subject"])

    X_train = X_train.drop(columns=["Subject"])
    X_val = X_val.drop(columns=["Subject"])
    X_test = X_test.drop(columns=["Subject"])
    X_test_80 = X_test_80.drop(columns=["Subject"])
    X_test_90 = X_test_90.drop(columns=["Subject"])
    #X_sub_train = X_sub_train.drop(columns=["Subject"])

    # build the pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", lgbm)
    ])
    
    # =========================================================
    # Training and Eval
    #==========================================================
    for target in targets:
        Y_tr = Y_train[target]
        Y_v = Y_val[target]
        Y_t = Y_test[target]
        Y_t80 = Y_test_80[target]
        Y_t90 = Y_test_90[target]
        #Y_tr_sub = Y_sub_train[target]

        # Train
        lgbm.fit(X_train, Y_tr)                               

        # Pred
        Y_pred_tr = lgbm.predict(X_train)
        Y_val_pred = lgbm.predict(X_val)
        Y_pred = lgbm.predict(X_test)
        #Y_tr_sub_pred = lgbm.predict(X_sub_train)
        Y_t_80_pred = lgbm.predict(X_test_80)
        Y_t_90_pred = lgbm.predict(X_test_90)

        # Eval
        Path_res = PERFORMANCE_RESULTS_PAPER / r"Clean_80_DS"
        metrics_train = eval.evaluate(Y_tr, Y_pred_tr, 
                                             BA_path=Path_res / f"BA_train_subset_{target}.png",
                                             R2_path= Path_res / f"R2_train_subset_{target}.png")
        metrics_val = eval.evaluate(Y_v, Y_val_pred, 
                                             BA_path=Path_res / f"BA_val_{target}.png",
                                             R2_path= Path_res / f"R2_val_{target}.png")
        metrics_test_original = eval.evaluate(Y_t, Y_pred, 
                                              BA_path=Path_res / f"BA_test_original_{target}.png",
                                              R2_path= Path_res / f"R2_test_original_{target}.png")
        metrics_test_80 = eval.evaluate(Y_t80, Y_t_80_pred, 
                                              BA_path=Path_res / f"BA_test_80_{target}.png",
                                              R2_path= Path_res / f"R2_test_80_{target}.png")
        metrics_test_90 = eval.evaluate(Y_t90, Y_t_90_pred, 
                                              BA_path=Path_res / f"BA_test_90_{target}.png",
                                              R2_path= Path_res / f"R2_test_80_{target}.png")
        
        # Create dict for saving results
        results = {
            "Train": metrics_train,
            "Val": metrics_val,
            "Test_original": metrics_test_original,
            "Test_80": metrics_test_80,
            "Test_90": metrics_test_90
        }
        eval.save_results_dict(results, Path_res/ f"Results_{target}.csv", "Data_Subset")

        # Print results
        print(f"=== Results for {target}, in TRAIN ===")
        print(metrics_train, "\n")
        print(f"=== Results for {target}, in VAL ===")
        print(metrics_val, "\n")
        print(f"=== Results for {target}, in TEST ===")
        print(metrics_test_original, "\n")
        print(f"=== Results for {target}, in TEST at 90% cleaning ===")
        print(metrics_test_90, "\n")
        print(f"=== Results for {target}, in TEST at 80% cleaning===")
        print(metrics_test_80, "\n")
        