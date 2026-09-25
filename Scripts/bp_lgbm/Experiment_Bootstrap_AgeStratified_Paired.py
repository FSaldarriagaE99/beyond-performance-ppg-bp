############ BOOTSTRAP CI — AGE-STRATIFIED, PAIRED (SAME TEST SUBJECTS) ######
#                                                                             #
# Branch of Experiment_Bootstrap_AgeStratified.py. The original Age>=60 vs   #
# Full contrast bootstraps two UNEQUAL-sized, INDEPENDENT test pools (144    #
# full-test subjects vs 76 Age>=60 subjects) — see                           #
# Experiment_MDE_Equivalence_AgeStratified.py, which has to fall back to a   #
# Wald/quadrature approximation because of this.                             #
#                                                                             #
# Earlier attempt (superseded): downsample the full test set to N_TARGET     #
# subjects and pair the two arms by shared resample POSITION. Checked        #
# post-hoc that this does NOT induce real pairing — the two arms' subject-ID #
# arrays index different, almost-disjoint people, so matching by list        #
# position doesn't mean matching by patient. Correlation between arms came   #
# out ~0.                                                                    #
#                                                                             #
# This version fixes that by construction: BOTH arms are evaluated on the    #
# literal SAME 76 Age>=60 test subjects — no downsampled comparator cohort.  #
#                                                                             #
#   full_model : trained on the FULL (all-ages) training set, evaluated on   #
#                the Age>=60 test subjects.                                  #
#   age_gte60  : trained on the Age>=60-only training subset (unchanged from #
#                Experiment_Bootstrap_AgeStratified.py), evaluated on the    #
#                same Age>=60 test subjects.                                 #
#                                                                             #
# Because both arms share the identical test rows, one bootstrap resample    #
# list (subject-level, with replacement) is drawn ONCE and reused for both   #
# arms — same pattern as Experiment_Bootstrap_Filtering.py — making the      #
# comparison genuinely paired: row i of both Distribution_{target}.csv files #
# comes from the same resampled subjects.                                    #
#                                                                             #
# Reframes the question from "full random sample vs Age>=60, same N" to      #
# "does training on everyone hurt performance specifically on Age>=60        #
# patients, vs a model trained only on them."                                #
#                                                                             #
# Does NOT read or write anything under Bootstrap/ppg or                     #
# Bootstrap/age_stratified/ — fully self-contained, output root:             #
#   Bootstrap_AgeStratified_Paired/{full_model,age_gte60}/                   #
#     Distribution_{target}.csv, PointEstimates_{target}.csv, CI_{target}.csv#
###############################################################################

import pandas as pd
from tqdm import tqdm

import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, PERFORMANCE_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config
from models import build_lgbm
from sklearn.model_selection import train_test_split

N_RESAMPLES   = 1000
RANDOM_STATE  = 42
AGE_THRESHOLD = 60
TARGETS       = ["SBP", "DBP", "MAP"]

FEATURE_NAMES = [
    "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25",
    "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75",
    "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio",
    "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness",
    "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
    "FullSkew", "sdPRV", "IQR_PRV",
]


def train_ppg_models(df_train_stratum, cfg, random_state=RANDOM_STATE):
    """Train one PPG-only LightGBM per target on a (possibly age-filtered)
    training stratum. Mirrors the per-stratum training block in
    Experiment_Bootstrap_AgeStratified.py exactly."""
    X_train_full = df_train_stratum.drop(columns=["SF"] + TARGETS)
    Y_train_full = df_train_stratum[TARGETS]

    X_train, _, Y_train, _ = train_test_split(
        X_train_full, Y_train_full,
        test_size=0.1, random_state=random_state, shuffle=True,
        stratify=X_train_full["Subject"],
    )
    X_train = X_train.drop(columns=["Subject"])[FEATURE_NAMES]

    models = {}
    for target in TARGETS:
        lgbm = build_lgbm(cfg)
        lgbm.fit(X_train, Y_train[target])
        models[target] = lgbm
    return models


if __name__ == "__main__":

    # =========================================================
    # Paths
    # =========================================================
    train_path = PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5"
    test_path  = PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5"
    res_root   = PERFORMANCE_RESULTS_PAPER / "Bootstrap_AgeStratified_Paired"
    res_root.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # Load + impute
    # =========================================================
    df_train = load_PulseDB_sup_ds(train_path, feature_names=FEATURE_NAMES)
    df_test  = load_PulseDB_sup_ds(test_path,  feature_names=FEATURE_NAMES)

    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test  = preprocessing.median_impute_patientwise(df_test,  patient_col="Subject")

    grid_path = GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json"
    cfg = load_config(grid_path)

    # =========================================================
    # Age>=60 test stratum — the SHARED test set for both arms
    # =========================================================
    df_test_gte60 = df_test[df_test["Age"] >= AGE_THRESHOLD].copy()
    n_test_subjects = df_test_gte60["Subject"].nunique()
    print(f"Age>={AGE_THRESHOLD} test stratum (shared by both arms): "
          f"{n_test_subjects} subjects, {len(df_test_gte60)} signals")

    pd.DataFrame({"Subject": sorted(df_test_gte60["Subject"].unique())}).to_csv(
        res_root / "Test_Subjects.csv", index=False
    )

    strata = {
        "full_model": df_train.copy(),
        "age_gte60":  df_train[df_train["Age"] >= AGE_THRESHOLD].copy(),
    }

    # =========================================================
    # Train both arms, evaluate BOTH on df_test_gte60
    # =========================================================
    arm_models = {}

    for arm_name, df_train_s in strata.items():
        print(f"\n--- Arm: {arm_name} ---")
        print(f"  Train: {df_train_s['Subject'].nunique()} subjects, {len(df_train_s)} signals")
        print(f"  Test:  {n_test_subjects} subjects, {len(df_test_gte60)} signals (shared)")

        res_dir = res_root / arm_name
        res_dir.mkdir(parents=True, exist_ok=True)

        arm_models[arm_name] = train_ppg_models(df_train_s, cfg, random_state=RANDOM_STATE)

        for target in TARGETS:
            y_pred = arm_models[arm_name][target].predict(df_test_gte60[FEATURE_NAMES])
            pt = eval.compute_metrics(df_test_gte60[target].values, y_pred)
            pt_df = pd.DataFrame([{"metric": k, "value": v} for k, v in pt.items()])
            pt_df.to_csv(res_dir / f"PointEstimates_{target}.csv", index=False)

    # =========================================================
    # ONE shared bootstrap resample list — both arms evaluated on the same
    # test rows, so this is a genuinely paired design.
    # =========================================================
    resamples     = eval.allocate_bootstrap_resamples(
        df_test_gte60["Subject"], n_resamples=N_RESAMPLES, random_state=RANDOM_STATE
    )
    subject_index = eval.build_subject_index(df_test_gte60, subject_col="Subject")

    # =========================================================
    # Bootstrap loop
    # =========================================================
    boot_metrics = {arm: {t: [] for t in TARGETS} for arm in strata}

    for resample in tqdm(resamples, total=N_RESAMPLES, desc="paired bootstrap"):
        sample = eval.build_bootstrap_sample(df_test_gte60, resample, subject_index)
        for arm_name in strata:
            for target in TARGETS:
                y_true = sample[target].values
                y_pred = arm_models[arm_name][target].predict(sample[FEATURE_NAMES])
                boot_metrics[arm_name][target].append(eval.compute_metrics(y_true, y_pred))

    # =========================================================
    # Save distributions + CI
    # =========================================================
    for arm_name in strata:
        res_dir = res_root / arm_name
        for target in TARGETS:
            dist_df = pd.DataFrame(boot_metrics[arm_name][target])
            assert len(dist_df) == N_RESAMPLES
            dist_df.to_csv(res_dir / f"Distribution_{target}.csv", index=False)

            pt_df         = pd.read_csv(res_dir / f"PointEstimates_{target}.csv")
            point_metrics = dict(zip(pt_df["metric"], pt_df["value"]))

            ci_df = eval.compute_bootstrap_ci(dist_df, alpha=0.05)
            eval.save_bootstrap_ci(
                ci_df,
                path=res_dir / f"CI_{target}.csv",
                point_metrics=point_metrics,
            )

    print(f"\nDone. N_test={n_test_subjects} (shared, Age>={AGE_THRESHOLD}). Results under: {res_root}")
