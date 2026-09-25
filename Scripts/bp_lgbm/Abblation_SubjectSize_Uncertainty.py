############ ABLATION UNCERTAINTY: SUBJECT SIZE #########################
#                                                                        #
# Same subject-size ablation as Abblation_subject_size.py, but repeats   #
# each level's training draw N_REPEATS times to build a train-side       #
# uncertainty band, and bootstraps the test set (1,000 resamples,        #
# single seed) around the repeat that used the seed shared with the      #
# rest of the paper's experiments (42), for the test-side band.          #
#                                                                        #
# Seed separation:                                                       #
#   MODEL_SEED / BOOTSTRAP_SEED — fixed at 42 always.                    #
#   draw_seed  — the only thing that varies across repeats (which        #
#                subjects get selected; their segments follow them).     #
#                                                                        #
# No train/val split: val was never used (no early stopping, no          #
# reporting), so each repeat trains on the full drawn subset.            #
#                                                                        #
# Kept separate from Abblation_subject_size.py so that script stays      #
# untouched. Output = raw distributions only (no CI computed here).      #
###########################################################################

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import preprocessing
import eval
from local_paths import PULSE_DB_SUP_DIR, ABBLATION_RESULTS_PAPER, GS_RESULT_PAPER
from data import load_PulseDB_sup_ds
from config import load_config, ExperimentConfig, stress_test_params
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from models import build_lgbm

# =========================================================
# Fixed experiment seeds (kept separate on purpose)
# =========================================================
MODEL_SEED = 42       # LGBM random_state — fixed across every repeat
BOOTSTRAP_SEED = 42   # test-set bootstrap resample allocation — single seed only

N_REPEATS = 10
DRAW_SEEDS = [MODEL_SEED + i for i in range(N_REPEATS)]  # [42, 43, ..., 51]; 42 is repeat 0

N_RESAMPLES = 1000

TARGETS = ["SBP", "DBP", "MAP"]
FEATURE_NAMES = ["IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25", "Tsw50", "Tsw75", "Tdw25", "Tdw50",
                  "Tdw75", "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio", "Ad-Aa_ratio",
                  "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness", "L-H_ratio", "ShannonEntropy", "Tpp",
                  "PRV", "FullKurt", "FullSkew", "sdPRV", "IQR_PRV"]
ID_COLS = ["Age", "Gender", "Height", "Weight", "BMI", "SF"] + TARGETS

SUBJECT_COUNTS = [25, 50, 100, 200, 400, 600, 800, 1000, 1293]
FULL_N_SUBJECTS = 1293

# "generalisable" -> GS grid-search config (matches Subject_Size/ point-estimate results)
# "fit_biased"    -> stress_test_params (matches Subject_Size_Stress_Test/ point-estimate results)
# Override with the ABLATION_CONFIG_VARIANT env var to run both variants without editing this file.
CONFIG_VARIANT = os.environ.get("ABLATION_CONFIG_VARIANT", "generalisable")


def build_config():
    if CONFIG_VARIANT == "generalisable":
        cfg = load_config(GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json")
        cfg.random_state = MODEL_SEED
        subfolder = "Generalisable"
    elif CONFIG_VARIANT == "fit_biased":
        cfg = ExperimentConfig(n_splits=5, random_state=MODEL_SEED,
                                experiment_name="Subject_Size_Uncertainty_FitBiased",
                                verbose=-1, model_params=stress_test_params)
        subfolder = "FitBiased"
    else:
        raise ValueError(f"Unknown CONFIG_VARIANT: {CONFIG_VARIANT!r}")
    return cfg, subfolder


def run_level(df_train, df_test, n_subjects, cfg, base_path,
              bootstrap_resamples, bootstrap_subject_index):
    n_repeats = 1 if n_subjects == FULL_N_SUBJECTS else N_REPEATS
    draw_seeds = DRAW_SEEDS[:n_repeats]

    level_path = base_path / f"Subject_Size_{n_subjects}"
    level_path.mkdir(parents=True, exist_ok=True)

    rows = []
    canonical_pipelines = {}

    for repeat_idx, draw_seed in enumerate(draw_seeds):
        chosen_subjects = preprocessing.select_subjects_by_target_distribution(
            df_train, subject_col="Subject", target_col="MAP",
            n_subjects=n_subjects, n_bins=12, random_state=draw_seed
        )
        df_train_draw = df_train[df_train["Subject"].isin(chosen_subjects)].copy()

        X_draw = df_train_draw.drop(columns=ID_COLS)
        Y_tr = df_train_draw[TARGETS]

        n_train_rows = len(X_draw)
        n_train_subjects = int(X_draw["Subject"].nunique())
        X_tr = X_draw.drop(columns=["Subject"])

        is_canonical = (draw_seed == MODEL_SEED)

        for target in TARGETS:
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", build_lgbm(cfg)),
            ])
            pipeline.fit(X_tr, Y_tr[target])

            if is_canonical:
                canonical_pipelines[target] = pipeline

            common = {
                "repeat_idx": repeat_idx, "draw_seed": draw_seed, "target": target,
                "level": n_subjects, "n_train_rows": n_train_rows,
                "n_train_subjects": n_train_subjects, "used_for_bootstrap": is_canonical,
            }

            y_tr_pred = pipeline.predict(X_tr)
            train_metrics = eval.compute_metrics(Y_tr[target].values, y_tr_pred)
            rows.append({**train_metrics, **common, "subset": "Train"})

            X_test = df_test[FEATURE_NAMES]
            y_test_pred = pipeline.predict(X_test)
            test_metrics = eval.compute_metrics(df_test[target].values, y_test_pred)
            rows.append({**test_metrics, **common, "subset": "Test"})

    dist_path = level_path / "TrainDraws_Distribution.csv"
    pd.DataFrame(rows).to_csv(dist_path, index=False)
    print(f"  Saved: {dist_path}  ({len(draw_seeds)} repeats)")

    # =========================================================
    # Test-set bootstrap — canonical (seed=42) draw only.
    # A resample only changes which rows (with repeats) are pulled into
    # each pseudo-sample; the model's per-row prediction never changes.
    # So predict once per target on the full test set, then just gather
    # the (y_true, y_pred) pairs at the resampled indices per resample
    # instead of re-predicting 1,000x (which was the entire bottleneck).
    # =========================================================
    y_pred_full = {
        target: pd.Series(canonical_pipelines[target].predict(df_test[FEATURE_NAMES]),
                           index=df_test.index)
        for target in TARGETS
    }

    boot_metrics = {t: [] for t in TARGETS}
    for resample in tqdm(bootstrap_resamples, total=N_RESAMPLES,
                          desc=f"  bootstrap n={n_subjects}"):
        idx = np.concatenate([bootstrap_subject_index[s] for s in resample])
        for target in TARGETS:
            y_pred = y_pred_full[target].loc[idx].values
            y_true = df_test[target].loc[idx].values
            boot_metrics[target].append(eval.compute_metrics(y_true, y_pred))

    for target in TARGETS:
        boot_path = level_path / f"TestBootstrap_Distribution_{target}.csv"
        pd.DataFrame(boot_metrics[target]).to_csv(boot_path, index=False)
        print(f"  Saved: {boot_path}")


if __name__ == "__main__":
    df_train = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5", feature_names=FEATURE_NAMES)
    df_test = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5", feature_names=FEATURE_NAMES)

    df_train = preprocessing.median_impute_patientwise(df_train, "Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, "Subject")

    cfg, subfolder = build_config()
    base_path = ABBLATION_RESULTS_PAPER / "Subject_Size_Uncertainty" / subfolder
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"Pre-allocating {N_RESAMPLES} test-set bootstrap resamples (seed={BOOTSTRAP_SEED})...")
    bootstrap_resamples = eval.allocate_bootstrap_resamples(
        df_test["Subject"], n_resamples=N_RESAMPLES, random_state=BOOTSTRAP_SEED
    )
    bootstrap_subject_index = eval.build_subject_index(df_test, subject_col="Subject")

    for n in SUBJECT_COUNTS:
        print(f"\n=== Subject count = {n} ({CONFIG_VARIANT}) ===")
        run_level(df_train, df_test, n, cfg, base_path, bootstrap_resamples, bootstrap_subject_index)

    print("\nDone. Distributions saved under:", base_path)
