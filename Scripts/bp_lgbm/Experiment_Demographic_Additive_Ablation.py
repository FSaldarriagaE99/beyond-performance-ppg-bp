############ INCREMENTAL DEMOGRAPHIC ABLATION (R1#2, R2#4) ###########################
#                                                                                    #
# Adds demographics cumulatively (Age -> Sex -> BMI -> Height -> Weight) and tracks  #
# train / test R2 at every step.                                                     #
#                                                                                    #
#   Arm A - demographics only : LightGBM (generalisable), LightGBM (fit-biased), OLS #
#   Arm B - 28 PPG features + the same demographic sequence (both LightGBM configs)  #
#           Step 0 = PPG only (sanity check against the manuscript values)           #
#                                                                                    #
# Everything mirrors the manuscript pipeline (Experiment_PPG_Demographics.py,        #
# Stress_test.py, LGBM_Demographic_Baseline_Model.py, Demographic_Baseline_Model.py):#
# same subject-disjoint train/test files, same stratified 90/10 sample-wise fit      #
# split (seed 42; models are fit on the 90% and "train R2" is measured on that       #
# FULL 90% fit set, never a subsample), same hyperparameters (no tuning), Gender as  #
# a pandas Categorical for LightGBM and one-hot for OLS, no re-split.                #
#                                                                                    #
# Column order (LightGBM feature_fraction subsampling is column-order sensitive):    #
# demographics FIRST in manuscript order (Age, Gender, Height, Weight, BMI)          #
# restricted to the active variables, THEN the 28 PPG features, so the final         #
# step is column-for-column identical to the Table 5 model. See                      #
# Sensitivity_ColumnOrder_Demographic_Ablation.py for how much this choice matters.  #
#                                                                                    #
# Fit-biased LightGBM models run inside the Stress_test.py scaled pipeline           #
# (z-scored numeric columns, Gender NOT scaled). Generalisable LightGBM is unscaled, #
# as in Experiment_PPG_Demographics.py.                                              #
#                                                                                    #
# Test R2 95% CI: subject-level bootstrap, 1,000 resamples, seed 42, subjects as     #
# clusters (eval.allocate_bootstrap_resamples, the same call the manuscript uses),   #
# percentile CI (eval.compute_bootstrap_ci). Predict once, then gather per resample. #
#                                                                                    #
# Uniqueness % = share of fit-set subjects whose demographic vector (active          #
# variables at that step, subject's most frequent vector if it changes across        #
# sessions) is shared with no other fit-set subject.                                 #
#                                                                                    #
# Outputs -> ABBLATION_RESULTS_PAPER.parent / "demographic_additive_ablation":       #
#   Demographic_Additive_Ablation.csv, Fig_Arm{A,B}_*.png/.pdf, run_log.txt          #
#                                                                                    #
# Usage (from Scripts/bp_lgbm):                                                      #
#   python Experiment_Demographic_Additive_Ablation.py              full run         #
#   python Experiment_Demographic_Additive_Ablation.py --smoke      tiny wiring test #
#   python Experiment_Demographic_Additive_Ablation.py --figures-only  redraw plots  #
#########################################################################################

import argparse
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import eval
import preprocessing
from config import ExperimentConfig, load_config, stress_test_params
from data import load_PulseDB_sup_ds
from local_paths import (ABBLATION_RESULTS_PAPER, GS_RESULT_PAPER,
                         PERFORMANCE_RESULTS_PAPER, PULSE_DB_SUP_DIR)
from models import build_lgbm

np.random.seed(42)

# =========================================================
# Constants
# =========================================================
SEED = 42              # model, fit split and bootstrap allocation (all the same, as in the manuscript)
N_RESAMPLES = 1000
ALPHA = 0.05
TARGETS = ["SBP", "DBP", "MAP"]

FEATURE_NAMES = [
    "IPR", "Tsp", "TWRRF25", "TWRRF50", "Tsw25",
    "Tsw50", "Tsw75", "Tdw25", "Tdw50", "Tdw75",
    "AUCpi", "IPA", "Av-Au_ratio", "Ab-Aa_ratio", "Ac-Aa_ratio",
    "Ad-Aa_ratio", "Ap2-Ap1_ratio", "AGI", "Kurtosis", "Skewness",
    "L-H_ratio", "ShannonEntropy", "Tpp", "PRV", "FullKurt",
    "FullSkew", "sdPRV", "IQR_PRV",
]
DEMO_CANONICAL = ["Age", "Gender", "Height", "Weight", "BMI"]   # manuscript column order
ADDITION_ORDER = ["Age", "Gender", "BMI", "Height", "Weight"]   # requested cumulative order
STEP_LABELS = {0: "PPG only", 1: "Age", 2: "+Sex", 3: "+BMI", 4: "+Height", 5: "+Weight"}

R2_TOL = 1e-5          # "floating-point noise" tolerance for the manuscript reproduction checks

CSV_NAME = "Demographic_Additive_Ablation.csv"
LOG_NAME = "run_log.txt"

log = logging.getLogger("demog_additive_ablation")


# =========================================================
# Helpers
# =========================================================
def out_dir_for(smoke: bool) -> Path:
    base = ABBLATION_RESULTS_PAPER.parent / "demographic_additive_ablation"
    return base / "_smoke" if smoke else base


def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(out_dir / LOG_NAME, mode="w", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)


def active_demographics(step: int) -> list:
    """Demographic columns active at `step`, in manuscript column order."""
    active = set(ADDITION_ORDER[:step])
    return [c for c in DEMO_CANONICAL if c in active]


def unique_demographic_pct(demo_fit: pd.DataFrame, cols: list) -> float:
    """% of subjects whose (modal) demographic vector over `cols` is not shared by any other subject."""
    counts = demo_fit.groupby(["Subject"] + cols, observed=True).size().reset_index(name="n")
    modal = (counts.sort_values(["Subject", "n"], ascending=[True, False], kind="stable")
                   .drop_duplicates("Subject"))
    shared = modal.groupby(cols, observed=True)["Subject"].transform("size")
    return float(100.0 * (shared == 1).mean())


def bootstrap_r2(y_true: np.ndarray, y_pred: np.ndarray, resamples, subject_index) -> np.ndarray:
    """Subject-cluster bootstrap of R2. Index labels == positions (RangeIndex asserted by caller)."""
    out = np.empty(len(resamples))
    for i, resample in enumerate(resamples):
        idx = np.concatenate([subject_index[s] for s in resample])
        out[i] = r2_score(y_true[idx], y_pred[idx])
    return out


def make_lgbm_cfg(variant: str, smoke: bool):
    if variant == "generalisable":
        cfg = load_config(GS_RESULT_PAPER / "Full_Grid_Randomized_search_3targets.json")
        cfg.random_state = SEED
    elif variant == "fit_biased":
        cfg = ExperimentConfig(n_splits=5, random_state=SEED, experiment_name="Demog_Additive_FitBiased",
                               verbose=-1, model_params=dict(stress_test_params))
    else:
        raise ValueError(variant)
    if smoke:
        cfg.model_params = {**cfg.model_params, "n_estimators": 20}
    return cfg


def make_ols(cols: list) -> Pipeline:
    """Same implementation as Demographic_Baseline_Model.py, restricted to the active columns."""
    transformers = []
    num = [c for c in cols if c != "Gender"]
    if num:
        transformers.append(("num", StandardScaler(), num))
    if "Gender" in cols:
        transformers.append(("cat", OneHotEncoder(drop="first"), ["Gender"]))
    return Pipeline([("preprocessor", ColumnTransformer(transformers)),
                     ("regressor", LinearRegression())])


def make_estimator(model: str, cols: list, cfgs: dict):
    """One estimator per (model, active columns); shared with the sensitivity script."""
    if model == "OLS":
        return make_ols(cols)
    if model == "LGBM_fit_biased":
        # Same scaled pipeline as Stress_test.py (Fig 8). Sex (Gender) is deliberately
        # EXCLUDED from scaling: it is a pandas Categorical that LightGBM handles
        # natively, and z-scoring its category codes would be meaningless and would
        # drop the categorical dtype. Every other column (PPG features, Age, Height,
        # Weight, BMI) is z-scored; on PPG-only this is exactly StandardScaler.
        return Pipeline([("scaler", preprocessing.NumericOnlyScaler()),
                         ("model", build_lgbm(cfgs["fit_biased"]))])
    return build_lgbm(cfgs["generalisable"])   # generalisable: unscaled, as in the manuscript


# =========================================================
# Experiment
# =========================================================
def run_experiment(out_dir: Path, smoke: bool):
    t_start = time.time()
    n_resamples = 20 if smoke else N_RESAMPLES

    log.info("=== Incremental demographic ablation (R1#2, R2#4) ===")
    if smoke:
        log.info("SMOKE MODE: n_estimators=20, %d bootstrap resamples, manuscript checks skipped", n_resamples)
    log.info("python %s | numpy %s | pandas %s", platform.python_version(), np.__version__, pd.__version__)
    import lightgbm, sklearn
    log.info("lightgbm %s | scikit-learn %s | CPU only", lightgbm.__version__, sklearn.__version__)
    log.info("seeds: global numpy=%d, LightGBM random_state=%d, fit split=%d, bootstrap=%d",
             SEED, SEED, SEED, SEED)
    log.info("output dir: %s", out_dir)

    # ---- Load (manuscript preprocessing) ----
    df_train = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_Train_Subset.h5", feature_names=FEATURE_NAMES)
    df_test = load_PulseDB_sup_ds(PULSE_DB_SUP_DIR / "Features_VitalDB_CalFree_Test_Subset.h5", feature_names=FEATURE_NAMES)
    df_train = preprocessing.median_impute_patientwise(df_train, patient_col="Subject")
    df_test = preprocessing.median_impute_patientwise(df_test, patient_col="Subject")
    log.info("train: %d rows / %d subjects | test: %d rows / %d subjects",
             len(df_train), df_train["Subject"].nunique(), len(df_test), df_test["Subject"].nunique())

    # Raw (object) Gender kept for OLS and uniqueness; Categorical for LightGBM, as in the manuscript.
    df_train["Gender_raw"] = df_train["Gender"]
    df_test["Gender_raw"] = df_test["Gender"]
    df_train["Gender"] = df_train["Gender"].astype("category")
    df_test["Gender"] = df_test["Gender"].astype("category")

    # ---- Fit split (identical to Experiment_Bootstrap.py) ----
    X_full = df_train.drop(columns=["SF"] + TARGETS)
    Y_full = df_train[TARGETS]
    X_fit, _, Y_fit, _ = train_test_split(X_full, Y_full, test_size=0.1, random_state=SEED,
                                          shuffle=True, stratify=X_full["Subject"])
    subj_fit = X_fit["Subject"]
    log.info("fit split: %d rows / %d subjects (10%% sample-wise hold-out discarded, as in the manuscript)",
             len(X_fit), subj_fit.nunique())

    # ---- Bootstrap allocation (once, reused everywhere) ----
    assert isinstance(df_test.index, pd.RangeIndex), "bootstrap gathers by position; expected RangeIndex"
    resamples = eval.allocate_bootstrap_resamples(df_test["Subject"], n_resamples=n_resamples, random_state=SEED)
    subject_index = eval.build_subject_index(df_test, subject_col="Subject")
    y_test = {t: df_test[t].to_numpy() for t in TARGETS}
    y_fit = {t: Y_fit[t].to_numpy() for t in TARGETS}

    # ---- Uniqueness per step ----
    demo_fit = X_fit[["Subject", "Gender_raw", "Age", "Height", "Weight", "BMI"]].rename(columns={"Gender_raw": "Gender"})
    uniq = {s: unique_demographic_pct(demo_fit, active_demographics(s)) if s > 0 else np.nan for s in range(6)}
    log.info("uniqueness %% by step: %s", {STEP_LABELS[s]: round(v, 2) for s, v in uniq.items() if s > 0})

    # ---- Run grid ----
    cfgs = {v: make_lgbm_cfg(v, smoke) for v in ("generalisable", "fit_biased")}
    log.info("generalisable params: %s", cfgs["generalisable"].model_params)
    log.info("fit-biased params:    %s", cfgs["fit_biased"].model_params)

    plan = []
    for model in ("LGBM_generalisable", "LGBM_fit_biased", "OLS"):
        plan += [("A", model, s) for s in range(1, 6)]
    for model in ("LGBM_generalisable", "LGBM_fit_biased"):
        plan += [("B", model, s) for s in range(0, 6)]

    rows = []
    csv_path = out_dir / CSV_NAME
    for arm, model, step in plan:
        demo_cols = active_demographics(step)
        cols = demo_cols + (FEATURE_NAMES if arm == "B" else [])   # demographics first, then PPG
        if model == "OLS":
            Xtr = X_fit[cols].assign(**({"Gender": X_fit["Gender_raw"]} if "Gender" in cols else {}))
            Xte = df_test[cols].assign(**({"Gender": df_test["Gender_raw"]} if "Gender" in cols else {}))
        else:
            Xtr, Xte = X_fit[cols], df_test[cols]

        t0 = time.time()
        for target in TARGETS:
            t1 = time.time()
            est = make_estimator(model, cols, cfgs)
            est.fit(Xtr, y_fit[target])
            fit_s = time.time() - t1
            p_tr, p_te = est.predict(Xtr), est.predict(Xte)

            boots = bootstrap_r2(y_test[target], p_te, resamples, subject_index)
            ci = eval.compute_bootstrap_ci(pd.DataFrame({"R2": boots}), alpha=ALPHA).iloc[0]
            rows.append({
                "arm": arm, "model": model, "step": step, "step_label": STEP_LABELS[step] if arm == "B" or step > 0 else "",
                "demographics": "+".join(demo_cols), "n_features": len(cols), "target": target,
                "train_r2": float(r2_score(y_fit[target], p_tr)),
                "test_r2": float(r2_score(y_test[target], p_te)),
                "test_r2_ci_lower": float(ci["ci_lower"]), "test_r2_ci_upper": float(ci["ci_upper"]),
                "test_r2_boot_mean": float(ci["bootstrap_mean"]), "test_r2_boot_sd": float(ci["bootstrap_std"]),
                "n_resamples": n_resamples, "unique_demographic_pct": uniq[step],
                "n_fit_subjects": int(subj_fit.nunique()), "n_fit_rows": len(X_fit),
                "fit_seconds": round(fit_s, 2),
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False)   # progressive save
        last = [r for r in rows[-3:]]
        log.info("Arm %s | %-18s | step %d (%s) | %.1fs | test R2 SBP/DBP/MAP = %s | train R2 = %s",
                 arm, model, step, "+".join(demo_cols) or "PPG only", time.time() - t0,
                 "/".join(f"{r['test_r2']:.4f}" for r in last), "/".join(f"{r['train_r2']:.4f}" for r in last))

    res = pd.DataFrame(rows)
    res.to_csv(csv_path, index=False)
    log.info("saved %s (%d rows)", csv_path, len(res))

    # ---- Sanity checks ----
    if smoke:
        log.info("sanity checks vs manuscript skipped in smoke mode")
    else:
        run_sanity_checks(res, uniq)
    log.info("total runtime: %.1f min", (time.time() - t_start) / 60)
    return res


# =========================================================
# Sanity checks
# =========================================================
# Reference = the result files whose numbers are PRINTED in the paper:
#   Original_DS                 PPG-only, generalisable      (Experiment_eval.py)
#   Stress_Test                 PPG-only, fit-biased, Fig 8   (Stress_test.py)
#   LGBM_Demographic_Baseline   demographics-only, generalisable (LGBM_Demographic_Baseline_Model.py)
#   Original_PPG_Demographics   PPG + demographics, generalisable, Table 5 (Experiment_PPG_Demographics.py)
#   Demographic_Baseline        demographics-only OLS        (Demographic_Baseline_Model.py)
# plus a cross-check against the regenerated Bootstrap/ and Bootstrap_StressTest/ point estimates.
PAPER_REFS = [  # (description, arm, model, step, results folder, Data_Subset rows to compare)
    ("Arm B step 0, generalisable vs Original_DS", "B", "LGBM_generalisable", 0, "Original_DS", ["Test_original"]),
    ("Arm B step 0, fit-biased vs Stress_Test (Fig 8)", "B", "LGBM_fit_biased", 0, "Stress_Test", ["Test_original", "Train"]),
    ("Arm A step 5, generalisable vs LGBM_Demographic_Baseline", "A", "LGBM_generalisable", 5, "LGBM_Demographic_Baseline", ["Test_original", "Train"]),
    ("Arm A step 5, OLS vs Demographic_Baseline", "A", "OLS", 5, "Demographic_Baseline", ["Test_original", "Train"]),
    ("Arm B step 5, generalisable vs Original_PPG_Demographics (Table 5)", "B", "LGBM_generalisable", 5, "Original_PPG_Demographics", ["Test_original", "Train"]),
]
SUBSET_COL = {"Test_original": "test_r2", "Train": "train_r2"}
BOOTSTRAP_CROSS = [  # (arm, step, Bootstrap* sub-folder)
    ("B", 0, "ppg"), ("A", 5, "demo"), ("B", 5, "ppg_demo"),
]


def _check(name: str, ours: float, ref: float, tol: float = R2_TOL) -> bool:
    ok = abs(ours - ref) <= tol
    log.info("  [%s] %-30s ours=%.8f reference=%.8f |diff|=%.2e", "PASS" if ok else "FAIL", name, ours, ref, abs(ours - ref))
    return ok


def run_sanity_checks(res: pd.DataFrame, uniq: dict):
    log.info("--- Sanity checks (tolerance |dR2| <= %.0e) ---", R2_TOL)
    results = []

    def ours(arm, model, step, target, col="test_r2"):
        m = res[(res.arm == arm) & (res.model == model) & (res.step == step) & (res.target == target)]
        return float(m[col].iloc[0])

    log.info("1) Against the files whose numbers are printed in the paper")
    for desc, arm, model, step, folder, rows in PAPER_REFS:
        log.info(" %s", desc)
        for t in TARGETS:
            ref_path = PERFORMANCE_RESULTS_PAPER / folder / f"Results_{t}.csv"
            if not ref_path.exists():
                log.info("  skipped: %s not found (run the script that writes %s first)", ref_path.name, folder)
                continue
            ref = pd.read_csv(ref_path).set_index("Data_Subset")["R2"]
            for row in rows:
                results.append(_check(f"{t} {row}", ours(arm, model, step, t, SUBSET_COL[row]), float(ref[row])))

    log.info("2) Cross-check against regenerated Experiment_Bootstrap*.py point estimates (test R2)")
    for model, folder in {"LGBM_generalisable": "Bootstrap", "LGBM_fit_biased": "Bootstrap_StressTest"}.items():
        for arm, step, sub_dir in BOOTSTRAP_CROSS:
            log.info(" %s, %s (step %d) vs %s/%s", model, "PPG only" if sub_dir == "ppg" else sub_dir, step, folder, sub_dir)
            for t in TARGETS:
                pe_path = PERFORMANCE_RESULTS_PAPER / folder / sub_dir / f"PointEstimates_{t}.csv"
                if not pe_path.exists():
                    log.info("  skipped: %s/%s not found (run the bootstrap script first)", folder, sub_dir)
                    continue
                pe = pd.read_csv(pe_path)
                results.append(_check(t, ours(arm, model, step, t), float(pe.loc[pe["metric"] == "R2", "value"].iloc[0])))

    log.info("3) Uniqueness %% step 4 -> 5 (Height -> +Weight): %.4f -> %.4f, change = %+.4f pp (expected ~0)",
             uniq[4], uniq[5], uniq[5] - uniq[4])
    log.info("--- %d/%d R2 checks passed ---", sum(results), len(results))


# =========================================================
# Figures
# =========================================================
MODEL_STYLE = {   # project palette (Fig 6: grey = generalisable, green = fit-biased; navy = baseline)
    "LGBM_generalisable": ("#7F7F7F", "LightGBM generalisable", "o"),
    "LGBM_fit_biased": ("#27AE60", "LightGBM fit-biased", "s"),
    "OLS": ("#1F4E79", "OLS (manuscript baseline)", "^"),
}


def make_figures(res: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    sys.path.insert(0, str(Path(__file__).resolve().parent / "figures"))
    from fig_style import DPI, FONT_FAMILY, W_FULL, apply_base_style, save_fig
    plt.rcParams["font.family"] = FONT_FAMILY   # same font as the other manuscript figures

    titles = {"A": "Incremental demographic ablation: demographics only",
              "B": "Incremental demographic ablation: PPG features + demographics"}
    stems = {"A": "Fig_ArmA_Demographics_Only", "B": "Fig_ArmB_PPG_Plus_Demographics"}

    for arm in ("A", "B"):
        d = res[res.arm == arm]
        if d.empty:
            continue
        steps = sorted(d.step.unique())
        xs = np.arange(len(steps))
        fig, axes = plt.subplots(1, 3, figsize=(W_FULL, W_FULL * 0.42), sharey=True)
        twins = []
        for ax, target in zip(axes, TARGETS):
            apply_base_style(ax)
            tw = ax.twinx()
            twins.append(tw)
            u = d[(d.target == target)].drop_duplicates("step").set_index("step").loc[steps, "unique_demographic_pct"]
            tw.bar(xs, u.fillna(0).to_numpy(), width=0.55, color="#D9D9D9", alpha=0.4, zorder=1, linewidth=0)
            for x, v in zip(xs, u.to_numpy()):   # value label at the top of each bar (none where undefined)
                if not np.isnan(v):
                    tw.text(x, v + 1.5, f"{v:.1f}%" if v < 99.95 else "100%", ha="center", va="bottom",
                            fontsize=5.5, color="#8A8A8A", zorder=1,
                            bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.6))
            tw.set_ylim(0, 108)
            tw.set_yticks([0, 20, 40, 60, 80, 100])
            tw.spines[["top"]].set_visible(False)
            tw.spines["right"].set_linewidth(0.7)
            tw.tick_params(labelsize=7, colors="#8A8A8A")
            ax.set_zorder(tw.get_zorder() + 1)   # lines above the bars
            ax.patch.set_visible(False)

            for model, (color, _, marker) in MODEL_STYLE.items():
                m = d[(d.target == target) & (d.model == model)].set_index("step").reindex(steps)
                if m["test_r2"].isna().all():
                    continue
                ax.plot(xs, 100 * m["train_r2"], color=color, marker=marker, ms=4, lw=1.2, ls="-", zorder=3)
                ax.fill_between(xs, 100 * m["test_r2_ci_lower"], 100 * m["test_r2_ci_upper"],
                                color=color, alpha=0.18, lw=0, zorder=2)
                ax.plot(xs, 100 * m["test_r2"], color=color, marker=marker, ms=4, lw=1.2, ls="--",
                        mfc="white", mew=1.0, zorder=4)
            ax.axhline(0, color="black", lw=0.5, zorder=2)
            ax.set_title(target, fontsize=9)
            ax.set_xticks(xs)
            ax.set_xticklabels([STEP_LABELS[s] for s in steps], rotation=35, ha="right", fontsize=7)
            if target != "MAP":
                tw.set_yticklabels([])
                tw.tick_params(right=False)
                tw.spines["right"].set_visible(False)
        lo, hi = axes[0].get_ylim()   # after all panels are drawn (shared y)
        axes[0].set_ylim(lo, hi + 0.2 * (hi - lo))   # headroom so curves clear the uniqueness bar labels
        axes[0].set_ylabel("R$^2$ (%)", fontsize=8)
        twins[-1].set_ylabel("Training subjects with a unique\ndemographic vector (%)", fontsize=7, color="#8A8A8A")
        present = [m for m in MODEL_STYLE if (d.model == m).any()]
        handles = [Line2D([0], [0], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][2], ms=4, lw=1.2) for m in present]
        labels = [MODEL_STYLE[m][1] for m in present]
        handles += [Line2D([0], [0], color="black", lw=1.2, ls="-"),
                    Line2D([0], [0], color="black", lw=1.2, ls="--", marker="o", ms=4, mfc="white"),
                    Patch(facecolor="grey", alpha=0.25),
                    Patch(facecolor="#D9D9D9", alpha=0.4)]
        labels += ["Train", "Test (shaded: 95% subject-bootstrap CI)", "", "Unique demographic vector (%, right axis)"]
        keep = [i for i, l in enumerate(labels) if l]
        fig.legend([handles[i] for i in keep], [labels[i] for i in keep], loc="lower center",
                   ncol=3, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.12))
        fig.suptitle(titles[arm], fontsize=9, y=1.0)
        fig.tight_layout()
        save_fig(fig, stems[arm], out_dir, dpi=DPI)
        plt.close(fig)
        log.info("saved figure %s", out_dir / f"{stems[arm]}.png")


# =========================================================
# Entrypoint
# =========================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny wiring test (20 trees, 20 resamples) into a _smoke subfolder")
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from the existing CSV")
    args = ap.parse_args()

    out_dir = out_dir_for(args.smoke)
    if args.figures_only:
        out_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        make_figures(pd.read_csv(out_dir / CSV_NAME), out_dir)
    else:
        setup_logging(out_dir)
        results = run_experiment(out_dir, args.smoke)
        make_figures(results, out_dir)
        log.info("Done.")
