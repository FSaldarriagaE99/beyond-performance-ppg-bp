######## CONFIG SCRIPT ###############################################################
#                                                                                    #
# This script handles the model and the experiment config scripts                    #
# of the different tree based models that I am using.                                #
#                                                                                    #
######################################################################################

# ----------------------------
# Experiment Config
# ----------------------------
import json
from dataclasses import dataclass, asdict, field, is_dataclass
from pathlib import Path

# ----------------------------
# Experiment Configs
# ----------------------------
@dataclass # Decorator only to create a class that holds data. (this creates the builder and so on automatically)
class ExperimentConfig:
    random_state: int = 42 #
    n_splits: int = 5
    experiment_name: str = "baseline"
    verbose:int = -1
    model_params: dict = None
    n_jobs:int = -1

# ----------------------------
# Models Configs
# ----------------------------
lightGBM_default_params = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "bagging_fraction": 0.9,   # instead of subsample
    "feature_fraction": 0.9,   # instead of colsample_bytree
    "lambda_l1": 0.0,          # instead of reg_alpha
    "lambda_l2": 0.0,          # instead of reg_lambda
}

lightGBM_best_guess_1 = {
    # --- Core capacity ---
    "num_leaves": 31,        # modest complexity, avoids overfitting
    "max_depth": -1,         # let leaves control depth

    # --- Learning dynamics ---
    "learning_rate": 0.05,   # stable but not too slow
    "n_estimators": 1000,    # complements low LR

    # --- Regularization ---
    "min_data_in_leaf": 50,  # avoids tiny, overfit leaves
    "lambda_l1": 0.0,        # usually not needed
    "lambda_l2": 1.0,        # light smoothing, helps generalization

    # --- Subsampling ---
    "feature_fraction": 0.8, # use 80% of features per tree
    "bagging_fraction": 0.8, # use 80% of samples per tree
    "bagging_freq": 1,       # resample every iteration
}

stress_test_params = { #Empirically derived from the GS subject wise done. Selected the one with highest train score (lowest test score --> overfitted)
    'num_leaves': 63, 
    'n_estimators': 1000, 
    'min_gain_to_split': 0.1, 
    'min_data_in_leaf': 50, 
    'max_depth': -1, 
    'learning_rate': 0.2, 
    'lambda_l2': 1, 
    'lambda_l1': 0, 
    'feature_fraction': 0.7, 
    'bagging_freq': 1, 
    'bagging_fraction': 0.8
}

def save_config(cfg, filepath: str):
    """
    Save the current ExperimentConfig as a JSON file.
    If model_params contains 'estimator__' keys, strip the prefix.
    Args:
        cfg: ExperimentConfig object
        filepath: where to save (e.g., "configs/baseline_tuned.json")
    """
    # Convert dataclass to dict if needed
    data = asdict(cfg) if is_dataclass(cfg) else dict(cfg)

    # Clean up model_params
    if "model_params" in data:
        cleaned_params = {}
        for k, v in data["model_params"].items():
            if k.startswith("estimator__"):
                new_key = k.replace("estimator__", "")
                cleaned_params[new_key] = v
            else:
                cleaned_params[k] = v
        data["model_params"] = cleaned_params

    # Ensure parent directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Config saved to {filepath}")


def load_config(filepath: str) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a JSON file.

    Args:
        filepath: path to JSON file

    Returns:
        ExperimentConfig object
    """
    with open(filepath, "r") as f:
        data = json.load(f)
    return ExperimentConfig(**data)

# ----------------------------
# Param grids for GridSearch
# ----------------------------

# Small grid: quick runs
lightGBM_small_grid = {
    "model__num_leaves": [31, 63],
    "model__learning_rate": [0.05, 0.1],
    "model__n_estimators": [200, 500],
}

# Baseline grid: balanced
lightGBM_baseline_grid = {
    "model__num_leaves": [31, 63, 127],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [500, 1000],
    "model__subsample": [0.8, 0.9],
    "model__colsample_bytree": [0.8, 0.9],
}

# Large grid: exhaustive search (slower)
lightGBM_large_grid = {
    "model__num_leaves": [31, 63, 127, 255],
    "model__learning_rate": [0.01, 0.05, 0.1],
    "model__n_estimators": [500, 1000, 2000],
    "model__subsample": [0.7, 0.8, 0.9],
    "model__colsample_bytree": [0.7, 0.8, 0.9],
    "model__reg_alpha": [0.0, 0.1, 0.5],
    "model__reg_lambda": [0.0, 0.1, 0.5],
}

# param_grid for GridSearchCV with LightGBM
lightGBM_full_grid = {
    "num_leaves": [15, 31, 63],                # Controls model complexity
    "max_depth": [-1, 5, 10],                  # -1 = unlimited, but shallow depths add regularization
    "learning_rate": [0.01, 0.05, 0.1, 0.2],   # Lower = stable, higher = faster but riskier
    "n_estimators": [100, 500, 1000],          # Number of boosting trees
    "min_data_in_leaf": [20, 50, 100],         # Prevents overfitting to small groups
    "lambda_l1": [0, 0.1, 1, 10],              # L1 regularization
    "lambda_l2": [0, 0.1, 1, 10],              # L2 regularization
    "min_gain_to_split": [0.0, 0.1, 0.5],      # Minimum loss reduction for splits
    "feature_fraction": [0.7, 0.8, 0.9, 1.0],  # Random subset of features per tree
    "bagging_fraction": [0.7, 0.8, 0.9, 1.0],  # Random subset of data per tree
    "bagging_freq": [1],                       # How often to resample (1 = every iteration)
}

# param_grid for GridSearchCV with LightGBM --> Multioutput regressor
lightGBM_full_grid_3target = {
    "model__estimator__num_leaves": [15, 31, 63],
    "model__estimator__max_depth": [-1, 5, 10],
    "model__estimator__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__estimator__n_estimators": [100, 500, 1000],
    "model__estimator__min_data_in_leaf": [20, 50, 100],
    "model__estimator__lambda_l1": [0, 0.1, 1, 10],
    "model__estimator__lambda_l2": [0, 0.1, 1, 10],
    "model__estimator__min_gain_to_split": [0.0, 0.1, 0.5],
    "model__estimator__feature_fraction": [0.7, 0.8, 0.9, 1.0],
    "model__estimator__bagging_fraction": [0.7, 0.8, 0.9, 1.0],
    "model__estimator__bagging_freq": [1],
}

lightGBM_tiny_grid_3target = {
    "model__estimator__num_leaves": [31, 63],        # small vs larger trees
    "model__estimator__max_depth": [5],              # fix depth, keep search lean
    "model__estimator__learning_rate": [0.05, 0.1],  # stable vs faster learning
    "model__estimator__n_estimators": [500],         # one good baseline
    "model__estimator__min_data_in_leaf": [50, 100], # regularization strength
    "model__estimator__lambda_l2": [0, 1],           # quick check on L2 reg
    "model__estimator__feature_fraction": [0.8, 1.0],# subset vs full features
    "model__estimator__bagging_fraction": [0.8, 1.0],# row sampling
    "model__estimator__bagging_freq": [1],
}

lightGBM_small_grid_3target = {
    "model__estimator__num_leaves": [31, 63],              # tree complexity
    "model__estimator__max_depth": [5, 10],                # depth control
    "model__estimator__learning_rate": [0.05, 0.1],        # stable vs faster
    "model__estimator__n_estimators": [500, 1000],         # balance with LR
    "model__estimator__min_data_in_leaf": [50, 100],       # regularization
    "model__estimator__lambda_l1": [0, 1],                 # L1 reg
    "model__estimator__lambda_l2": [0, 1],                 # L2 reg
    "model__estimator__min_gain_to_split": [0.0, 0.1],     # split threshold
    "model__estimator__feature_fraction": [0.8, 1.0],      # col sampling
    "model__estimator__bagging_fraction": [0.8, 1.0],      # row sampling
    "model__estimator__bagging_freq": [1],
}

lightGBM_baseline_grid_3target = {
    "model__estimator__num_leaves": [31, 63],          # tree complexity
    "model__estimator__learning_rate": [0.05, 0.1],    # hot zone
    "model__estimator__n_estimators": [500, 1000],     # complements LR
    "model__estimator__min_data_in_leaf": [50, 100],   # regularization
    "model__estimator__feature_fraction": [0.8, 1.0],  # col subsampling
    "model__estimator__bagging_fraction": [0.8, 1.0],  # row subsampling
    "model__estimator__bagging_freq": [1],
}

lightGBM_ultralean_grid_3target = {
    "model__estimator__num_leaves": [31, 63],
    "model__estimator__max_depth": [-1, 10],
    "model__estimator__learning_rate": [0.05, 0.1],
    "model__estimator__n_estimators": [500, 1000],
}


# Small grid: quick runs
lightGBM_small_grid = {
    "model__estimator__num_leaves": [31, 63],
    "model__estimator__learning_rate": [0.05, 0.1],
    "model__estimator__n_estimators": [200, 500],
}