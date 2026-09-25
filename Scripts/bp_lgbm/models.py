######## MODEL SCRIPT ################################################################
#                                                                                    #
# This script handles the model building (with config) and the training,             #
# of the different tree based models that I am using.                                #
#                                                                                    #
######################################################################################

# lgbm_pipeline.py
# LightGBM training

from typing import Any, Protocol
import numpy as np
try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise SystemExit(
        "LightGBM is not installed. Install it with:\n"
        "  pip install lightgbm"
    )
# Protocol for sklearn-like estimator
class SklearnModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, *args, **kwargs): ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...

# ----------------------------
# Light GBM Model
# ----------------------------
def build_lgbm(cfg: Any) -> LGBMRegressor: #I have to tell that it receives a config like object at some point
    return LGBMRegressor(
        **cfg.model_params,              # unpack hyperparameters dict. I am doing it this way, so I can keep the random state consistent throughout the experiments and just change the model params after each grid search iteration.
        random_state=cfg.random_state,   # still keep global random_state
        n_jobs= -1,
        verbose = -1
    )

# ----------------------------
# Training
# ----------------------------
def train_model(
    model: SklearnModel,
    X_train,
    y_train,
    X_val,
    y_val,
    metric: str | None = None,
) -> SklearnModel:
    """Train a sklearn-like model with LightGBM-style eval metric."""
    eval_metric = metric if metric is not None else "l2"
    model.fit(
        X_train,
        y_train,
        eval_set=[("val", X_val, y_val)],
        eval_metric=eval_metric,
        verbose=False,
    )
    return model

# ----------------------------
# Main
# ----------------------------
"""
def main():
    cfg = Config()

    # 1) Dummy data
    X, y = make_dummy_regression(cfg)

    # 2) Preprocess (split + optional scale)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_split(X, y, cfg)

    # 3) Model
    model = build_lgbm(cfg)

    # 4) Train
    model = train_model(model, X_train, y_train, X_val, y_val)

    # 5) Evaluate on test
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # Print neat summary
    print("\n=== Evaluation (Test Set) ===")
    for k in [
        "MAE", "MAE_SD", "ME", "ME_SD", "MSE", "RMSE", "R2",
        "AbsError_mean", "AbsError_std", "AbsError_min", "AbsError_max",
        "BA_bias_ME", "BA_sd_diff", "BA_loa_low", "BA_loa_high",
        "BA_plot_path",
    ]:
        print(f"{k:>14}: {metrics[k]:.4f}" if isinstance(metrics[k], float) else f"{k:>14}: {metrics[k]}")

    # If you also want the absolute error vector for downstream analysis:
    # (Recompute here to avoid storing a huge array in 'metrics' by default)
    abs_errors = np.abs(y_pred - y_test)
    # Example: show first 10 absolute errors
    print("\nFirst 10 absolute errors:", np.round(abs_errors[:10], 3))

if __name__ == "__main__":
    main()
    """