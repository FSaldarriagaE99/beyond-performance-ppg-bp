######## GRID SEARCH SCRIPT ##########################################################
#                                                                                    #
# This script handles searching and finding the best hyperparameters.                #
#                                                                                    #
#                                                                                    #
######################################################################################
"""
Note: It uses CV, either pateint wise or sample wise
"""

# grid_search.py
import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, RandomizedSearchCV


def run_grid_search(
    pipeline,
    X,
    y,
    groups=None,
    param_grid: dict = None,
    n_splits: int = 5,
    cv_type: str = "group",   # "group" (GroupKFold) or "sample" (KFold)
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
    verbose: int = 2,
    save_results: bool = False,
    search_mode="grid",
    n_iter=100,
    random_state: int = 42,
):
    """
    Run grid search over a given pipeline with patient-wise (GroupKFold) or sample-wise (KFold) CV.

    Args:
        pipeline: sklearn Pipeline object (scaler + model)
        X, y: data
        groups: patient IDs for grouping (required if cv_type="group")
        param_grid: dict of hyperparams (must use 'model__' prefix)
        n_splits: folds
        cv_type: "group" (GroupKFold) or "sample" (KFold)
        scoring: sklearn scoring string (default MSE)
        n_jobs: parallel jobs
        verbose: verbosity
        search_mode: "grid" or "random". grid does the full grid search whereas random conducts randomized grid search.
        n_iter: how many different combinations does the random select.

    Returns:
        search: fitted GridSearchCV object
        results_df: pandas DataFrame with all results
    """
    if cv_type == "group":
        if groups is None:
            raise ValueError("groups must be provided for GroupKFold")
        cv = GroupKFold(n_splits=n_splits)
        cv_split = cv.split(X, y, groups)
    elif cv_type == "sample":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_split = cv.split(X, y)
    else:
        raise ValueError("cv_type must be 'group' or 'sample'")

    if search_mode == "grid":
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_split,
            n_jobs=n_jobs,
            refit=False,   # refit best model on all training data
            verbose=verbose,
            return_train_score=True,
        )
    elif search_mode == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score = True
        )
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")
    search.fit(X, y, groups = groups)

    # convert results into a dataframe
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.rename(columns=lambda x: x.replace("param_model__", ""))

    # add flipped MSE columns (positive values)
    if scoring == "neg_mean_squared_error":
        results_df["mean_test_MSE"] = -results_df["mean_test_score"]
        results_df["std_test_MSE"] = results_df["std_test_score"]
        results_df["mean_train_MSE"] = -results_df["mean_train_score"]
        results_df["std_train_MSE"] = results_df["std_train_score"]

    # optionally save results
    if save_results:
        folder_name = f"CV_results_{cv_type}"
        os.makedirs(folder_name, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(folder_name, f"grid_results_{timestamp}.csv")
        results_df.to_csv(file_path, index=False)
        print(f"✅ Results saved to {file_path}")

    return search, results_df

