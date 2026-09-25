| Hyperparameter                              | Effect on Model                                       | Why It Matters (with 28 features & BP data)                                                                              |
| ------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **`num_leaves`**                            | Controls tree complexity (more leaves = more splits). | With 28 features, too many leaves risks fitting noise. Range 15–63 balances expressiveness vs overfitting.               |
| **`max_depth`**                             | Maximum depth per tree.                               | Depth 5–10 captures non-linear relations between PPG features & BP without creating overly complex rules.                |
| **`learning_rate`**                         | Step size per boosting iteration.                     | Lower rates (0.05–0.1) stabilize learning; higher rates (0.2) are faster but risk overshooting with only 28 features.    |
| **`n_estimators`**                          | Number of boosting trees.                             | Needs to complement `learning_rate`. More trees (500–1000) with lower LR avoids unstable fits.                           |
| **`min_data_in_leaf`**                      | Minimum samples per leaf.                             | Ensures splits are based on many patients, not just edge cases. Recommended 50–100 for biomedical data.                  |
| **`lambda_l1`, `lambda_l2`**                | L1/L2 regularization on leaf weights.                 | Smooths predictions and reduces reliance on redundant PPG features. Useful since engineered features may be correlated.  |
| **`min_gain_to_split`**                     | Minimum loss reduction to allow a split.              | Avoids unhelpful splits driven by noise in BP/PPG data.                                                                  |
| **`feature_fraction`**                      | Fraction of features used per tree.                   | With 28 features, using 0.7–0.9 forces diversity (≈20–25 features/tree), reducing reliance on a few dominant predictors. |
| **`bagging_fraction`** & **`bagging_freq`** | Fraction/frequency of data subsampling per iteration. | Using 70–90% of \~100k samples per tree improves generalization, speeds training, and reduces overfitting.               |
| **`max_bin`**                               | Number of bins for feature histogram.                 | Controls split precision. Default (255) is usually fine for continuous PPG features.                                     |


🩺 Why These Matter in BP Estimation

PPG features are noisy → Regularization (min_data_in_leaf, lambda_l1/l2) prevents overfitting.

28 features = moderate dimensionality → Subsampling (feature_fraction) and controlled capacity (num_leaves, max_depth) balance flexibility with stability.

Clinical robustness needed → Stable hyperparameters (lower learning rate, higher n_estimators) reduce the risk of spurious patient-specific patterns.