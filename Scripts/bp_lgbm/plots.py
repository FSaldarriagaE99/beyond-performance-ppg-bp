############ PLOTS CODE #######################################
#                                                             #
# All of the plots stored here                                #
#                                                             #
###############################################################

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set_style("whitegrid")

# =========================================================
# Demographics plots
#==========================================================
# Color mapping for demographics
demo_colors = {
    "Age": "#1f77b4",      # blue
    "BMI": "#ff7f0e",      # orange
    "Height": "#2ca02c",   # green
    "Weight": "#9467bd",   # purple
    "SBP": "#d62728",      # red
    "DBP": "#17becf"       # teal
}


def plot_numeric_distributions(df, cols, dataset_name="Train"):
    """Plot histograms for numeric demographic variables."""
    for col in cols:
        plt.figure(figsize=(6,4))
        sns.histplot(
            df[col], bins=30, kde=False,
            color=demo_colors.get(col, "gray"), edgecolor="black"
        )
        plt.title(f"{col} distribution ({dataset_name})", fontsize=14, weight="bold")
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout()
        plt.show()

def plot_gender_distribution(df, dataset_name="Train"):
    """Plot gender distribution with Male=blue, Female=red."""
    plt.figure(figsize=(6,4))
    gender_order = ["M", "F"] if "M" in df["Gender"].unique() else None
    sns.countplot(
        x="Gender", data=df,
        order=gender_order,
        palette={"M": "blue", "F": "red"}
    )
    plt.title(f"Gender distribution ({dataset_name})", fontsize=14, weight="bold")
    plt.xlabel("Gender", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_threshold_proportions(df, thresholds_dict, dataset_name="Dataset", subject_col="Subject"):
    """
    Plot bar plots showing number of signals, percentage, 
    and number of unique subjects (in parentheses).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the variables.
    thresholds_dict : dict
        Dictionary with variable name as key and list of thresholds as value.
        Example: {"Age": [40, 60], "BMI": [18.5, 25, 30]}
    dataset_name : str
        Label to show in the plot titles.
    subject_col : str
        Column name for subject IDs.
    """
    for col, cutoffs in thresholds_dict.items():
        if col not in df.columns:
            print(f"⚠️ Skipping {col}: not found in DataFrame.")
            continue

        # Build bins
        bins = [df[col].min()] + cutoffs + [df[col].max()]
        labels = [f"{bins[i]:.1f}–{bins[i+1]:.1f}" for i in range(len(bins)-1)]

        # Bin data
        binned = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True, right=False)

        # Counts and percentages
        counts = binned.value_counts().sort_index()
        percentages = counts / counts.sum() * 100

        # Unique subjects per bin
        subjects_per_bin = df.groupby(binned)[subject_col].nunique().reindex(labels)

        # Update labels to include subjects
        labels_with_subjects = [f"{lab}\n({subs} subj.)" for lab, subs in zip(labels, subjects_per_bin)]

        # Plot
        plt.figure(figsize=(8,5))
        ax = sns.barplot(
            x=labels_with_subjects, y=counts.values,
            color=demo_colors.get(col, "gray")
        )

        # Annotate with percentages
        for i, (c, p) in enumerate(zip(counts.values, percentages.values)):
            ax.text(i, c + max(counts.values)*0.01, f"{p:.1f}%", 
                    ha="center", va="bottom", fontsize=10, weight="bold")

        plt.title(f"{col} categories ({dataset_name})", fontsize=14, weight="bold")
        plt.ylabel("Number of signals")
        plt.xlabel(col)
        plt.xticks(rotation=45)
        plt.show()