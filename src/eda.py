import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats

# All plots will be saved to reports/plots/
PLOTS_DIR = "reports/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_class_distribution(df, target_col):
    """
    For classification — shows how many samples per class.
    Tells you if data is imbalanced.
    """
    plt.figure(figsize=(6, 4))
    df[target_col].value_counts().plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(f"Class Distribution — {target_col}")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/class_distribution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_target_distribution(df, target_col):
    """
    For regression — shows distribution of target variable.
    Tells you if target is skewed.
    """
    plt.figure(figsize=(6, 4))
    sns.histplot(df[target_col], kde=True, color='steelblue')
    plt.title(f"Target Distribution — {target_col}")
    plt.xlabel(target_col)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/target_distribution.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_missing_values(df):
    """
    Bar chart showing percentage of missing values per column.
    Helps decide imputation strategy.
    """
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values found.")
        return None

    plt.figure(figsize=(8, 4))
    missing.plot(kind='bar', color='salmon', edgecolor='black')
    plt.title("Missing Values (%) per Column")
    plt.ylabel("Missing %")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/missing_values.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_numerical_distributions(df, col_types):
    """
    Histogram for each numerical column.
    Shows skewness and outliers visually.
    """
    numerical_cols = [col for col, ctype in col_types.items()
                      if ctype == 'numerical']

    if not numerical_cols:
        print("No numerical columns found.")
        return None

    n = len(numerical_cols)
    fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3,
                             figsize=(15, (n + 2) // 3 * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        axes[i].hist(df[col].dropna(), bins=30,
                     color='steelblue', edgecolor='black')
        axes[i].set_title(f"{col}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numerical Column Distributions", fontsize=14)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/numerical_distributions.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_correlation_heatmap(df, col_types):
    """
    Heatmap showing correlation between numerical columns.
    High correlation between features = redundancy.
    """
    numerical_cols = [col for col, ctype in col_types.items()
                      if ctype == 'numerical']

    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation heatmap.")
        return None

    plt.figure(figsize=(10, 6))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f",
                cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    path = f"{PLOTS_DIR}/correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_outliers(df, col_types):
    """
    Boxplot for each numerical column.
    Points beyond whiskers are outliers.
    """
    numerical_cols = [col for col, ctype in col_types.items()
                      if ctype == 'numerical']

    if not numerical_cols:
        print("No numerical columns for outlier detection.")
        return None

    n = len(numerical_cols)
    fig, axes = plt.subplots(nrows=(n + 2) // 3, ncols=3,
                             figsize=(15, (n + 2) // 3 * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f"{col}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Outlier Detection (Boxplots)", fontsize=14)
    plt.tight_layout()
    path = f"{PLOTS_DIR}/outliers.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path


def compute_eda_stats(df, col_types, target_col):
    """
    Computes summary statistics for the report.
    Returns a dictionary of stats.
    """
    numerical_cols = [col for col, ctype in col_types.items()
                      if ctype == 'numerical']
    categorical_cols = [col for col, ctype in col_types.items()
                        if ctype == 'categorical']

    stats_dict = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "numerical_columns": len(numerical_cols),
        "categorical_columns": len(categorical_cols),
        "total_missing_values": int(df.isnull().sum().sum()),
        "missing_percentage": round(df.isnull().mean().mean() * 100, 2),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_column": target_col,
        "target_unique_values": int(df[target_col].nunique()),
        "class_distribution": df[target_col].value_counts().to_dict()
    }

    # Skewness for numerical columns
    skewness = {}
    for col in numerical_cols:
        skewness[col] = round(df[col].skew(), 2)
    stats_dict["skewness"] = skewness

    return stats_dict


def run_eda(df, col_types, target_col, problem_type):
    """
    Master EDA function — runs all plots and stats.
    Call this from app.py
    """
    print("\nRunning EDA...")
    plot_paths = {}

    # Target distribution
    if problem_type == 'classification':
        path = plot_class_distribution(df, target_col)
    else:
        path = plot_target_distribution(df, target_col)
    plot_paths['target'] = path

    # Missing values
    path = plot_missing_values(df)
    plot_paths['missing'] = path

    # Numerical distributions
    path = plot_numerical_distributions(df, col_types)
    plot_paths['distributions'] = path

    # Correlation heatmap
    path = plot_correlation_heatmap(df, col_types)
    plot_paths['correlation'] = path

    # Outliers
    path = plot_outliers(df, col_types)
    plot_paths['outliers'] = path

    # Stats
    stats = compute_eda_stats(df, col_types, target_col)

    print("\nEDA Summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nEDA complete.")
    return plot_paths, stats