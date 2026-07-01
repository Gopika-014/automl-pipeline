import pandas as pd
import numpy as np


def check_missing(df):
    """
    Checks missing values per column.
    Returns percentage of missing values per column.
    Penalizes score if any column exceeds 20% missing.
    """
    missing = df.isnull().mean() * 100
    missing_dict = missing[missing > 0].round(2).to_dict()

    penalty = 0
    warnings = []

    for col, pct in missing_dict.items():
        if pct > 40:
            penalty += 15
            warnings.append(f"CRITICAL: '{col}' has {pct}% missing values. Consider dropping.")
        elif pct > 20:
            penalty += 8
            warnings.append(f"WARNING: '{col}' has {pct}% missing values.")
        elif pct > 0:
            penalty += 3
            warnings.append(f"INFO: '{col}' has {pct}% missing values.")

    return missing_dict, penalty, warnings


def check_class_imbalance(df, target_col, problem_type):
    """
    For classification — checks if minority class is below 20%.
    Severe imbalance affects model performance significantly.
    """
    if problem_type != 'classification':
        return None, 0, []

    class_counts = df[target_col].value_counts()
    total = len(df)
    minority_pct = (class_counts.min() / total) * 100

    penalty = 0
    warnings = []

    if minority_pct < 10:
        penalty = 20
        warnings.append(f"CRITICAL: Severe class imbalance. Minority class is only {minority_pct:.1f}% of data. SMOTE will be applied.")
    elif minority_pct < 20:
        penalty = 10
        warnings.append(f"WARNING: Class imbalance detected. Minority class is {minority_pct:.1f}% of data.")
    else:
        warnings.append(f"INFO: Class distribution is acceptable. Minority class is {minority_pct:.1f}% of data.")

    distribution = class_counts.to_dict()
    return distribution, penalty, warnings


def check_duplicates(df):
    """
    Checks for duplicate rows.
    Duplicates can cause data leakage and inflate model performance.
    """
    duplicate_count = df.duplicated().sum()
    duplicate_pct = (duplicate_count / len(df)) * 100

    penalty = 0
    warnings = []

    if duplicate_pct > 10:
        penalty = 10
        warnings.append(f"CRITICAL: {duplicate_count} duplicate rows ({duplicate_pct:.1f}%). Remove before training.")
    elif duplicate_pct > 5:
        penalty = 5
        warnings.append(f"WARNING: {duplicate_count} duplicate rows ({duplicate_pct:.1f}%) found.")
    else:
        warnings.append(f"INFO: {duplicate_count} duplicate rows found. Acceptable.")

    return int(duplicate_count), penalty, warnings


def check_outliers(df, col_types):
    """
    Detects outliers in numerical columns using Z-score method.
    Z-score > 3 means the value is 3 standard deviations away from mean.
    These are considered outliers.
    """
    numerical_cols = [col for col, ctype in col_types.items()
                      if ctype == 'numerical']

    outlier_dict = {}
    penalty = 0
    warnings = []

    for col in numerical_cols:
        col_data = df[col].dropna()
        z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
        outlier_count = int((z_scores > 3).sum())
        outlier_pct = (outlier_count / len(col_data)) * 100
        outlier_dict[col] = outlier_count

        if outlier_pct > 10:
            penalty += 5
            warnings.append(f"WARNING: '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%).")
        elif outlier_count > 0:
            warnings.append(f"INFO: '{col}' has {outlier_count} outliers.")

    return outlier_dict, penalty, warnings


def compute_quality_score(missing_penalty, imbalance_penalty,
                          duplicate_penalty, outlier_penalty):
    """
    Computes overall data quality score out of 100.
    Starts at 100 and deducts penalties.
    """
    total_penalty = (missing_penalty + imbalance_penalty +
                     duplicate_penalty + outlier_penalty)
    score = max(0, 100 - total_penalty)
    return round(score, 1)


def get_quality_label(score):
    """
    Converts score to human readable label.
    """
    if score >= 80:
        return "Good"
    elif score >= 60:
        return "Moderate"
    elif score >= 40:
        return "Poor"
    else:
        return "Critical"


def run_quality_check(df, col_types, target_col, problem_type):
    """
    Master function — runs all quality checks.
    Returns full quality report dictionary.
    Call this from app.py
    """
    print("\nRunning Data Quality Check...")

    # Run all checks
    missing_dict, missing_penalty, missing_warnings = check_missing(df)

    distribution, imbalance_penalty, imbalance_warnings = check_class_imbalance(
        df, target_col, problem_type)

    duplicate_count, duplicate_penalty, duplicate_warnings = check_duplicates(df)

    outlier_dict, outlier_penalty, outlier_warnings = check_outliers(
        df, col_types)

    # Compute final score
    score = compute_quality_score(
        missing_penalty, imbalance_penalty,
        duplicate_penalty, outlier_penalty)

    label = get_quality_label(score)

    # Collect all warnings
    all_warnings = (missing_warnings + imbalance_warnings +
                    duplicate_warnings + outlier_warnings)

    # Build quality report
    quality_report = {
        "quality_score": score,
        "quality_label": label,
        "missing_values": missing_dict,
        "missing_penalty": missing_penalty,
        "class_distribution": distribution,
        "imbalance_penalty": imbalance_penalty,
        "duplicate_rows": duplicate_count,
        "duplicate_penalty": duplicate_penalty,
        "outliers_per_column": outlier_dict,
        "outlier_penalty": outlier_penalty,
        "total_penalty": (missing_penalty + imbalance_penalty +
                          duplicate_penalty + outlier_penalty),
        "warnings": all_warnings
    }

    # Print summary
    print(f"\nData Quality Score: {score}/100 — {label}")
    print("\nWarnings:")
    for w in all_warnings:
        print(f"  {w}")

    return quality_report