import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

PLOTS_DIR = "reports/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_tree_feature_importance(best_model, feature_names, model_name):
    """
    Level 1 — Decision Tree / Tree-based Feature Importance.

    How it works:
    Every time a tree splits on a feature, it reduces
    impurity (Gini or entropy) in the resulting nodes.
    Feature importance = total impurity reduction
    caused by that feature across all splits.

    Features that cause bigger, more frequent splits
    get higher importance scores.

    Limitation:
    Biased toward high cardinality features.
    A feature with many unique values gets more
    split opportunities and appears more important
    even if it isn't.
    """
    if not hasattr(best_model, 'feature_importances_'):
        print(f"{model_name} does not have built-in feature importance.")
        return None, None

    importances = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n[Level 1] Tree Feature Importance:")
    print(importance_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][:10],
             importance_df['importance'][:10],
             color='steelblue', edgecolor='black')
    plt.xlabel("Importance Score")
    plt.title(f"Tree Feature Importance — {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = f"{PLOTS_DIR}/tree_feature_importance.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    return importance_df, path


def get_random_forest_importance(X_train, y_train,
                                 feature_names, problem_type):
    """
    Level 2 — Random Forest Feature Importance.

    Same concept as decision tree importance
    but averaged across hundreds of trees.

    Why this is more reliable than single tree:
    A single decision tree is unstable — small
    changes in data can change which features
    get selected for splits.

    Random Forest trains 100 trees on random
    subsets of data and features. Averaging
    importance across all trees gives a much
    more stable and reliable estimate.

    This is the standard way to get feature
    importance in industry for tabular data.
    """
    if problem_type == 'classification':
        rf = RandomForestClassifier(
            n_estimators=100, random_state=42)
    else:
        rf = RandomForestRegressor(
            n_estimators=100, random_state=42)

    rf.fit(X_train, y_train)
    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print("\n[Level 2] Random Forest Feature Importance:")
    print(importance_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][:10],
             importance_df['importance'][:10],
             color='darkorange', edgecolor='black')
    plt.xlabel("Importance Score")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = f"{PLOTS_DIR}/rf_feature_importance.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    return importance_df, path


def get_permutation_importance(best_model, X_test, y_test,
                               feature_names, model_name):
    """
    Level 3 — Permutation Importance.

    How it works:
    1. Get baseline model score on test set.
    2. For each feature, randomly shuffle its values.
       This breaks the relationship between that
       feature and the target.
    3. Measure how much the score drops.
    4. Big drop = that feature was important.
       No drop = model didn't rely on that feature.

    Why this is the most reliable method:
    - Works for ANY model, not just trees
    - Measures actual impact on predictions
    - Not biased toward high cardinality features
    - Based on held-out test data, not training data

    Limitation:
    Slow on large datasets.
    Correlated features can split importance
    between them — if two features carry same
    info, shuffling one barely hurts because
    the other still carries that info.
    """
    print("\n[Level 3] Permutation Importance...")

    perm_result = permutation_importance(
        best_model, X_test, y_test,
        n_repeats=10,
        random_state=42
    )

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=False)

    print(importance_df.to_string(index=False))

    # Plot
    plt.figure(figsize=(8, 5))
    plt.barh(importance_df['feature'][:10],
             importance_df['importance_mean'][:10],
             xerr=importance_df['importance_std'][:10],
             color='seagreen', edgecolor='black')
    plt.xlabel("Mean Importance (score drop when shuffled)")
    plt.title(f"Permutation Importance — {model_name}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    path = f"{PLOTS_DIR}/permutation_importance.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")

    return importance_df, path


def generate_plain_english_summary(tree_importance_df,
                                   perm_importance_df,
                                   model_name):
    """
    Converts feature importance into plain English.
    This goes into the HTML report as a human readable
    explanation — no ML knowledge needed to understand it.
    """
    summary_lines = []

    if tree_importance_df is not None:
        top_features = tree_importance_df['feature'].head(3).tolist()
        summary_lines.append(
            f"According to {model_name}, the three most important "
            f"factors are: {top_features[0]}, {top_features[1]}, "
            f"and {top_features[2]}."
        )

    if perm_importance_df is not None:
        top_perm = perm_importance_df['feature'].head(1).tolist()[0]
        summary_lines.append(
            f"Permutation analysis confirms that '{top_perm}' "
            f"has the strongest actual impact on predictions — "
            f"shuffling this feature causes the biggest drop in performance."
        )

    if tree_importance_df is not None:
        bottom_features = tree_importance_df['feature'].tail(2).tolist()
        summary_lines.append(
            f"The least influential features are "
            f"'{bottom_features[0]}' and '{bottom_features[1]}', "
            f"which contribute minimally to the model's decisions."
        )

    summary = " ".join(summary_lines)
    print(f"\nPlain English Summary:\n{summary}")
    return summary


def run_feature_importance(best_model, best_model_name,
                           X_train, X_test, y_train, y_test,
                           selected_features, problem_type):
    """
    Master feature importance function.
    Runs all three levels and generates plain English summary.
    Call this from app.py
    """
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE")
    print("="*50)

    feature_names = selected_features
    plot_paths = {}
    importance_results = {}

    # Level 1 — Tree importance
    tree_df, tree_path = get_tree_feature_importance(
        best_model, feature_names, best_model_name)
    if tree_path:
        plot_paths['tree_importance'] = tree_path
        importance_results['tree'] = tree_df

    # Level 2 — Random Forest importance
    rf_df, rf_path = get_random_forest_importance(
        X_train, y_train, feature_names, problem_type)
    plot_paths['rf_importance'] = rf_path
    importance_results['random_forest'] = rf_df

    # Level 3 — Permutation importance
    perm_df, perm_path = get_permutation_importance(
        best_model, X_test, y_test,
        feature_names, best_model_name)
    plot_paths['permutation_importance'] = perm_path
    importance_results['permutation'] = perm_df

    # Plain English summary
    summary = generate_plain_english_summary(
        tree_df, perm_df, best_model_name)

    print("\n" + "="*50)
    print("FEATURE IMPORTANCE COMPLETE")
    print("="*50)

    return importance_results, plot_paths, summary