import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.preprocessor import preprocess
from src.trainer import run_training, get_scoring_metric
from src.tuner import run_tuning
from src.evaluator import run_evaluation
from src.feature_importance import run_feature_importance

def test_feature_importance():
    df, col_types, problem_type = ingest(
        "data/Titanic-Dataset.csv", target_col="Survived")

    quality_report = run_quality_check(
        df, col_types, "Survived", problem_type)

    X_train, X_test, y_train, y_test, selected_features, scaler = preprocess(
        df, "Survived", col_types, problem_type, quality_report)

    best_model, best_model_name, cv_results, leaderboard, test_metrics = run_training(
        X_train, X_test, y_train, y_test, problem_type)

    scoring_metric = get_scoring_metric(problem_type, y_train)

    best_tuned_model, best_tuned_name, tuned_results = run_tuning(
        cv_results, X_train, y_train,
        problem_type, scoring_metric, top_n=3)

    metrics, eval_plot_paths = run_evaluation(
        best_tuned_model, best_tuned_name,
        X_test, y_test, problem_type)

    importance_results, plot_paths, summary = run_feature_importance(
        best_tuned_model, best_tuned_name,
        X_train, X_test, y_train, y_test,
        selected_features, problem_type)

    # Results should exist
    assert len(importance_results) > 0

    # Plots should be saved
    assert os.path.exists(
        "reports/plots/rf_feature_importance.png")
    assert os.path.exists(
        "reports/plots/permutation_importance.png")

    # Summary should be a non empty string
    assert isinstance(summary, str)
    assert len(summary) > 0

    print(f"\nSummary: {summary}")
    print("\nAll feature importance tests passed.")

test_feature_importance()