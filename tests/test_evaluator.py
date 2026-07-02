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

def test_evaluator():
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

    metrics, plot_paths = run_evaluation(
        best_tuned_model, best_tuned_name,
        X_test, y_test, problem_type)

    # Metrics should exist
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    # Score should be reasonable
    assert metrics["accuracy"] > 0.5
    assert metrics["roc_auc"] > 0.5

    # Plots should be saved
    assert os.path.exists("reports/plots/confusion_matrix.png")
    assert os.path.exists("reports/plots/roc_curve.png")

    print(f"\nAccuracy: {metrics['accuracy']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    print("\nAll evaluator tests passed.")

test_evaluator()