import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.preprocessor import preprocess
from src.trainer import run_training

def test_trainer():
    df, col_types, problem_type = ingest(
        "data/Titanic-Dataset.csv", target_col="Survived")

    quality_report = run_quality_check(
        df, col_types, "Survived", problem_type)

    X_train, X_test, y_train, y_test, selected_features, scaler = preprocess(
        df, "Survived", col_types, problem_type, quality_report)

    best_model, best_model_name, cv_results, leaderboard, test_metrics = run_training(
        X_train, X_test, y_train, y_test, problem_type)

    # Best model should exist
    assert best_model is not None
    assert best_model_name in cv_results

    # Leaderboard should have all models
    assert len(leaderboard) > 0

    # Test metrics should exist
    assert "accuracy" in test_metrics
    assert test_metrics["accuracy"] > 0.5

    # CV results should have all models
    assert len(cv_results) == 7

    print(f"\nBest model: {best_model_name}")
    print(f"Test accuracy: {test_metrics['accuracy']}")
    print("\nAll trainer tests passed.")

test_trainer()