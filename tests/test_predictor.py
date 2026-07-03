import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.preprocessor import preprocess
from src.trainer import run_training, get_scoring_metric
from src.tuner import run_tuning
from src.predictor import save_model, predict


def test_predictor():
    # Run full pipeline to get trained model
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

    # Save model
    model_path = save_model(
        best_tuned_model, best_tuned_name,
        selected_features, scaler, problem_type)

    # Check model file exists
    assert os.path.exists(model_path)

    # Predict on test data
    result = predict(X_test, model_path)

    # Check predictions exist
    assert result is not None
    assert len(result["predictions"]) == len(X_test)
    assert result["model_used"] == best_tuned_name

    # Check confidence scores exist
    assert "confidence" in result
    assert len(result["confidence"]) == len(X_test)

    # Check all confidence values are between 0 and 100
    for conf in result["confidence"]:
        assert 0 <= conf <= 100

    print(f"\nModel used: {result['model_used']}")
    print(f"Predictions made: {result['num_predictions']}")
    print(f"Sample confidence: {result['confidence'][:5]}")
    print("\nAll predictor tests passed.")


test_predictor()