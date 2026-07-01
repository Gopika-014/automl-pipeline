import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.preprocessor import preprocess
from src.trainer import run_training, get_scoring_metric
from src.tuner import run_tuning

def test_tuner():
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

    # Tuned results should exist
    assert len(tuned_results) > 0

    # Best tuned model should exist
    assert best_tuned_model is not None
    assert best_tuned_name in tuned_results

    # Tuned score should be reasonable
    best_score = tuned_results[best_tuned_name]["best_score"]
    assert best_score > 0.5

    print(f"\nBest tuned model: {best_tuned_name}")
    print(f"Best tuned score: {best_score}")
    print("\nAll tuner tests passed.")

test_tuner()