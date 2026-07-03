import sys
import os
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.eda import run_eda
from src.preprocessor import preprocess
from src.trainer import run_training, get_scoring_metric
from src.tuner import run_tuning
from src.evaluator import run_evaluation
from src.feature_importance import run_feature_importance
from src.reporter import generate_report


def test_reporter():
    df, col_types, problem_type = ingest(
        "data/Titanic-Dataset.csv", target_col="Survived")

    quality_report = run_quality_check(
        df, col_types, "Survived", problem_type)

    eda_plot_paths, eda_stats = run_eda(
        df, col_types, "Survived", problem_type)

    X_train, X_test, y_train, y_test, selected_features, scaler = preprocess(
        df, "Survived", col_types, problem_type, quality_report)

    best_model, best_model_name, cv_results, leaderboard, test_metrics = run_training(
        X_train, X_test, y_train, y_test, problem_type)

    scoring_metric = get_scoring_metric(problem_type, y_train)

    best_tuned_model, best_tuned_name, tuned_results = run_tuning(
        cv_results, X_train, y_train,
        problem_type, scoring_metric, top_n=3)

    eval_metrics, eval_plot_paths = run_evaluation(
        best_tuned_model, best_tuned_name,
        X_test, y_test, problem_type)

    importance_results, fi_plot_paths, summary = run_feature_importance(
        best_tuned_model, best_tuned_name,
        X_train, X_test, y_train, y_test,
        selected_features, problem_type)

    report_path = generate_report(
        quality_report, eda_stats, eda_plot_paths,
        leaderboard, cv_results, best_tuned_name,
        tuned_results, eval_metrics, eval_plot_paths,
        importance_results, fi_plot_paths,
        summary, problem_type)

    # Report file should exist
    assert os.path.exists(report_path)

    # Report should not be empty
    assert os.path.getsize(report_path) > 1000

    print(f"\nReport generated: {report_path}")
    print("\nAll reporter tests passed.")


test_reporter()