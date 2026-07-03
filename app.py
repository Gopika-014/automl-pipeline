import os
import io
import json
import pandas as pd
from flask import Flask, request, jsonify, send_file
from src.ingestor import ingest
from src.quality import run_quality_check
from src.eda import run_eda
from src.preprocessor import preprocess
from src.trainer import run_training, get_scoring_metric
from src.tuner import run_tuning
from src.evaluator import run_evaluation
from src.feature_importance import run_feature_importance
from src.reporter import generate_report
from src.predictor import save_model, predict

app = Flask(__name__)

UPLOAD_FOLDER = "data"
REPORTS_DIR = "reports"
MODELS_DIR = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# ROUTE 1 — Health Check
# ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """
    Health check endpoint.
    Tells you the API is running.
    """
    return jsonify({
        "status": "running",
        "message": "AutoML Pipeline API is live.",
        "endpoints": {
            "POST /upload": "Upload CSV and run full AutoML pipeline",
            "POST /predict": "Upload new CSV and get predictions",
            "GET /report/<filename>": "Download generated HTML report"
        }
    })


# ─────────────────────────────────────────────
# ROUTE 2 — Upload CSV and Run Full Pipeline
# ─────────────────────────────────────────────

@app.route("/upload", methods=["POST"])
def upload():
    """
    Main endpoint — runs the full AutoML pipeline.

    How to call this:
    POST /upload
    Form data:
        - file: CSV file
        - target_col: name of target column

    Returns:
        - best model name
        - evaluation metrics
        - feature importance summary
        - report filename to download
    """
    # Check file was uploaded
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded. "
                     "Send CSV as 'file' in form data."
        }), 400

    file = request.files["file"]
    target_col = request.form.get("target_col", None)

    if not target_col:
        return jsonify({
            "error": "No target column specified. "
                     "Send 'target_col' in form data."
        }), 400

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not file.filename.endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:
        # Save uploaded file
        filepath = os.path.join(
            UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        print(f"\nFile saved: {filepath}")

        # ── Run full pipeline ──

        # Step 1 — Ingest
        df, col_types, problem_type = ingest(
            filepath, target_col)

        # Step 2 — Quality check
        quality_report = run_quality_check(
            df, col_types, target_col, problem_type)

        # Step 3 — EDA
        eda_plot_paths, eda_stats = run_eda(
            df, col_types, target_col, problem_type)

        # Step 4 — Preprocess
        (X_train, X_test, y_train, y_test,
         selected_features, scaler) = preprocess(
            df, target_col, col_types,
            problem_type, quality_report)

        # Step 5 — Train
        (best_model, best_model_name,
         cv_results, leaderboard,
         test_metrics) = run_training(
            X_train, X_test, y_train,
            y_test, problem_type)

        # Step 6 — Tune
        scoring_metric = get_scoring_metric(
            problem_type, y_train)

        (best_tuned_model, best_tuned_name,
         tuned_results) = run_tuning(
            cv_results, X_train, y_train,
            problem_type, scoring_metric, top_n=3)

        # Step 7 — Evaluate
        eval_metrics, eval_plot_paths = run_evaluation(
            best_tuned_model, best_tuned_name,
            X_test, y_test, problem_type)

        # Step 8 — Feature importance
        (importance_results, fi_plot_paths,
         summary) = run_feature_importance(
            best_tuned_model, best_tuned_name,
            X_train, X_test, y_train, y_test,
            selected_features, problem_type)

        # Step 9 — Save model
        model_path = save_model(
            best_tuned_model, best_tuned_name,
            selected_features, scaler, problem_type)

        # Step 10 — Generate report
        report_path = generate_report(
            quality_report, eda_stats, eda_plot_paths,
            leaderboard, cv_results, best_tuned_name,
            tuned_results, eval_metrics, eval_plot_paths,
            importance_results, fi_plot_paths,
            summary, problem_type)

        report_filename = os.path.basename(report_path)

        # Build response
        response = {
            "status": "success",
            "problem_type": problem_type,
            "data_quality": {
                "score": quality_report["quality_score"],
                "label": quality_report["quality_label"]
            },
            "best_model": best_tuned_name,
            "tuned_score": tuned_results[
                best_tuned_name]["best_score"],
            "best_params": tuned_results[
                best_tuned_name]["best_params"],
            "evaluation_metrics": eval_metrics,
            "top_features": selected_features[:5],
            "feature_summary": summary,
            "leaderboard": leaderboard,
            "report_url": f"/report/{report_filename}"
        }

        return jsonify(response), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({
            "error": f"Pipeline failed: {str(e)}"
        }), 500


# ─────────────────────────────────────────────
# ROUTE 3 — Predict on New Data
# ─────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict_route():
    """
    Prediction endpoint.
    Loads saved model and predicts on new CSV.

    How to call this:
    POST /predict
    Form data:
        - file: CSV file with same columns as training data
                (without target column)

    Returns:
        - predictions list
        - confidence scores (for classification)
    """
    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded."
        }), 400

    file = request.files["file"]

    if not file.filename.endswith(".csv"):
        return jsonify({
            "error": "Only CSV files are supported."
        }), 400

    try:
        # Read uploaded CSV into dataframe
        input_df = pd.read_csv(file)
        print(f"\nPrediction input shape: {input_df.shape}")

        # Run prediction
        result = predict(input_df)

        return jsonify({
            "status": "success",
            "model_used": result["model_used"],
            "problem_type": result["problem_type"],
            "num_predictions": result["num_predictions"],
            "predictions": result["predictions"],
            "confidence": result.get("confidence", None)
        }), 200

    except FileNotFoundError as fe:
        return jsonify({
            "error": str(fe)
        }), 404
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ─────────────────────────────────────────────
# ROUTE 4 — Download Report
# ─────────────────────────────────────────────

@app.route("/report/<filename>", methods=["GET"])
def get_report(filename):
    """
    Report download endpoint.
    Returns the generated HTML report file.

    How to call this:
    GET /report/report_20240624_123456.html
    """
    report_path = os.path.join(REPORTS_DIR, filename)

    if not os.path.exists(report_path):
        return jsonify({
            "error": f"Report '{filename}' not found."
        }), 404

    return send_file(
        report_path,
        mimetype="text/html",
        as_attachment=False
    )


# ─────────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("AutoML Pipeline API")
    print("="*50)
    print("Server starting at http://127.0.0.1:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              — Health check")
    print("  POST /upload        — Run full pipeline")
    print("  POST /predict       — Predict on new data")
    print("  GET  /report/<name> — Download report")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)