import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def save_model(best_model, best_model_name,
               selected_features, scaler,
               problem_type):
    """
    Saves the trained model and metadata to disk.

    Why save a model:
    Training takes time. Once trained, you don't
    want to retrain every time someone needs a
    prediction. Save once, load and predict anytime.

    What joblib does:
    Serializes Python objects to disk.
    Much faster than pickle for large numpy arrays
    and scikit-learn models.

    What we save:
    - The trained model object
    - Selected feature names
      (so new data uses exact same columns)
    - Scaler if used
      (so new data is scaled the same way)
    - Problem type
      (so predictor knows classification vs regression)
    - Model name
    """
    model_data = {
        "model": best_model,
        "model_name": best_model_name,
        "selected_features": selected_features,
        "scaler": scaler,
        "problem_type": problem_type
    }

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(model_data, model_path)

    print(f"\nModel saved: {model_path}")
    print(f"Model name: {best_model_name}")
    print(f"Features: {selected_features}")
    return model_path


def load_model(model_path="models/best_model.pkl"):
    """
    Loads a previously saved model from disk.

    This is what happens in production:
    The API loads the model once when the server
    starts, then uses it for every prediction
    request without retraining.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            f"Run the full pipeline first to train and save a model."
        )

    model_data = joblib.load(model_path)
    print(f"Model loaded: {model_data['model_name']}")
    print(f"Expected features: {model_data['selected_features']}")
    return model_data


def validate_input(input_df, selected_features):
    """
    Validates that new input data has the right columns.

    Why this matters:
    In production, users may upload CSV files with
    wrong column names, missing columns, or extra
    columns. This catches those errors before
    the model crashes with a confusing error.
    """
    errors = []
    missing_cols = []
    extra_cols = []

    input_cols = set(input_df.columns.tolist())
    expected_cols = set(selected_features)

    # Check for missing columns
    missing_cols = list(expected_cols - input_cols)
    if missing_cols:
        errors.append(
            f"Missing columns: {missing_cols}. "
            f"These columns are required for prediction."
        )

    # Check for extra columns
    extra_cols = list(input_cols - expected_cols)
    if extra_cols:
        print(f"Note: Extra columns will be ignored: {extra_cols}")

    if errors:
        for e in errors:
            print(f"Input Error: {e}")
        raise ValueError("Input validation failed.")

    return True


def preprocess_input(input_df, selected_features, scaler):
    """
    Prepares new input data for prediction.

    Critical rule — apply SAME transformations
    as training:
    The model was trained on scaled, encoded data.
    New data must go through identical transformations
    or predictions will be wrong.

    What we do:
    1. Keep only the features the model was trained on
    2. Fill any missing values with median/mode
    3. Apply the same scaler that was fit on training data
    """
    # Keep only selected features in correct order
    df = input_df[selected_features].copy()

    # Fill missing values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Apply scaler if it was used during training
    if scaler is not None:
        df = pd.DataFrame(
            scaler.transform(df),
            columns=selected_features
        )
        print("Scaler applied to input data.")
    else:
        print("No scaler needed — tree-based model.")

    return df


def predict(input_df, model_path="models/best_model.pkl"):
    """
    Master prediction function.
    Takes new data, returns predictions.

    Input: DataFrame with same columns as training data
    Output: Dictionary with predictions and probabilities

    Call this from app.py for the prediction endpoint.
    """
    print("\n" + "="*50)
    print("PREDICTION")
    print("="*50)

    # Load model
    model_data = load_model(model_path)
    best_model = model_data["model"]
    model_name = model_data["model_name"]
    selected_features = model_data["selected_features"]
    scaler = model_data["scaler"]
    problem_type = model_data["problem_type"]

    # Validate input
    validate_input(input_df, selected_features)

    # Preprocess input
    processed_df = preprocess_input(
        input_df, selected_features, scaler)

    # Predict
    predictions = best_model.predict(processed_df)

    result = {
        "model_used": model_name,
        "problem_type": problem_type,
        "num_predictions": len(predictions),
        "predictions": predictions.tolist()
    }

    # Add probabilities for classification
    if (problem_type == 'classification' and
            hasattr(best_model, 'predict_proba')):
        probabilities = best_model.predict_proba(processed_df)
        result["probabilities"] = probabilities.tolist()
        result["confidence"] = [
            round(max(prob) * 100, 1)
            for prob in probabilities
        ]
        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"  Row {i+1}: Predicted = {predictions[i]}, "
                  f"Confidence = {result['confidence'][i]}%")
    else:
        print(f"\nSample predictions (first 5):")
        for i in range(min(5, len(predictions))):
            print(f"  Row {i+1}: Predicted = {predictions[i]:.4f}")

    print(f"\nTotal predictions made: {len(predictions)}")
    print("\n" + "="*50)
    print("PREDICTION COMPLETE")
    print("="*50)

    return result