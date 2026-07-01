import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest
from src.quality import run_quality_check
from src.preprocessor import preprocess

def test_preprocessor():
    df, col_types, problem_type = ingest(
        "data/Titanic-Dataset.csv", target_col="Survived")
    
    quality_report = run_quality_check(
        df, col_types, "Survived", problem_type)
    
    X_train, X_test, y_train, y_test, selected_features, scaler = preprocess(
        df, "Survived", col_types, problem_type, quality_report)

    # Check shapes are valid
    assert X_train.shape[0] > 0, "X_train is empty"
    assert X_test.shape[0] > 0, "X_test is empty"

    # Check no missing values remain
    assert X_train.isnull().sum().sum() == 0, "Missing values in X_train"
    assert X_test.isnull().sum().sum() == 0, "Missing values in X_test"

    # Check features were selected
    assert len(selected_features) > 0, "No features selected"

    # Check y has correct length
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

    print(f"\nSelected features: {selected_features}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("\nAll preprocessor tests passed.")

test_preprocessor()