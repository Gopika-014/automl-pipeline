import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestor import ingest

# Download titanic.csv from kaggle and put it inside data/ folder
# Then run this test

def test_ingestor():
    df, col_types, problem_type = ingest("data/Titanic-Dataset.csv", target_col="Survived")
    
    # Check dataframe is not empty
    assert df.shape[0] > 0, "Dataframe is empty"
    
    # Check problem type is classification
    assert problem_type == "classification", f"Expected classification, got {problem_type}"
    
    # Check col_types is a dictionary
    assert isinstance(col_types, dict), "col_types should be a dictionary"
    
    # Check target column is in col_types
    assert "Survived" in col_types, "Target column missing from col_types"
    
    print("\nAll tests passed for ingestor.")

test_ingestor()