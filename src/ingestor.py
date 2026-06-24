import pandas as pd
import numpy as np
import os

def load_data(filepath):
    """
    Reads a CSV file and returns a dataframe.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if not filepath.endswith('.csv'):
        raise ValueError("Only CSV files are supported.")
    
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def detect_column_types(df):
    """
    Detects whether each column is numerical, categorical, or datetime.
    Returns a dictionary mapping column name to type.
    """
    col_types = {}

    for col in df.columns:
        # Try datetime detection first
        if df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col])
                col_types[col] = 'datetime'
            except:
                col_types[col] = 'categorical'
        elif df[col].dtype in ['int64', 'float64']:
            col_types[col] = 'numerical'
        else:
            col_types[col] = 'categorical'

    return col_types


def detect_problem_type(df, target_col):
    """
    Detects whether the problem is classification or regression.
    Logic:
    - If target has <= 10 unique values → classification
    - If target is continuous numerical → regression
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    target = df[target_col]
    unique_values = target.nunique()

    if unique_values <= 10:
        problem_type = 'classification'
    elif target.dtype in ['int64', 'float64']:
        problem_type = 'regression'
    else:
        problem_type = 'classification'

    print(f"Problem type detected: {problem_type}")
    print(f"Target column: {target_col} | Unique values: {unique_values}")
    return problem_type


def validate_data(df, target_col):
    """
    Basic validation before processing starts.
    Checks for minimum rows, target column existence, all null columns.
    """
    errors = []

    # Check minimum rows
    if df.shape[0] < 10:
        errors.append("Dataset has less than 10 rows. Too small to train.")

    # Check target column exists
    if target_col not in df.columns:
        errors.append(f"Target column '{target_col}' not found.")

    # Check if any column is completely empty
    fully_null = df.columns[df.isnull().all()].tolist()
    if fully_null:
        errors.append(f"These columns are completely empty: {fully_null}")

    # Check if dataset has at least 2 columns
    if df.shape[1] < 2:
        errors.append("Dataset must have at least 2 columns.")

    if errors:
        for error in errors:
            print(f"Validation Error: {error}")
        raise ValueError("Data validation failed. Fix the errors above.")
    
    print("Data validation passed.")
    return True


def ingest(filepath, target_col):
    """
    Master function that runs the full ingestion pipeline.
    Call this from app.py
    """
    df = load_data(filepath)
    validate_data(df, target_col)
    col_types = detect_column_types(df)
    problem_type = detect_problem_type(df, target_col)

    print("\nColumn Types Detected:")
    for col, ctype in col_types.items():
        print(f"  {col}: {ctype}")

    return df, col_types, problem_type