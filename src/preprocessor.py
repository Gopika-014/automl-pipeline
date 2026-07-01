import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# STEP 1 — DROP USELESS COLUMNS
# ─────────────────────────────────────────────

def drop_useless_columns(df, target_col, col_types):
    """
    Drops columns that are useless for ML:
    - Columns with more than 70% missing values
    - ID-like columns (unique values == total rows)
    - Datetime columns (not handling time series here)
    """
    cols_to_drop = []

    for col in df.columns:
        if col == target_col:
            continue

        # Drop if more than 70% missing
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 70:
            cols_to_drop.append(col)
            print(f"Dropping '{col}' — {missing_pct:.1f}% missing values.")
            continue

        # Drop if ID-like (all unique values)
        if df[col].nunique() == len(df):
            cols_to_drop.append(col)
            print(f"Dropping '{col}' — looks like an ID column.")
            continue

        # Drop datetime columns
        if col_types.get(col) == 'datetime':
            cols_to_drop.append(col)
            print(f"Dropping '{col}' — datetime column.")
            continue

    df = df.drop(columns=cols_to_drop)
    print(f"\nColumns after dropping useless ones: {list(df.columns)}")
    return df, cols_to_drop


# ─────────────────────────────────────────────
# STEP 2 — HANDLE MISSING VALUES
# ─────────────────────────────────────────────

def handle_missing_values(df, target_col, col_types):
    """
    Imputes missing values:
    - Numerical columns → median
      (median is preferred over mean because it is
       robust to outliers)
    - Categorical columns → mode
      (most frequent value is the safest assumption)
    """
    for col in df.columns:
        if col == target_col:
            continue

        if df[col].isnull().sum() == 0:
            continue

        if col_types.get(col) == 'numerical':
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed '{col}' with median: {median_val:.2f}")

        elif col_types.get(col) == 'categorical':
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"Imputed '{col}' with mode: {mode_val}")

    return df


# ─────────────────────────────────────────────
# STEP 3 — ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────

def encode_categoricals(df, target_col, col_types, problem_type):
    """
    Encodes categorical columns:
    - Binary categories (2 unique values) → Label Encoding
      (0 and 1, simple and sufficient)
    - Multi categories (>2 unique values) → One Hot Encoding
      (creates separate column per category)

    Why not always One Hot?
    High cardinality columns create too many columns
    and slow down training significantly.

    Why not always Label Encoding?
    Label encoding implies order (0 < 1 < 2) which is
    wrong for non-ordinal categories like color or city.
    """
    categorical_cols = [col for col, ctype in col_types.items()
                        if ctype == 'categorical' and col in df.columns
                        and col != target_col]

    label_encoded = []
    ohe_encoded = []

    for col in categorical_cols:
        n_unique = df[col].nunique()

        if n_unique == 2:
            # Binary — Label Encoding
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoded.append(col)
            print(f"Label Encoded '{col}' — {n_unique} unique values.")

        elif n_unique <= 15:
            # Multi category — One Hot Encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            ohe_encoded.append(col)
            print(f"One Hot Encoded '{col}' — {n_unique} unique values.")

        else:
            # High cardinality — drop for now
            df = df.drop(columns=[col])
            print(f"Dropped '{col}' — too many unique values ({n_unique}).")

    # Encode target column if it is categorical
    if problem_type == 'classification':
        if df[target_col].dtype == 'object':
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
            print(f"Label Encoded target column '{target_col}'.")

    return df, label_encoded, ohe_encoded


# ─────────────────────────────────────────────
# STEP 4 — FEATURE SELECTION
# ─────────────────────────────────────────────

def select_features(X, y, problem_type, k=10):
    """
    Removes features that are not useful:

    Step 1 — Variance Threshold:
    Removes features with near zero variance.
    A feature that barely changes across samples
    teaches the model nothing.

    Step 2 — Correlation Filter:
    Removes highly correlated features (correlation > 0.90).
    Two features saying the same thing = redundancy.
    Keep one, drop the other.

    Step 3 — SelectKBest:
    Keeps top K features most related to target.
    Uses statistical test (f_classif for classification,
    f_regression for regression).
    """
    print(f"\nFeature selection — starting with {X.shape[1]} features.")

    # Step 1 — Variance Threshold
    selector = VarianceThreshold(threshold=0.01)
    X = pd.DataFrame(selector.fit_transform(X),
                     columns=X.columns[selector.get_support()])
    print(f"After Variance Threshold: {X.shape[1]} features.")

    # Step 2 — Correlation Filter
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper_triangle.columns
                      if any(upper_triangle[col] > 0.90)]
    X = X.drop(columns=high_corr_cols)
    print(f"After Correlation Filter: {X.shape[1]} features.")
    if high_corr_cols:
        print(f"Dropped highly correlated: {high_corr_cols}")

    # Step 3 — SelectKBest
    k = min(k, X.shape[1])
    if problem_type == 'classification':
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)

    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_cols)
    print(f"After SelectKBest: {X.shape[1]} features.")
    print(f"Selected features: {selected_cols}")

    return X, selected_cols


# ─────────────────────────────────────────────
# STEP 5 — SCALING
# ─────────────────────────────────────────────

def scale_features(X_train, X_test, model_type='tree'):
    """
    Scales numerical features.

    Why scaling matters:
    Distance-based models (KNN, SVM) and linear models
    (Logistic Regression) are sensitive to feature scale.
    A feature with range 0-1000 dominates one with range 0-1.

    Why tree models don't need scaling:
    Decision trees split on thresholds, not distances.
    Scale doesn't affect the split logic.

    Scalers:
    - StandardScaler: mean=0, std=1. Best for linear models.
    - MinMaxScaler: range 0-1. Best for neural nets.
    - RobustScaler: uses median and IQR. Best when outliers present.
    """
    if model_type == 'tree':
        print("Tree-based model — skipping scaling.")
        return X_train, X_test, None

    # Use RobustScaler as default — handles outliers well
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns)
    print("Applied RobustScaler to features.")
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────
# STEP 6 — TRAIN TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X, y, problem_type, test_size=0.2, random_state=42):
    """
    Splits data into training and test sets.

    Why 80/20 split:
    80% for training gives model enough data to learn.
    20% for testing gives reliable performance estimate.

    Why stratify for classification:
    Ensures both train and test have same class ratio.
    Without stratify, test set might have no minority class samples.

    Why random_state=42:
    Makes the split reproducible. Anyone running
    your code gets the same split every time.

    Data leakage warning:
    Never fit scalers or encoders on the full dataset.
    Always fit on train set only, transform both train and test.
    Fitting on full data = your model has seen test data = cheating.
    """
    if problem_type == 'classification':
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        print(f"Stratified split applied.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    print(f"Train size: {X_train.shape[0]} rows")
    print(f"Test size: {X_test.shape[0]} rows")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STEP 7 — HANDLE CLASS IMBALANCE (SMOTE)
# ─────────────────────────────────────────────

def handle_imbalance(X_train, y_train, problem_type, quality_report):
    """
    Applies SMOTE if class imbalance is detected.

    What SMOTE does:
    Creates synthetic samples of minority class by
    interpolating between existing minority samples.
    Not just duplicating — creating new realistic samples.

    Why only on training data:
    Never apply SMOTE to test data.
    Test data must reflect real world distribution.
    """
    if problem_type != 'classification':
        return X_train, y_train

    minority_pct = (y_train.value_counts().min() /
                    len(y_train)) * 100

    if minority_pct < 20:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied. New training size: {X_train.shape[0]}")
            print(f"New class distribution: {pd.Series(y_train).value_counts().to_dict()}")
        except ImportError:
            print("imbalanced-learn not installed. Skipping SMOTE.")
            print("Install with: pip install imbalanced-learn")
    else:
        print(f"Class balance acceptable ({minority_pct:.1f}%). SMOTE not needed.")

    return X_train, y_train


# ─────────────────────────────────────────────
# MASTER FUNCTION
# ─────────────────────────────────────────────

def preprocess(df, target_col, col_types, problem_type, quality_report):
    """
    Runs the full preprocessing pipeline in order.
    Returns train/test splits ready for model training.
    """
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE")
    print("="*50)

    # Step 1 — Drop useless columns
    print("\n[Step 1] Dropping useless columns...")
    df, dropped_cols = drop_useless_columns(df, target_col, col_types)

    # Step 2 — Handle missing values
    print("\n[Step 2] Handling missing values...")
    df = handle_missing_values(df, target_col, col_types)

    # Step 3 — Encode categoricals
    print("\n[Step 3] Encoding categorical columns...")
    df, label_encoded, ohe_encoded = encode_categoricals(
        df, target_col, col_types, problem_type)

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Step 4 — Feature selection
    print("\n[Step 4] Selecting features...")
    X, selected_features = select_features(X, y, problem_type)

    # Step 5 — Train test split
    print("\n[Step 5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y, problem_type)

    # Step 6 — Scale features
    print("\n[Step 6] Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Step 7 — Handle class imbalance
    print("\n[Step 7] Handling class imbalance...")
    X_train, y_train = handle_imbalance(
        X_train, y_train, problem_type, quality_report)

    print("\n" + "="*50)
    print("PREPROCESSING COMPLETE")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print("="*50)

    return X_train, X_test, y_train, y_test, selected_features, scaler