import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, mean_squared_error, r2_score,
                              mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')


def get_models(problem_type):
    """
    Returns a dictionary of models to train.
    One model from each ML family:

    - Linear: Logistic/Linear Regression
      Simple, fast, interpretable baseline.
      Works well when relationship between
      features and target is linear.

    - Tree: Decision Tree
      Interpretable, handles nonlinearity.
      Prone to overfitting without depth limit.

    - Ensemble Bagging: Random Forest
      Multiple trees trained on random subsets.
      Reduces overfitting via averaging.

    - Ensemble Boosting: Gradient Boosting
      Trees trained sequentially, each fixing
      errors of the previous. Usually best performer.

    - Kernel: SVM
      Finds optimal decision boundary.
      Strong on smaller, clean datasets.

    - Instance Based: KNN
      Predicts based on k nearest neighbors.
      Simple but sensitive to scale and noise.

    - Probabilistic: Naive Bayes (classification only)
      Fast, works well on text and small data.
      Assumes feature independence.
    """
    if problem_type == 'classification':
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(
                random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB()
        }
    else:
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(
                random_state=42),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor()
        }
    return models


def get_scoring_metric(problem_type, y_train):
    """
    Selects the right evaluation metric automatically.

    Classification:
    - Imbalanced (minority < 20%) → F1 score
      Accuracy is misleading on imbalanced data.
      A model predicting majority class always
      gets high accuracy but is useless.
    - Balanced → Accuracy + F1

    Regression:
    - RMSE (Root Mean Squared Error)
      Penalizes large errors more than small ones.
    - R² (coefficient of determination)
      How much variance in target your model explains.
      R²=1 is perfect. R²=0 means model learned nothing.
    """
    if problem_type == 'classification':
        minority_pct = (pd.Series(y_train).value_counts().min()
                        / len(y_train)) * 100
        if minority_pct < 20:
            metric = 'f1'
            print("Imbalanced data detected — using F1 score.")
        else:
            metric = 'f1_weighted'
            print("Balanced data — using weighted F1 score.")
    else:
        metric = 'r2'
        print("Regression problem — using R² score.")

    return metric


def run_cross_validation(models, X_train, y_train,
                         problem_type, scoring_metric):
    """
    Runs 5-fold cross validation for each model.

    What is cross validation:
    Instead of one train/test split, divide training
    data into 5 equal parts (folds).
    Train on 4 folds, test on 1 fold.
    Repeat 5 times, each time using a different fold as test.
    Average the 5 scores.

    Why cross validation over single split:
    Single split score depends heavily on which samples
    ended up in test set — high variance.
    Cross validation gives more reliable estimate
    by testing on all samples at least once.

    Why StratifiedKFold for classification:
    Ensures each fold has same class ratio as full dataset.
    Prevents a fold from having no minority class samples.
    """
    print("\nRunning 5-fold cross validation...")
    print("-" * 50)

    cv_results = {}

    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring=scoring_metric
        )
        mean_score = round(scores.mean(), 4)
        std_score = round(scores.std(), 4)
        cv_results[name] = {
            "mean_score": mean_score,
            "std_score": std_score,
            "scores": scores.tolist()
        }
        print(f"{name:25s} | {scoring_metric}: "
              f"{mean_score:.4f} ± {std_score:.4f}")

    return cv_results


def train_best_model(models, cv_results, X_train, y_train):
    """
    Trains the best model on full training data.

    Why retrain on full data after cross validation:
    Cross validation only evaluates performance.
    The model trained during CV used only 80% of
    training data each time.
    Final model should use all available training data
    to maximize learning before deployment.
    """
    best_model_name = max(cv_results,
                          key=lambda x: cv_results[x]["mean_score"])
    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)

    print(f"\nBest model: {best_model_name}")
    print(f"CV Score: {cv_results[best_model_name]['mean_score']:.4f}"
          f" ± {cv_results[best_model_name]['std_score']:.4f}")

    return best_model, best_model_name


def evaluate_on_test(best_model, best_model_name,
                     X_test, y_test, problem_type):
    """
    Final evaluation on held-out test set.

    This is the true performance estimate.
    Test set was never seen during training or CV.

    Classification metrics:
    - Accuracy: correct predictions / total predictions
    - Precision: of all predicted positive, how many are actually positive
    - Recall: of all actual positive, how many did we catch
    - F1: harmonic mean of precision and recall
      Use when false positives and false negatives both matter.

    Regression metrics:
    - MAE: average absolute error. Easy to interpret.
    - RMSE: penalizes large errors more. Sensitive to outliers.
    - R²: proportion of variance explained. Higher is better.
    """
    y_pred = best_model.predict(X_test)

    if problem_type == 'classification':
        test_metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(
                y_test, y_pred, average='weighted'), 4),
            "recall": round(recall_score(
                y_test, y_pred, average='weighted'), 4),
            "f1": round(f1_score(
                y_test, y_pred, average='weighted'), 4)
        }
    else:
        mse = mean_squared_error(y_test, y_pred)
        test_metrics = {
            "mae": round(mean_absolute_error(y_test, y_pred), 4),
            "rmse": round(np.sqrt(mse), 4),
            "r2": round(r2_score(y_test, y_pred), 4)
        }

    print(f"\nTest Set Metrics for {best_model_name}:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value}")

    return test_metrics


def build_leaderboard(cv_results):
    """
    Creates a sorted leaderboard of all models.
    Higher score = better model.
    """
    leaderboard = []
    for name, result in cv_results.items():
        leaderboard.append({
            "model": name,
            "mean_score": result["mean_score"],
            "std_score": result["std_score"]
        })

    leaderboard = sorted(leaderboard,
                         key=lambda x: x["mean_score"],
                         reverse=True)

    print("\nModel Leaderboard:")
    print("-" * 50)
    print(f"{'Rank':<6}{'Model':<25}{'Score':<12}{'Std'}")
    print("-" * 50)
    for i, entry in enumerate(leaderboard, 1):
        print(f"{i:<6}{entry['model']:<25}"
              f"{entry['mean_score']:<12}{entry['std_score']}")

    return leaderboard


def run_training(X_train, X_test, y_train, y_test, problem_type):
    """
    Master training function.
    Call this from app.py
    """
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)

    # Get models
    models = get_models(problem_type)

    # Get scoring metric
    scoring_metric = get_scoring_metric(problem_type, y_train)

    # Cross validation
    cv_results = run_cross_validation(
        models, X_train, y_train, problem_type, scoring_metric)

    # Build leaderboard
    leaderboard = build_leaderboard(cv_results)

    # Train best model on full training data
    best_model, best_model_name = train_best_model(
        models, cv_results, X_train, y_train)

    # Evaluate on test set
    test_metrics = evaluate_on_test(
        best_model, best_model_name,
        X_test, y_test, problem_type)

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)

    return best_model, best_model_name, cv_results, leaderboard, test_metrics