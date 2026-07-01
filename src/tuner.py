import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')


def get_param_grids(problem_type):
    """
    Defines hyperparameter search spaces for each model.

    What are hyperparameters:
    Parameters you set BEFORE training starts.
    The model cannot learn these from data.
    You have to search for the best values manually.

    Examples:
    - n_estimators: how many trees in Random Forest
    - max_depth: how deep each tree can grow
    - learning_rate: how fast model updates weights
    - C: regularization strength in SVM

    GridSearchCV vs RandomizedSearchCV:
    - GridSearchCV: tries EVERY combination in the grid.
      Exhaustive but slow on large grids.
      Use when search space is small.
    - RandomizedSearchCV: tries RANDOM sample of combinations.
      Faster, good enough on large search spaces.
      Use when search space is large.

    Strategy here:
    - Small grids → GridSearchCV
    - Large grids → RandomizedSearchCV
    """
    if problem_type == 'classification':
        param_grids = {
            "Logistic Regression": {
                "model": LogisticRegression(
                    max_iter=1000, random_state=42),
                "params": {
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"]
                },
                "search": "grid"
            },
            "Decision Tree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["gini", "entropy"]
                },
                "search": "grid"
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200, 300],
                    "max_depth": [3, 5, 7, None],
                    "min_samples_split": [2, 5, 10],
                    "max_features": ["sqrt", "log2"]
                },
                "search": "random"
            },
            "Gradient Boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5],
                    "subsample": [0.7, 0.8, 1.0]
                },
                "search": "random"
            },
            "SVM": {
                "model": SVC(probability=True, random_state=42),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto"]
                },
                "search": "grid"
            },
            "KNN": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan"]
                },
                "search": "grid"
            },
            "Naive Bayes": {
                "model": GaussianNB(),
                "params": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
                },
                "search": "grid"
            }
        }
    else:
        param_grids = {
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {},
                "search": "grid"
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(random_state=42),
                "params": {
                    "max_depth": [3, 5, 7, 10, None],
                    "min_samples_split": [2, 5, 10]
                },
                "search": "grid"
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [3, 5, 7, None],
                    "min_samples_split": [2, 5, 10]
                },
                "search": "random"
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 4, 5]
                },
                "search": "random"
            },
            "SVM": {
                "model": SVR(),
                "params": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"]
                },
                "search": "grid"
            },
            "KNN": {
                "model": KNeighborsRegressor(),
                "params": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "search": "grid"
            }
        }

    return param_grids


def tune_model(name, config, X_train, y_train,
               problem_type, scoring_metric):
    """
    Tunes a single model using GridSearchCV or RandomizedSearchCV.

    How GridSearchCV works:
    1. Takes your parameter grid
    2. Creates all possible combinations
    3. Trains and evaluates each combination
       using cross validation
    4. Returns the combination with best score

    How RandomizedSearchCV works:
    1. Takes your parameter distributions
    2. Randomly samples n_iter combinations
    3. Trains and evaluates each
    4. Returns the best one found

    n_iter=20 means try 20 random combinations.
    More iterations = better search but slower.
    """
    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=5, shuffle=True,
                             random_state=42)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Skip tuning if no params defined
    if not config["params"]:
        config["model"].fit(X_train, y_train)
        return config["model"], {}, 0.0

    if config["search"] == "grid":
        searcher = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0
        )
    else:
        searcher = RandomizedSearchCV(
            estimator=config["model"],
            param_distributions=config["params"],
            n_iter=20,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=42
        )

    searcher.fit(X_train, y_train)

    print(f"{name:25s} | Best score: "
          f"{searcher.best_score_:.4f} | "
          f"Params: {searcher.best_params_}")

    return searcher.best_estimator_, searcher.best_params_, searcher.best_score_


def run_tuning(cv_results, X_train, y_train,
               problem_type, scoring_metric, top_n=3):
    """
    Tunes only the top N models from training phase.

    Why not tune all models:
    Tuning is expensive — especially GridSearch
    on large grids. Tuning only top performers
    saves time without losing quality.

    top_n=3 means tune only top 3 models
    from the leaderboard.
    """
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)

    # Get top N model names from cv_results
    sorted_models = sorted(cv_results.items(),
                           key=lambda x: x[1]["mean_score"],
                           reverse=True)
    top_model_names = [name for name, _ in sorted_models[:top_n]]
    print(f"\nTuning top {top_n} models: {top_model_names}")

    # Get param grids
    param_grids = get_param_grids(problem_type)

    tuned_results = {}

    for name in top_model_names:
        if name not in param_grids:
            print(f"Skipping {name} — no param grid defined.")
            continue

        print(f"\nTuning: {name}")
        config = param_grids[name]

        best_estimator, best_params, best_score = tune_model(
            name, config, X_train, y_train,
            problem_type, scoring_metric)

        tuned_results[name] = {
            "model": best_estimator,
            "best_params": best_params,
            "best_score": round(best_score, 4)
        }

    # Find overall best tuned model
    best_tuned_name = max(tuned_results,
                          key=lambda x: tuned_results[x]["best_score"])
    best_tuned_model = tuned_results[best_tuned_name]["model"]

    print("\n" + "-"*50)
    print("Tuning Results Summary:")
    print("-"*50)
    for name, result in tuned_results.items():
        print(f"{name:25s} | "
              f"Tuned Score: {result['best_score']:.4f}")

    print(f"\nBest tuned model: {best_tuned_name}")
    print(f"Best tuned score: "
          f"{tuned_results[best_tuned_name]['best_score']:.4f}")

    print("\n" + "="*50)
    print("TUNING COMPLETE")
    print("="*50)

    return best_tuned_model, best_tuned_name, tuned_results