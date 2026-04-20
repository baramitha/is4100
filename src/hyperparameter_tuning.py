# SCRUM-21: Hyperparameter Tuning
# Assignee: Naura
# Sprint: 2
# Epic: SCRUM-7 Model Development
# Story Points: 8 (highest effort — grid search is computationally expensive)
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
 
 
# Definition of Done: F1 > 0.75
F1_THRESHOLD = 0.75
f1_scorer = make_scorer(f1_score)
 
 
def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    SCRUM-21: Tune Random Forest hyperparameters using GridSearchCV.
    Uses 5-fold cross validation, optimising for F1 score.
    """
    print("\n========== SCRUM-21: HYPERPARAMETER TUNING (Random Forest) ==========")
    print("⏳ Running GridSearchCV — this may take a few minutes...")
 
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced"]
    }
 
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
 
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross validation
        scoring="f1",            # optimise for F1 (DoD metric)
        n_jobs=-1,
        verbose=1
    )
 
    grid_search.fit(X_train, y_train)
 
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
 
    print(f"\n✅ Best Parameters Found:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
 
    print(f"\n📊 Best Cross-Validation F1: {best_score:.4f}")
 
    if best_score >= F1_THRESHOLD:
        print(f"✅ DoD PASSED — F1={best_score:.4f} > {F1_THRESHOLD}")
    else:
        print(f"❌ DoD FAILED — F1={best_score:.4f} < {F1_THRESHOLD}")
 
    print("=" * 65)
    return grid_search.best_estimator_
 
 
def tune_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    SCRUM-21: Tune Logistic Regression hyperparameters.
    """
    print("\n========== SCRUM-21: HYPERPARAMETER TUNING (Logistic Regression) ==========")
    print("⏳ Running GridSearchCV...")
 
    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "class_weight": ["balanced"]
    }
 
    lr = LogisticRegression(max_iter=1000, random_state=42)
 
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )
 
    grid_search.fit(X_train, y_train)
 
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
 
    print(f"\n✅ Best Parameters Found:")
    for param, value in best_params.items():
        print(f"   {param}: {value}")
 
    print(f"\n📊 Best Cross-Validation F1: {best_score:.4f}")
 
    if best_score >= F1_THRESHOLD:
        print(f"✅ DoD PASSED — F1={best_score:.4f} > {F1_THRESHOLD}")
    else:
        print(f"❌ DoD FAILED — F1={best_score:.4f} < {F1_THRESHOLD}")
 
    print("=" * 65)
    return grid_search.best_estimator_
 
 
def compare_tuning_results(base_model, tuned_model, X_test, y_test, model_name: str):
    """
    SCRUM-21: Compare base vs tuned model performance.
    """
    print(f"\n📊 Tuning Impact — {model_name}:")
    print(f"{'Metric':<20} {'Before Tuning':<20} {'After Tuning':<20}")
    print("-" * 60)
 
    base_f1 = f1_score(y_test, base_model.predict(X_test))
    tuned_f1 = f1_score(y_test, tuned_model.predict(X_test))
 
    improvement = ((tuned_f1 - base_f1) / base_f1) * 100
    print(f"{'F1 Score':<20} {base_f1:<20.4f} {tuned_f1:<20.4f}")
    print(f"\n📈 Improvement: {improvement:+.2f}%")
 
    if tuned_f1 >= F1_THRESHOLD:
        print(f"✅ Tuned model meets Definition of Done (F1 > {F1_THRESHOLD})")
    else:
        print(f"❌ Still below threshold — consider more features or different algorithm")
 
 
if __name__ == "__main__":
    from src.data_exploration import load_data
    from src.data_cleaning import clean_data
    from src.feature_engineering import engineer_features
    from src.train_test_split import split_data
    from src.model import train_random_forest, train_logistic_regression
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    # Base models
    base_rf = train_random_forest(X_train, y_train)
    base_lr = train_logistic_regression(X_train, y_train)
 
    # Tuned models
    tuned_rf = tune_random_forest(X_train, y_train)
    tuned_lr = tune_logistic_regression(X_train, y_train)
 
    # Compare
    compare_tuning_results(base_rf, tuned_rf, X_test, y_test, "Random Forest")
    compare_tuning_results(base_lr, tuned_lr, X_test, y_test, "Logistic Regression")
