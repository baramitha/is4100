# IS4100 – IT Project Management
# Agile/Scrum ML Pipeline — Customer Churn Prediction
# Team: Group 4 | Darryl (Scrum Master) + Naura (Developer)
#
# Run this file to execute the full pipeline end-to-end
# All steps are linked to their Jira tickets (SCRUM-XX)
 
from src.data_exploration import load_data, explore_data       # SCRUM-14
from src.data_quality import assess_quality                     # SCRUM-15
from src.data_cleaning import clean_data, save_clean_data      # SCRUM-16
from src.feature_engineering import engineer_features           # SCRUM-17
from src.train_test_split import split_data, validate_split    # SCRUM-18
from src.model import (                                         # SCRUM-7
    train_logistic_regression,
    train_random_forest,
    evaluate_model,
    compare_models,
    get_feature_importance
)
 
 
def run_pipeline(data_path: str = "data/raw/churn.csv"):
    print("=" * 60)
    print("  IS4100 GROUP 4 — CUSTOMER CHURN ML PIPELINE")
    print("=" * 60)
 
    # SCRUM-14: Data Exploration
    print("\n[SCRUM-14] Data Exploration & Profiling")
    df = load_data(data_path)
    explore_data(df)
 
    # SCRUM-15: Data Quality Assessment
    print("\n[SCRUM-15] Data Quality Assessment")
    assess_quality(df)
 
    # SCRUM-16: Data Cleaning
    print("\n[SCRUM-16] Data Cleaning")
    df_clean = clean_data(df)
    save_clean_data(df_clean)
 
    # SCRUM-17: Feature Engineering
    print("\n[SCRUM-17] Feature Engineering")
    X, y, feature_names, scaler = engineer_features(df_clean)
 
    # SCRUM-18: Train/Test Split
    print("\n[SCRUM-18] Train/Test Split & Validation")
    X_train, X_test, y_train, y_test = split_data(X, y)
    validate_split(X_train, X_test, y_train, y_test)
 
    # SCRUM-7: Model Training & Evaluation
    print("\n[SCRUM-7] Model Development & Evaluation")
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
 
    results_lr = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    results_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")
 
    compare_models(results_lr, results_rf)
    get_feature_importance(rf_model, feature_names, "Random Forest")
 
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE ✅")
    print("=" * 60)
 
    return {
        "logistic_regression": results_lr,
        "random_forest": results_rf
    }
 
 
if __name__ == "__main__":
    run_pipeline()
