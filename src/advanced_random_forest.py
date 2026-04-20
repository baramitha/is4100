# SCRUM-20: Advanced Model — Random Forest
# Assignee: Naura
# Sprint: 2
# Epic: SCRUM-7 Model Development
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
 
 
# Definition of Done: F1 > 0.75
F1_THRESHOLD = 0.75
 
 
def train_advanced_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    SCRUM-20: Train an advanced Random Forest model with optimized settings.
    Uses class_weight='balanced' to handle churn class imbalance.
    """
    print("\n========== SCRUM-20: ADVANCED RANDOM FOREST ==========")
 
    model = RandomForestClassifier(
        n_estimators=200,        # more trees = more stable predictions
        max_depth=15,            # deeper trees for complex churn patterns
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",     # standard for classification
        class_weight="balanced", # handles churn imbalance
        random_state=42,
        n_jobs=-1                # use all CPU cores
    )
 
    model.fit(X_train, y_train)
    print("✅ Advanced Random Forest trained")
    print(f"   Trees: {model.n_estimators}")
    print(f"   Max depth: {model.max_depth}")
    print(f"   Features per split: sqrt({X_train.shape[1]}) = {model.max_features}")
 
    # Quick F1 check on training data
    train_f1 = f1_score(y_train, model.predict(X_train))
    print(f"   Training F1: {train_f1:.4f}")
 
    print("=" * 55)
    return model
 
 
def get_feature_importance(model, feature_names: list, top_n: int = 10):
    """
    SCRUM-20: Extract and display top N most important features.
    """
    print(f"\n🔍 Top {top_n} Features for Churn Prediction:")
 
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(top_n)
 
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["Importance"] * 200)
        print(f"   {row['Feature']:<35} {row['Importance']:.4f}  {bar}")
 
    return importance_df
 
 
if __name__ == "__main__":
    from src.data_exploration import load_data
    from src.data_cleaning import clean_data
    from src.feature_engineering import engineer_features
    from src.train_test_split import split_data
 
    df = load_data("data/raw/churn.csv")
    df = clean_data(df)
    X, y, feature_names, scaler = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
 
    model = train_advanced_random_forest(X_train, y_train)
    get_feature_importance(model, feature_names)
